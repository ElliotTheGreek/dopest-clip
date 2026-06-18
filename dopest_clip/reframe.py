"""Portrait reframe with a SHOT TIMELINE.

The agent decides, across the clip's output timeline, when to zoom to the subject and when to
pull out wide. Every shot resolves to a per-frame crop rectangle; the compositor then fills
the target frame when the crop is portrait-aspect, or letterboxes it into a centre band with
blurred top/bottom bands when the crop is wider (the "full-band" look). Transitions between
shots lerp the crop rect, which reads as a smooth zoom in/out.

EDL: reframe = {
  "aspect": "9:16",
  "transition_s": 0.5,
  "shots": [ {"start": 0.0, "mode": "full"},
             {"start": 5.0, "mode": "track", "zoom": 1.15},
             {"start": 12.0, "mode": "full"} ]
}
Backward compatible: reframe = {"mode": "track"|"full"|"center"|"pad"|"focus", "zoom": {...}}
with no `shots` is treated as one shot spanning the whole clip.

Subject tracking is smoothed with a One-Euro filter + deadzone + pan-speed cap (config.
REFRAME_*). track needs a visible person; frames with no detection hold the last centre.

Heavy deps (ultralytics, cv2, numpy) are imported LAZILY inside the functions that need
them, so importing this module costs nothing and never pulls torch/cv2/ultralytics. The
One-Euro filter and crop-rect math below are pure Python and unit-testable directly.
"""

import math
import subprocess
import tempfile
from pathlib import Path

from . import config, media

_MODEL = None


def _resolve_model_path() -> str:
    """Resolve config.REFRAME_MODEL to a usable path. If it is the bare bundled default
    'yolo11n.pt', load it from the package assets dir where it ships; otherwise honour
    the configured value verbatim (absolute path or ultralytics-known name)."""
    model = config.REFRAME_MODEL
    if model == "yolo11n.pt":
        bundled = config.ASSETS_DIR / "yolo11n.pt"
        if not bundled.exists():
            raise FileNotFoundError(
                f"bundled reframe model not found at {bundled} — set REFRAME_MODEL to a model path"
            )
        return str(bundled)
    return model


def _load_model():
    global _MODEL
    if _MODEL is None:
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "reframe subject tracking needs 'ultralytics' (and torch). "
                "Install it: pip install ultralytics"
            ) from e
        _MODEL = YOLO(_resolve_model_path())
    return _MODEL


class _OneEuro:
    """One-Euro low-pass filter. Pure math — no heavy deps."""

    def __init__(self, freq, mincutoff, beta, dcutoff=1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self._x_prev = None
        self._dx_prev = 0.0

    @staticmethod
    def _alpha(cutoff, freq):
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x):
        if self._x_prev is None:
            self._x_prev = x
            return x
        dx = (x - self._x_prev) * self.freq
        a_d = self._alpha(self.dcutoff, self.freq)
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, self.freq)
        x_hat = a * x + (1 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


def _base_crop(sw, sh, tw, th):
    """Largest crop of source (sw x sh) with the target aspect (tw/th)."""
    ar = tw / th
    if ar <= sw / sh:
        return int(round(sh * ar)), sh
    return sw, int(round(sw / ar))


def _smooth_centers(centers, sw, sh, fps):
    """Deadzone + One-Euro + pan-speed cap over raw per-frame centres. Pure math."""
    dead = config.REFRAME_DEADZONE * sw
    max_step = config.REFRAME_MAX_PAN * sw / max(fps, 1.0)
    fx = _OneEuro(fps, config.REFRAME_MINCUTOFF, config.REFRAME_BETA)
    fy = _OneEuro(fps, config.REFRAME_MINCUTOFF, config.REFRAME_BETA)
    out = []
    committed_x = committed_y = None
    prev_sx = prev_sy = None
    last = (sw / 2, sh / 2)
    for c in centers:
        raw = c if c else last
        last = raw
        rx, ry = raw
        if committed_x is None:
            committed_x, committed_y = rx, ry
        if abs(rx - committed_x) > dead:
            committed_x = rx
        if abs(ry - committed_y) > dead:
            committed_y = ry
        sx, sy = fx(committed_x), fy(committed_y)
        if prev_sx is not None:
            sx = prev_sx + max(-max_step, min(max_step, sx - prev_sx))
            sy = prev_sy + max(-max_step, min(max_step, sy - prev_sy))
        prev_sx, prev_sy = sx, sy
        out.append((sx, sy))
    return out


def _clamp_rect(cx, cy, w, h, sw, sh):
    w = min(w, sw)
    h = min(h, sh)
    x = max(0, min(sw - w, cx - w / 2))
    y = max(0, min(sh - h, cy - h / 2))
    return [x, y, w, h]


def _shot_rect(shot, i, smoothed, sw, sh, base_cw, base_ch):
    """Resolve one shot to a crop rect at frame index i. Pure math."""
    mode = shot.get("mode", "full")
    z = max(1.0, float(shot.get("zoom") or 1.0))
    # explicit crop rect wins (agent picked pixels off a grabbed frame)
    crop = shot.get("crop")
    if crop:
        return _clamp_rect(crop["x"] + crop["w"] / 2, crop["y"] + crop["h"] / 2,
                           crop["w"], crop["h"], sw, sh)
    if mode == "focus":
        cw, ch = base_cw / z, base_ch / z
        cx = float(shot["x"]) if shot.get("x") is not None else sw / 2
        cy = float(shot["y"]) if shot.get("y") is not None else sh / 2
        if ch >= sh - 1:  # full-height crop can only pan horizontally
            cy = sh / 2
        return _clamp_rect(cx, cy, cw, ch, sw, sh)
    if mode in ("track", "zoom"):
        cx, cy = smoothed[i] if i < len(smoothed) else (sw / 2, sh / 2)
        cw, ch = base_cw / z, base_ch / z
        cy = sh / 2 if ch >= sh - 1 else cy
        return _clamp_rect(cx, cy, cw, ch, sw, sh)
    if mode == "center":
        return _clamp_rect(sw / 2, sh / 2, base_cw, base_ch, sw, sh)
    # "full" / "pad" (and anything else): the entire source frame -> letterboxed band
    return [0.0, 0.0, float(sw), float(sh)]


def _smoothstep(a):
    return a * a * (3 - 2 * a)


def _normalize_shots(shots, default_zoom):
    norm = sorted(
        [{"start": float(s.get("start", 0.0)), "mode": s.get("mode", "full"),
          "zoom": s.get("zoom", default_zoom), "x": s.get("x"), "y": s.get("y"),
          "crop": s.get("crop")} for s in shots],
        key=lambda s: s["start"],
    )
    return norm or [{"start": 0.0, "mode": "full", "zoom": default_zoom, "x": None, "y": None, "crop": None}]


def _detect_center(model, frame, conf, classes=None, center_frac=0.33):
    """Largest detection's centre. classes=None -> [0] (person); center_frac biases the
    y toward the top of the box (0.33 ~ head for a person; pass 0.5 for object centre).
    Returns (cx, cy) or None. (Used by reframe's subject track AND by obs.tracking.)"""
    cls = [0] if classes is None else classes
    res = model(frame, classes=cls, conf=conf, verbose=False, device=0)
    best = None
    for r in res:
        if r.boxes is None:
            continue
        for b in r.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            area = (x2 - x1) * (y2 - y1)
            if best is None or area > best[2]:
                best = ((x1 + x2) / 2, y1 + (y2 - y1) * center_frac, area, x2 - x1, y2 - y1)
    return (best[0], best[1]) if best else None


def _detect_box(model, frame, conf, classes, center_frac=0.5):
    """Like _detect_center but also returns the box size: (cx, cy, w, h) or None.
    Used by obs.tracking to size a ring/zoom to the tracked object."""
    cls = classes if classes else [0]
    res = model(frame, classes=cls, conf=conf, verbose=False, device=0)
    best = None
    for r in res:
        if r.boxes is None:
            continue
        for b in r.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            area = (x2 - x1) * (y2 - y1)
            if best is None or area > best[2]:
                best = ((x1 + x2) / 2, y1 + (y2 - y1) * center_frac, area, x2 - x1, y2 - y1)
    return (best[0], best[1], best[3], best[4]) if best else None


def _apply_censor(frame, regions, t):
    """Redact sensitive rects (SOURCE pixels) on a frame, in place, before any crop.
    Each region: {x,y,w,h, style?:'blur'|'pixelate'|'box', start?,end? (output seconds)}."""
    import cv2

    if not regions:
        return frame
    h0, w0 = frame.shape[:2]
    for r in regions:
        st, en = r.get("start"), r.get("end")
        if st is not None and t < st:
            continue
        if en is not None and t > en:
            continue
        x = int(max(0, r["x"]))
        y = int(max(0, r["y"]))
        w = int(min(w0 - x, r["w"]))
        h = int(min(h0 - y, r["h"]))
        if w <= 1 or h <= 1:
            continue
        roi = frame[y:y + h, x:x + w]
        style = r.get("style", "blur")
        if style == "box":
            frame[y:y + h, x:x + w] = 0
        elif style == "pixelate":
            small = cv2.resize(roi, (max(1, w // 16), max(1, h // 16)), interpolation=cv2.INTER_LINEAR)
            frame[y:y + h, x:x + w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        else:  # heavy blur
            k = max(21, (min(w, h) // 3) | 1)
            frame[y:y + h, x:x + w] = cv2.GaussianBlur(roi, (k, k), 0)
    return frame


def build_plan(cut_video: Path, tw: int, th: int, shots: list, transition_s: float, default_zoom):
    """Return (per-frame crop rects, fps, sw, sh). Reads frames via cv2 (lazy)."""
    import cv2

    cap = cv2.VideoCapture(str(cut_video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {cut_video}")
    sw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    sh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    base_cw, base_ch = _base_crop(sw, sh, tw, th)

    needs_track = any(s.get("mode") in ("track", "zoom") for s in shots)
    centers = []
    n = 0
    if needs_track:
        model = _load_model()
        sample = max(1, config.REFRAME_SAMPLE_EVERY)
        last = None
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % sample == 0:
                d = _detect_center(model, frame, config.REFRAME_CONF)
                last = d or last
            centers.append(last)
            idx += 1
        n = idx
    else:
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            n += 1
    cap.release()

    smoothed = _smooth_centers(centers, sw, sh, fps) if needs_track else []

    shots = _normalize_shots(shots, default_zoom)

    rects = []
    for i in range(max(n, 1)):
        t = i / fps
        ai = 0
        for k, s in enumerate(shots):
            if s["start"] <= t:
                ai = k
        cur = shots[ai]
        cur_rect = _shot_rect(cur, i, smoothed, sw, sh, base_cw, base_ch)
        if ai > 0 and transition_s > 0 and t < cur["start"] + transition_s:
            prev_rect = _shot_rect(shots[ai - 1], i, smoothed, sw, sh, base_cw, base_ch)
            a = _smoothstep((t - cur["start"]) / transition_s)
            cur_rect = [p + (c - p) * a for p, c in zip(prev_rect, cur_rect)]
        rects.append([int(round(v)) for v in cur_rect])
    return rects, fps, sw, sh


def _composite(frame, rect, tw, th, blur_k):
    import cv2

    x, y, w, h = rect
    w = max(2, min(w, frame.shape[1] - x))
    h = max(2, min(h, frame.shape[0] - y))
    crop = frame[y:y + h, x:x + w]
    new_h = max(1, int(round(tw * h / w)))
    resized = cv2.resize(crop, (tw, new_h), interpolation=cv2.INTER_AREA)
    if new_h >= th:
        off = (new_h - th) // 2
        return resized[off:off + th, 0:tw]
    bg = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_LINEAR)
    bg = cv2.GaussianBlur(bg, (blur_k, blur_k), 0)
    top = (th - new_h) // 2
    bg[top:top + new_h, 0:tw] = resized
    return bg


def preview_frame(cut_video, reframe_cfg: dict, at: float, tw: int, th: int, out_png: Path, censor=None):
    """Composite ONE output frame for the reframe plan at time `at` so the agent can see what
    the viewer sees and iterate on framing without a full render. Returns the crop rect used."""
    import cv2

    cap = cv2.VideoCapture(str(cut_video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {cut_video}")
    sw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    sh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    base_cw, base_ch = _base_crop(sw, sh, tw, th)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(at * fps))))
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("could not read a frame for preview")

    zoomv = reframe_cfg.get("zoom")
    dz = zoomv.get("factor", 1.0) if isinstance(zoomv, dict) else (zoomv or 1.0)
    single = {"start": 0.0, "mode": reframe_cfg.get("mode", "full"), "x": reframe_cfg.get("x"),
              "y": reframe_cfg.get("y"), "zoom": reframe_cfg.get("zoom"), "crop": reframe_cfg.get("crop")}
    shots = _normalize_shots(reframe_cfg.get("shots") or [single], dz)
    ai = 0
    for k, s in enumerate(shots):
        if s["start"] <= at:
            ai = k
    shot = shots[ai]

    smoothed = []
    if shot["mode"] in ("track", "zoom"):
        d = _detect_center(_load_model(), frame, config.REFRAME_CONF)
        smoothed = [d or (sw / 2, sh / 2)]
    if censor:
        _apply_censor(frame, censor, at)
    rect = [int(round(v)) for v in _shot_rect(shot, 0, smoothed, sw, sh, base_cw, base_ch)]
    comp = _composite(frame, rect, tw, th, config.REFRAME_BAND_BLUR | 1)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), comp)
    return rect


def apply_reframe(cut_video, rects, fps, tw, th, out_video, ass_path=None, loudnorm=False, fonts_dir=None, censor=None):
    """Encode the reframed clip: pipe composited RGB frames to ffmpeg, mux the cut's audio,
    burn captions, optional loudnorm. ffmpeg is invoked directly here (not via media.run_ff)
    because this is a streaming pipe, not a single fixed-arg command — stdin is the frame pipe."""
    import cv2

    out_video.parent.mkdir(parents=True, exist_ok=True)
    blur_k = config.REFRAME_BAND_BLUR | 1  # force odd

    vf_chains = []
    if ass_path is not None:
        sub = f"subtitles=filename='{media.escape_filter_path(ass_path)}'"
        if fonts_dir is not None:
            sub += f":fontsdir='{media.escape_filter_path(fonts_dir)}'"
        vf_chains.append(sub)
    vf = ",".join(vf_chains) if vf_chains else "null"

    cmd = [
        config.FFMPEG, "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{tw}x{th}", "-r", f"{fps}", "-i", "pipe:0",
        "-i", str(cut_video),
        "-map", "0:v", "-map", "1:a", "-vf", vf,
    ]
    if loudnorm:
        cmd += ["-af", "loudnorm=I=-14:TP=-1.5:LRA=11"]
    cmd += [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-shortest", str(out_video),
    ]

    log = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=log, stderr=log)
    cap = cv2.VideoCapture(str(cut_video))
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if censor:
                _apply_censor(frame, censor, i / fps)
            rect = rects[i] if i < len(rects) else rects[-1]
            comp = _composite(frame, rect, tw, th, blur_k)
            rgb = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)
            try:
                proc.stdin.write(rgb.tobytes())
            except (BrokenPipeError, OSError):
                break
            i += 1
    finally:
        cap.release()
        try:
            proc.stdin.close()
        except OSError:
            pass
        rc = proc.wait()
        log.seek(0)
        err = log.read().decode(errors="ignore")
        log.close()
    if rc != 0:
        raise RuntimeError(f"reframe ffmpeg failed ({rc}) after {i} frames:\n{err[-4000:]}")


# Reframe shot modes + export aspects, for list_reframe_modes / docs.
REFRAME_MODES = {
    "track": "subject-following zoom (YOLO + smoothing; needs a visible person, else holds centre)",
    "zoom": "track with a tighter crop (set per-shot 'zoom' factor, e.g. 1.2)",
    "focus": "frame on explicit source pixels you choose: {mode:'focus', x, y, zoom?} or an explicit {crop:{x,y,w,h}}",
    "full": "entire landscape shown in a centre band with blurred top/bottom bands (zoomed out)",
    "center": "static centre portrait crop",
    "pad": "static fit + blurred bars (legacy; same look as a single 'full' shot)",
    "none": "keep source framing",
}
