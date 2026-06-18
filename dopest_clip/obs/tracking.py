"""Per-frame target tracking — the position source that lets any compose effect FOLLOW a
moving target instead of sitting at static keyframes.

compute_track(video, target) runs a detector over the video and returns a dense per-frame
list of NORMALIZED positions [{t, x, y, w, h}] (x/y = centre 0..1, w/h = target size 0..1),
smoothed with reframe's One-Euro filter. sample_track(track, t) reads a position at time t.

Targets:
  "cursor"                      -> match the OS cursor sprite (assets/cursor.png) on the screen
  "face"                        -> OpenCV haar face detector on the camera
  "person" / a COCO class name  -> YOLO11 (reuse reframe._load_model / _detect_box)
  {"template_at": t, "region": [x,y,w,h]} -> match a seed crop frame-to-frame (UI element)

cv2 / ultralytics are imported LAZILY so importing this module is free and the pure helpers
(sample_track, _to_normalized, validation) are testable without them.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .. import config

# A small slice of the COCO classes YOLO11 knows, for "follow the <thing>" on the camera.
COCO = {
    "person": 0, "backpack": 24, "umbrella": 25, "handbag": 26, "bottle": 39, "cup": 41,
    "fork": 42, "knife": 43, "spoon": 44, "bowl": 45, "banana": 46, "apple": 47,
    "laptop": 63, "mouse": 64, "remote": 65, "keyboard": 66, "cell phone": 67,
    "phone": 67, "book": 73, "clock": 74, "scissors": 76, "teddy bear": 77,
}

_VALID_STR_TARGETS = {"cursor", "face"} | set(COCO)


def validate_target(target: Any) -> None:
    """Raise ValueError if `target` is not a supported tracking target."""
    if isinstance(target, str):
        if target not in _VALID_STR_TARGETS:
            raise ValueError(
                f"unknown track target {target!r}; use 'cursor', 'face', a COCO class "
                f"({', '.join(sorted(COCO))}), or {{'template_at': t, 'region': [x,y,w,h]}}")
        return
    if isinstance(target, dict):
        if "region" not in target or "template_at" not in target:
            raise ValueError("template target needs {'template_at': seconds, 'region': [x,y,w,h]}")
        return
    raise ValueError(f"track target must be a string or a template dict, got {type(target).__name__}")


def _target_key(target: Any) -> str:
    if isinstance(target, str):
        return target.replace(" ", "_")
    r = target["region"]
    return f"tmpl_{target['template_at']}_{'_'.join(str(int(v)) for v in r)}"


# --- pure position sampling (no deps) -------------------------------------------------

def sample_track(track: list[dict[str, float]], t: float) -> tuple[float, float, float, float] | None:
    """Interpolated (x, y, w, h) at time t from a dense per-frame track. None if empty."""
    if not track:
        return None
    if t <= track[0]["t"]:
        e = track[0]
        return (e["x"], e["y"], e["w"], e["h"])
    if t >= track[-1]["t"]:
        e = track[-1]
        return (e["x"], e["y"], e["w"], e["h"])
    lo, hi = 0, len(track) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if track[mid]["t"] < t:
            lo = mid + 1
        else:
            hi = mid
    a, b = track[lo - 1], track[lo]
    span = b["t"] - a["t"]
    p = 0.0 if span <= 0 else (t - a["t"]) / span
    return tuple(a[k] + (b[k] - a[k]) * p for k in ("x", "y", "w", "h"))  # type: ignore[return-value]


def apply_track_to_rect(rect, track, t: float, fw: int, fh: int, *,
                        anchor=(0.5, 0.5), clamp: bool = False, src_rect=None, offset=None):
    """Re-centre a pixel rect (x, y, w, h) on the tracked point at time t, KEEPING its size.
    This is the one bit of glue every tracked effect shares: a static keyframe still decides
    HOW BIG the effect is (overlay scale, blur radius, zoom level); the track only decides
    WHERE it sits. `anchor` = the (ax, ay) point on the rect that lands on the tracked centre
    (0.5,0.5 = box centre; an overlay passes its own anchor). `clamp` keeps the rect inside
    the frame (used for a screen crop).

    `src_rect` = (x, y, w, h) output-pixel rect the track's NORMALIZED coords live inside.
    A camera-source track gives the target's position in the CAMERA frame; pass the per-frame
    composited camera rect so the effect lands on the cutout wherever the camera sits (PIP,
    fullscreen, animated). None => the track coords are already output-frame normalized
    (screen targets). `offset` = (ox, oy) in units of the tracked box size — shift the
    placement relative to the target (oy=-0.9 puts a bulb just ABOVE a tracked head).
    Empty track => rect returned unchanged."""
    pos = sample_track(track, t)
    if pos is None:
        return rect
    w, h = rect[2], rect[3]
    nx, ny, nw, nh = pos
    if src_rect is not None:
        sx, sy, sw, sh = src_rect
        cx, cy = sx + nx * sw, sy + ny * sh
        bw, bh = nw * sw, nh * sh
    else:
        cx, cy = nx * fw, ny * fh
        bw, bh = nw * fw, nh * fh
    if offset is not None:
        cx += offset[0] * bw
        cy += offset[1] * bh
    x = cx - anchor[0] * w
    y = cy - anchor[1] * h
    if clamp:
        x = min(max(x, 0.0), max(0.0, fw - w))
        y = min(max(y, 0.0), max(0.0, fh - h))
    return int(round(x)), int(round(y)), int(round(w)), int(round(h))


def _to_normalized(per_frame: list, sw: int, sh: int, fps: float) -> list[dict[str, float]]:
    """per_frame = list of (cx, cy, w, h) pixel tuples (one per frame; hold last on a miss).
    Smooths the centres with reframe's One-Euro and returns normalized per-frame dicts."""
    from ..reframe import _smooth_centers
    centers = [(p[0], p[1]) for p in per_frame]
    smoothed = _smooth_centers(centers, sw, sh, fps)
    out = []
    for i, (cx, cy) in enumerate(smoothed):
        w, h = per_frame[i][2], per_frame[i][3]
        out.append({"t": round(i / fps, 3), "x": cx / sw, "y": cy / sh,
                    "w": max(0.0, w / sw), "h": max(0.0, h / sh)})
    return out


# --- detectors (lazy cv2 / ultralytics) -----------------------------------------------

def _cursor_detector(sw: int, sh: int):
    import cv2
    tmpl_path = str(config.ASSETS_DIR / "cursor.png")
    if not os.path.isfile(tmpl_path):
        raise FileNotFoundError(
            f"cursor template not found at {tmpl_path}. Crop the OS cursor from a frame "
            "(grab_frame) and save it there to enable cursor tracking.")
    tmpl = cv2.imread(tmpl_path, cv2.IMREAD_GRAYSCALE)
    th, tw = tmpl.shape[:2]

    def detect(frame):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(g, tmpl, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, maxloc = cv2.minMaxLoc(res)
        if maxv < 0.55:           # weak match -> miss (hold last)
            return None
        return (maxloc[0] + tw / 2.0, maxloc[1] + th / 2.0, float(tw), float(th))
    return detect, 1


def _face_detector(sw: int, sh: int):
    import cv2
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect(frame):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(g, 1.2, 5, minSize=(60, 60))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return (x + w / 2.0, y + h / 2.0, float(w), float(h))
    return detect, 2


def _yolo_detector(class_name: str):
    from .. import reframe
    model = reframe._load_model()
    cls = [COCO[class_name]]
    frac = 0.33 if class_name == "person" else 0.5

    def detect(frame):
        return reframe._detect_box(model, frame, 0.35, cls, center_frac=frac)
    return detect, 3


def _template_detector(video_path: str, template_at: float, region: list):
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(template_at * fps)))
    ok, seed = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"could not read the seed frame at {template_at}s for the template target")
    x, y, w, h = (int(v) for v in region)
    crop = cv2.cvtColor(seed[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)

    def detect(frame):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(g, crop, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, maxloc = cv2.minMaxLoc(res)
        if maxv < 0.45:
            return None
        return (maxloc[0] + w / 2.0, maxloc[1] + h / 2.0, float(w), float(h))
    return detect, 2


def _build_detector(target: Any, video_path: str, sw: int, sh: int):
    """Return (detect_fn(frame)->(cx,cy,w,h)|None, sample_every)."""
    if isinstance(target, dict):
        return _template_detector(video_path, float(target["template_at"]), target["region"])
    if target == "cursor":
        return _cursor_detector(sw, sh)
    if target == "face":
        return _face_detector(sw, sh)
    return _yolo_detector(target)  # a COCO class name (validated already)


# --- the entry point ------------------------------------------------------------------

def preview(video_path: str, target: Any, *, at: float | None = None,
            every: int = 2, max_points: int = 200) -> dict:
    """Confirm a target locks on before a full render: run the detector over `video_path`,
    draw the tracked point on a frame at `at` seconds (default the midpoint), and return the
    preview-frame path + a downsampled track. cv2-only (no GPU)."""
    import cv2
    validate_target(target)
    track = compute_track(video_path, target, every=every)
    if not track:
        return {"video": video_path, "target": _target_key(target), "points": 0,
                "preview_frame": None, "track": []}
    dur = track[-1]["t"]
    t = dur / 2.0 if at is None else float(at)
    pos = sample_track(track, t)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
    ok, frame = cap.read()
    cap.release()
    out_png = None
    if ok and pos is not None:
        h, w = frame.shape[:2]
        cx, cy = int(pos[0] * w), int(pos[1] * h)
        cv2.circle(frame, (cx, cy), 26, (0, 255, 0), 3)
        cv2.drawMarker(frame, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 44, 2)
        out_png = os.path.splitext(video_path)[0] + f".track_{_target_key(target)}.png"
        cv2.imwrite(out_png, frame)
    step = max(1, len(track) // max_points)
    return {"video": video_path, "target": _target_key(target), "points": len(track),
            "preview_frame": out_png, "at": round(t, 3),
            "at_point": ({"x": pos[0], "y": pos[1]} if pos else None),
            "track": track[::step]}


def resolve_track(track_spec, screen_path, cam_path, cache_dir):
    """Resolve a {target, source} spec to a per-frame normalized track, cached under
    cache_dir. source 'camera' tracks the camera video; anything else tracks the screen.
    Returns None when there is no spec or the chosen source video is absent. Shared by the
    GPU (camera_mix) and CPU (compositor) compose paths so 'follow X' behaves identically."""
    if not track_spec:
        return None
    source = track_spec.get("source", "screen")
    video = cam_path if source == "camera" else screen_path
    if not video:
        return None
    cache_path = None
    if cache_dir:
        cache_path = os.path.join(cache_dir, f"track_{source}_{_target_key(track_spec['target'])}.json")
    return compute_track(video, track_spec["target"], cache_path=cache_path)


def compute_track(video_path: str, target: Any, *, every: int | None = None,
                  cache_path: str | None = None, fps: float | None = None) -> list[dict[str, float]]:
    """Detect `target` across `video_path` and return a dense per-frame normalized track
    [{t, x, y, w, h}]. Cached to cache_path (json) when given. Detectors are lazy (cv2/YOLO)."""
    validate_target(target)
    if cache_path and os.path.isfile(cache_path):
        try:
            return json.loads(open(cache_path, encoding="utf-8").read())["track"]
        except (OSError, ValueError, KeyError):
            pass
    import cv2
    cap = cv2.VideoCapture(video_path)
    vfps = float(fps or cap.get(cv2.CAP_PROP_FPS) or 30.0)
    sw, sh = int(cap.get(3)), int(cap.get(4))
    detect, default_every = _build_detector(target, video_path, sw, sh)
    step = max(1, every if every is not None else default_every)

    samples: dict[int, Any] = {}
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            samples[idx] = detect(frame)
        idx += 1
    cap.release()
    total = idx
    if total == 0:
        return []

    per_frame = []
    last = (sw / 2.0, sh / 2.0, sw * 0.12, sw * 0.12)
    for i in range(total):
        s = samples.get(i)
        if s is not None:
            last = s
        per_frame.append(last)
    track = _to_normalized(per_frame, sw, sh, vfps)

    if cache_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({"target": _target_key(target), "track": track}, f)
        except OSError:
            pass
    return track
