"""GPU camera mixing for the long-form workflow.

Given a project whose SCREEN recording has been cut to an EDL, mix the separately
recorded camera back in, background-removed, floating over the cut screen:

  1. cut the camera to the SAME segment times as the screen cut (perfect sync),
  2. matte out the background with RobustVideoMatting on the GPU,
  3. composite the floating camera over the cut screen with animated position.

The expensive cut+matte is CACHED per (project, edl), so repositioning ("move me
over here for this part") only re-runs the cheap composite step.

torch / cv2 (opencv) / numpy / moviepy are imported LAZILY inside the functions that
need them, so importing this module needs none of them. Install with:
``pip install dopest-clip[matting]`` (torch + opencv + numpy + moviepy).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .. import config, edl as edlmod, project
from . import timeline, tracking

_RVM = None
_NVENC = None


def _split_track(param):
    """Pull an optional `track` spec out of a keyframe param so the effect can FOLLOW a
    target. The track may ride two ways (both keep the MCP schema a plain keyframe list):
      - on any keyframe dict:  [{t, zoom, focus, track:{target,source}}, ...]
      - as a wrapper dict:     {keyframes:[...], track:{target,source}}
    Returns (clean_keyframes_list, track_spec_or_None); the `track` key is stripped from the
    keyframes so the timeline normalizers never see it."""
    if isinstance(param, dict) and "keyframes" in param:
        return param["keyframes"], param.get("track")
    if isinstance(param, list):
        track = None
        clean = []
        for kf in param:
            if isinstance(kf, dict) and "track" in kf:
                k = dict(kf)
                track = k.pop("track") or track
                clean.append(k)
            else:
                clean.append(kf)
        return clean, track
    return param, None


def _compute_track(track_spec, screen_path, cam_path, cache_dir):
    """Thin alias for tracking.resolve_track (the shared GPU/CPU track resolver)."""
    return tracking.resolve_track(track_spec, screen_path, cam_path, cache_dir)


def _nvenc_ok() -> bool:
    """True if this ffmpeg has the h264_nvenc encoder (cached). Computed lazily so import
    stays cheap and the import-hygiene tests don't spawn ffmpeg."""
    global _NVENC
    if _NVENC is None:
        try:
            out = subprocess.run([config.FFMPEG, "-hide_banner", "-encoders"],
                                 capture_output=True, text=True,
                                 stdin=subprocess.DEVNULL).stdout
            _NVENC = "h264_nvenc" in out
        except Exception:  # noqa: BLE001
            _NVENC = False
    return _NVENC


def _cuda_available() -> bool:
    """True only if torch is importable AND a CUDA device is present. Used to pick the
    GPU RVM matte vs the CPU rembg matte so background removal works on CPU-only boxes."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def _load_rvm():
    """Lazy-load RobustVideoMatting on the GPU (cached locally after first download).
    Install with: pip install dopest-clip[matting]."""
    global _RVM
    if _RVM is None:
        try:
            import torch
        except ImportError as e:  # noqa: BLE001
            raise RuntimeError(
                "GPU camera matting needs torch. Install with: pip install dopest-clip[matting]"
            ) from e
        _RVM = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3").cuda().eval()
    return _RVM


def _resolve_segments(project_id: str, edl_id: str) -> list[tuple[float, float]]:
    """The exact (start, end) source-time segments the screen render kept -- recomputed
    deterministically (same cleanup + resolve the renderer ran)."""
    transcript = project.read_transcript(project_id)
    edl_obj = project.read_edl(project_id, edl_id)
    cleaned = edl_obj
    if edl_obj.get("cleanup"):
        cleaned, _ = edlmod.apply_cleanup(edl_obj, transcript)
    resolved = edlmod.resolve_edl(cleaned, transcript)
    return [(float(s["start"]), float(s["end"])) for s in resolved["segments"]]


def cut_video_only(src: str, segments: list[tuple[float, float]], out: str) -> None:
    """Trim `src` to `segments` and concat (video only -- audio comes from the screen).
    Uses a filter_complex SCRIPT file so hundreds of micro-cuts don't blow the cmdline."""
    parts, labels = [], []
    for i, (s, e) in enumerate(segments):
        parts.append(f"[0:v]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS[v{i}]")
        labels.append(f"[v{i}]")
    graph = ";".join(parts) + ";" + "".join(labels) + \
        f"concat=n={len(segments)}:v=1:a=0[outv]"
    fd, scriptf = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    with open(scriptf, "w", encoding="utf-8") as f:
        f.write(graph)
    try:
        # media.run_ff is the single ffmpeg runner; it sets stdin=DEVNULL so ffmpeg
        # never consumes the MCP server's stdio channel, and raises on non-zero exit.
        from .. import media
        media.run_ff(
            [config.FFMPEG, "-y", "-loglevel", "error", "-i", src,
             "-filter_complex_script", scriptf, "-map", "[outv]", "-an",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", out])
    finally:
        os.remove(scriptf)


def _ffmpeg_frame_writer(out: str, w: int, h: int, fps: float, gray: bool, crf: int):
    """Encode a raw frame stream to mp4. Uses h264_nvenc (GPU encoder block) so the matte
    loop is not bottlenecked on CPU x264 — on this pipeline the dual libx264 encodes of the
    foreground + alpha were the limiter (~15 fps); nvenc frees the CPU. Falls back to
    libx264 if nvenc is unavailable (no NVIDIA encoder / older ffmpeg)."""
    pix = "gray" if gray else "rgb24"
    base = [config.FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", pix,
            "-s", f"{w}x{h}", "-r", f"{fps:.4f}", "-i", "-"]
    if _nvenc_ok():
        enc = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", str(crf), "-pix_fmt", "yuv420p", out]
    else:
        enc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p", out]
    return subprocess.Popen(base + enc, stdin=subprocess.PIPE)


def rvm_matte(src: str, fgr_out: str, pha_out: str,
              downsample: float = 0.25, chunk: int = 12) -> None:
    """RVM GPU matte: write the foreground (fgr_out) and the alpha (pha_out) as videos,
    streaming so memory stays flat on long clips. Install: pip install dopest-clip[matting]."""
    try:
        import cv2
        import numpy as np
        import torch
    except ImportError as e:  # noqa: BLE001
        raise RuntimeError(
            "GPU camera matting needs torch + opencv-python + numpy. "
            "Install with: pip install dopest-clip[matting]"
        ) from e

    model = _load_rvm()
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wf = _ffmpeg_frame_writer(fgr_out, w, h, fps, gray=False, crf=18)
    wp = _ffmpeg_frame_writer(pha_out, w, h, fps, gray=True, crf=12)
    rec: list = [None] * 4
    buf: list = []

    def flush():
        nonlocal rec
        if not buf:
            return
        t = (torch.from_numpy(np.stack(buf)).cuda().float().div(255)
             .permute(0, 3, 1, 2).unsqueeze(0))  # [1,T,3,H,W] RGB
        with torch.no_grad():
            fgr, pha, *rec = model(t, *rec, downsample)
        fgr_np = (fgr[0].clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1) * 255).astype("uint8")
        pha_np = (pha[0, :, 0].clamp(0, 1).cpu().numpy() * 255).astype("uint8")
        for k in range(fgr_np.shape[0]):
            wf.stdin.write(fgr_np[k].tobytes())     # RGB
            wp.stdin.write(pha_np[k].tobytes())     # gray
        buf.clear()

    try:
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            buf.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            if len(buf) >= chunk:
                flush()
        flush()
    finally:
        cap.release()
        for p in (wf, wp):
            p.stdin.close()
            p.wait()


# --- GPU effect helpers (shared by composite_gpu and vertical_clip) ------------------

def _blur_gpu(img_hw3, strength: float):
    """Cheap strong blur of an HxWx3 GPU tensor via downsample/upsample (no conv kernel).
    `strength` ~ blur radius; larger = blurrier."""
    import torch
    h, w = int(img_hw3.shape[0]), int(img_hw3.shape[1])
    k = max(2, int(round(float(strength) / 3.0)))
    t = img_hw3.permute(2, 0, 1).unsqueeze(0)
    small = torch.nn.functional.interpolate(
        t, size=(max(1, h // k), max(1, w // k)), mode="bilinear", align_corners=False)
    return torch.nn.functional.interpolate(
        small, size=(h, w), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)


def _screen_zoom_gpu(screen_hw3, skfs, t: float, sw: int, sh: int, track=None):
    """Crop+zoom an HxWx3 screen tensor per normalized screen keyframes; same size out. When
    `track` is given the crop keeps its zoom level but rides the tracked point (clamped)."""
    if not skfs:
        return screen_hw3
    import torch
    cx, cy, cw, ch = timeline.sample_screen(skfs, t, sw, sh)
    if track:
        cx, cy, cw, ch = tracking.apply_track_to_rect((cx, cy, cw, ch), track, t, sw, sh, clamp=True)
    crop = screen_hw3[cy:cy + ch, cx:cx + cw, :].permute(2, 0, 1).unsqueeze(0)
    return torch.nn.functional.interpolate(
        crop, size=(sh, sw), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)


def _apply_blurs_gpu(screen_hw3, blur_specs, t: float, fw: int, fh: int, duration: float):
    """Blend a blurred copy of the screen where each active blur spec's mask is 1. Reuses
    blur._amp (time window) + blur._mask (rect/circle/svg; focus = invert)."""
    if not blur_specs:
        return screen_hw3
    import torch
    from . import blur as blurmod
    blurred = None
    for s in blur_specs:
        amp = blurmod._amp(s, t, duration)
        if amp <= 0:
            continue
        if blurred is None:
            blurred = _blur_gpu(screen_hw3, float(s.get("strength", 18)))
        rect = timeline.sample_overlay(s["_kfs"], t, fw, fh, float(s.get("aspect", 1.0)), [0.5, 0.5])
        if s.get("_track"):
            rect = tracking.apply_track_to_rect(rect, s["_track"], t, fw, fh)
        m = blurmod._mask(s, rect, fw, fh) * amp  # HxW float numpy
        mt = torch.from_numpy(m).to(screen_hw3.device).float().unsqueeze(-1)
        screen_hw3 = screen_hw3 * (1 - mt) + blurred * mt
        dim = float(s.get("dim", 0.0))
        if dim > 0:
            screen_hw3 = screen_hw3 * (1 - mt * dim)
    return screen_hw3


def _prep_overlays(overlays, fw: int, screen_path=None, cam_path=None, cache_dir=None):
    """Pre-rasterize each overlay to an RGBA numpy array ONCE (svg/kind via resvg, or a
    transparent PNG via image). If an overlay carries a `track` spec, its per-frame track is
    computed ONCE here (cached) and stored so the overlay rides the target. Returns prepared
    dicts for per-frame compositing."""
    if not overlays:
        return []
    from . import graphics
    from .compositor import _load_rgba
    out = []
    for spec in overlays:
        kfs = timeline.normalize_overlay_keyframes(spec["keyframes"])
        if "image" in spec:
            arr = _load_rgba(spec["image"])
            anchor = spec.get("anchor", [0.5, 0.5])
        else:
            svg, anchor = graphics.build(spec)
            base_w = max(k["scale"] for k in kfs) * fw
            arr = graphics.render_svg(svg, int(min(max(base_w * 2, 64), 2 * fw)))
        oh, ow = arr.shape[0], arr.shape[1]
        tspec = spec.get("track") or {}
        out.append({"arr": arr, "anchor": anchor, "kfs": kfs, "aspect": ow / oh,
                    "t_in": float(spec.get("t_in", kfs[0]["t"])), "t_out": spec.get("t_out"),
                    "fade": float(spec.get("fade", 0.3)), "opacity": float(spec.get("opacity", 1.0)),
                    "track": _compute_track(spec.get("track"), screen_path, cam_path, cache_dir),
                    "track_cam": tspec.get("source") == "camera", "track_offset": tspec.get("offset")})
    return out


def _paste_overlays_gpu(canvas_hw3, prepped, t: float, fw: int, fh: int, duration: float,
                        cam_rect=None):
    """Alpha-composite active prepared overlays onto an HxWx3 GPU canvas at time t. `cam_rect`
    = the per-frame composited camera rect (x,y,w,h), so a camera-source tracked overlay (e.g.
    a bulb over the tracked head) lands on the cutout wherever the camera currently sits."""
    if not prepped:
        return canvas_hw3
    import torch
    for o in prepped:
        t_out = duration if o["t_out"] is None else float(o["t_out"])
        t_in = o["t_in"]
        if t < t_in or t > t_out:
            continue
        amp = o["opacity"]
        fade = min(o["fade"], max(0.001, (t_out - t_in) / 2.0))
        if fade > 0:
            amp *= max(0.0, min((t - t_in) / fade, (t_out - t) / fade, 1.0))
        if amp <= 0:
            continue
        x, y, w, h = timeline.sample_overlay(o["kfs"], t, fw, fh, o["aspect"], o["anchor"])
        if o.get("track"):
            x, y, w, h = tracking.apply_track_to_rect(
                (x, y, w, h), o["track"], t, fw, fh, anchor=o["anchor"],
                src_rect=(cam_rect if o.get("track_cam") else None), offset=o.get("track_offset"))
        if w < 1 or h < 1:
            continue
        rgba = torch.from_numpy(o["arr"]).to(canvas_hw3.device).float().permute(2, 0, 1).unsqueeze(0)
        rs = torch.nn.functional.interpolate(rgba, size=(h, w), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(fw, x + w), min(fh, y + h)
        if x1 <= x0 or y1 <= y0:
            continue
        ox0, oy0 = x0 - x, y0 - y
        rgb = rs[oy0:oy0 + (y1 - y0), ox0:ox0 + (x1 - x0), :3]
        al = rs[oy0:oy0 + (y1 - y0), ox0:ox0 + (x1 - x0), 3:4] / 255.0 * amp
        canvas_hw3[y0:y1, x0:x1, :] = rgb * al + canvas_hw3[y0:y1, x0:x1, :] * (1 - al)
    return canvas_hw3


def _bg_mode_at(backgrounds: list[dict[str, Any]] | None, t: float,
                bg_visible_until: float | None) -> str:
    """Resolve the background mode at cut-time t. A window in `backgrounds` covering t wins
    (last one covering t, so later windows override earlier); otherwise fall back to the
    bg_visible_until rule ('real' while t < bg_visible_until, else 'screen'). Returns a mode:
    'real'/'camera' (full un-matted camera), 'screen' (cutout over the cut screen), or an
    image path (cutout over that cover-fit still). Pure."""
    mode = None
    for w in (backgrounds or []):
        if float(w["start"]) <= t < float(w["end"]):
            mode = w.get("mode", "real")
    if mode is not None:
        return mode
    if bg_visible_until is not None and t < float(bg_visible_until):
        return "real"
    return "screen"


def composite_gpu(screen_path: str, fgr_path: str, pha_path: str,
                  keyframes: list[dict[str, Any]], out_path: str, *,
                  cut_cam_path: str | None = None, overlays: list[dict[str, Any]] | None = None,
                  blurs: list[dict[str, Any]] | None = None,
                  screen_keyframes: list[dict[str, Any]] | None = None,
                  bg_visible_until: float | None = None,
                  backgrounds: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """GPU composite (torch/CUDA + NVENC): per-frame screen-zoom -> screen blur/focus ->
    camera over screen (matted cutout, OR full opaque camera while t < bg_visible_until for
    a mid-clip background drop) -> graphic overlays on top. Every effect is optional and
    paid for only when its spec is present. Audio muxed from the screen. Raises on failure
    so mix() can fall back to the CPU composite."""
    import cv2
    import torch

    cap_s = cv2.VideoCapture(screen_path)
    cap_f = cv2.VideoCapture(fgr_path)
    cap_p = cv2.VideoCapture(pha_path)
    # the un-matted camera is needed for any 'real'/'camera' background window (or the legacy
    # bg_visible_until phase). Image/'screen' windows only need the matte (fgr/pha).
    needs_cam = (bg_visible_until is not None) or any(
        w.get("mode", "real") in ("real", "camera") for w in (backgrounds or []))
    cap_c = cv2.VideoCapture(cut_cam_path) if (cut_cam_path and needs_cam) else None
    fps = cap_s.get(cv2.CAP_PROP_FPS) or 30.0
    fw = int(cap_s.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap_s.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(cap_s.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (nframes / fps) if nframes else 1e9
    cw = int(cap_f.get(cv2.CAP_PROP_FRAME_WIDTH))
    ch = int(cap_f.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspect = cw / ch
    cache_dir = os.path.dirname(os.path.abspath(out_path))
    # camera + screen-zoom params may arrive as {keyframes, track} to FOLLOW a target.
    kf_list, cam_track_spec = _split_track(keyframes)
    skf_list, screen_track_spec = _split_track(screen_keyframes or [])
    kfs = timeline.normalize_keyframes(kf_list)
    skfs = timeline.normalize_screen_keyframes(skf_list)
    cam_track = _compute_track(cam_track_spec, screen_path, cut_cam_path, cache_dir)
    screen_track = _compute_track(screen_track_spec, screen_path, cut_cam_path, cache_dir)
    blur_specs = []
    for s in (blurs or []):
        s2 = dict(s)
        s2["_kfs"] = timeline.normalize_overlay_keyframes(s["keyframes"])
        s2["_track"] = _compute_track(s.get("track"), screen_path, cut_cam_path, cache_dir)
        blur_specs.append(s2)
    prepped = _prep_overlays(overlays, fw, screen_path=screen_path, cam_path=cut_cam_path, cache_dir=cache_dir)
    # preload any image-mode background windows, cover-fit to the frame (cutout composites over them)
    bg_tensors: dict[str, Any] = {}
    for w in (backgrounds or []):
        m = w.get("mode", "real")
        if m not in ("real", "camera", "screen") and m not in bg_tensors:
            bgi = cv2.imread(m, cv2.IMREAD_COLOR)
            if bgi is None:
                raise ValueError(f"could not read background image: {m}")
            bg_tensors[m] = torch.from_numpy(
                cv2.cvtColor(_cover_to(bgi, fw, fh), cv2.COLOR_BGR2RGB)).to("cuda").float()
    os.makedirs(cache_dir, exist_ok=True)

    proc = subprocess.Popen(
        [config.FFMPEG, "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{fw}x{fh}", "-r", f"{fps:.4f}",
         "-i", "pipe:0", "-i", screen_path,
         "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "h264_nvenc", "-preset", "p4",
         "-cq", "21", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
         "-shortest", out_path],
        stdin=subprocess.PIPE)

    dev = "cuda"
    idx = 0
    try:
        with torch.no_grad():
            while True:
                ok_s, s = cap_s.read()
                if not ok_s:
                    break
                ok_f, f = cap_f.read()
                ok_p, p = cap_p.read()
                if not (ok_f and ok_p):
                    break
                cfrm = None
                if cap_c is not None:
                    ok_c, cfrm = cap_c.read()
                    if not ok_c:
                        cfrm = None
                t = idx / fps
                idx += 1
                screen = torch.from_numpy(cv2.cvtColor(s, cv2.COLOR_BGR2RGB)).to(dev).float()
                if skfs:
                    screen = _screen_zoom_gpu(screen, skfs, t, fw, fh, track=screen_track)
                if blur_specs:
                    screen = _apply_blurs_gpu(screen, blur_specs, t, fw, fh, duration)
                x, y, w, h = timeline.sample(kfs, t, fw, fh, aspect)
                if cam_track:
                    x, y, w, h = tracking.apply_track_to_rect((x, y, w, h), cam_track, t, fw, fh)
                x0, y0 = max(0, x), max(0, y)
                x1, y1 = min(fw, x + w), min(fh, y + h)
                mode = _bg_mode_at(backgrounds, t, bg_visible_until)
                if mode in ("real", "camera") and cfrm is not None:
                    # real-background phase: composite the FULL un-matted camera, opaque
                    cam = (torch.from_numpy(cv2.cvtColor(cfrm, cv2.COLOR_BGR2RGB)).to(dev).float()
                           .permute(2, 0, 1).unsqueeze(0))
                    cam_r = torch.nn.functional.interpolate(
                        cam, size=(h, w), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
                    if x1 > x0 and y1 > y0:
                        fx0, fy0 = x0 - x, y0 - y
                        screen[y0:y1, x0:x1, :] = cam_r[fy0:fy0 + (y1 - y0), fx0:fx0 + (x1 - x0), :]
                else:
                    if mode not in ("real", "camera", "screen"):
                        # image-mode window: the cutout composites over the cover-fit still
                        screen = bg_tensors[mode].clone()
                    fgr = (torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).to(dev).float()
                           .permute(2, 0, 1).unsqueeze(0))
                    pha = (torch.from_numpy(cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)).to(dev).float()
                           .div(255.0).unsqueeze(0).unsqueeze(0))
                    fgr_r = torch.nn.functional.interpolate(
                        fgr, size=(h, w), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
                    pha_r = torch.nn.functional.interpolate(
                        pha, size=(h, w), mode="bilinear", align_corners=False)[0, 0].unsqueeze(-1)
                    if x1 > x0 and y1 > y0:
                        fx0, fy0 = x0 - x, y0 - y
                        roi = screen[y0:y1, x0:x1, :]
                        a = pha_r[fy0:fy0 + (y1 - y0), fx0:fx0 + (x1 - x0), :]
                        fg = fgr_r[fy0:fy0 + (y1 - y0), fx0:fx0 + (x1 - x0), :]
                        screen[y0:y1, x0:x1, :] = fg * a + roi * (1 - a)
                if prepped:
                    screen = _paste_overlays_gpu(screen, prepped, t, fw, fh, duration,
                                                 cam_rect=(x, y, w, h))
                proc.stdin.write(screen.clamp(0, 255).byte().cpu().numpy().tobytes())
    finally:
        cap_s.release(); cap_f.release(); cap_p.release()
        if cap_c is not None:
            cap_c.release()
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"nvenc composite failed (ffmpeg rc={rc})")
    return {"output": out_path, "size": [fw, fh], "duration": round(idx / fps, 2)}


def composite(screen_path: str, fgr_path: str, pha_path: str | None,
              keyframes: list[dict[str, Any]], out_path: str) -> dict[str, Any]:
    """Composite the (matted) camera over the cut screen with animated position. Cheap
    relative to the matte -- this is what reposition re-runs. CPU moviepy path."""
    try:
        from moviepy import CompositeVideoClip, VideoFileClip
    except ImportError as e:  # noqa: BLE001
        raise RuntimeError(
            "composite() needs moviepy. Install with: pip install dopest-clip[matting]"
        ) from e

    screen = VideoFileClip(screen_path)
    fgr = VideoFileClip(fgr_path)
    cam = fgr.without_audio()
    if pha_path:
        cam = cam.with_mask(VideoFileClip(pha_path).to_mask())
    fw, fh = screen.size
    aspect = cam.w / cam.h
    kfs = timeline.normalize_keyframes(keyframes)
    dur = min(screen.duration, cam.duration)

    def rect(t: float):
        return timeline.sample(kfs, t, fw, fh, aspect)

    cam_layer = (cam.resized(lambda t: (rect(t)[2], rect(t)[3]))
                 .with_position(lambda t: (rect(t)[0], rect(t)[1])))
    final = CompositeVideoClip([screen, cam_layer], size=(fw, fh)).with_duration(dur)
    if screen.audio is not None:
        final = final.with_audio(screen.audio.subclipped(0, dur))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    final.write_videofile(out_path, codec="libx264", audio_codec="aac",
                          fps=screen.fps, preset="medium",
                          threads=os.cpu_count() or 4, logger=None)
    for c in (final, cam_layer, cam, fgr, screen):
        try:
            c.close()
        except Exception:  # noqa: BLE001
            pass
    return {"output": out_path, "size": [fw, fh], "duration": round(dur, 2)}


def mix(project_id: str, edl_id: str, camera_path: str,
        keyframes: list[dict[str, Any]] | None = None,
        remove_background: bool = True, output_path: str = "",
        rematte: bool = False, overlays: list[dict[str, Any]] | None = None,
        blurs: list[dict[str, Any]] | None = None,
        screen_keyframes: list[dict[str, Any]] | None = None,
        bg_visible_until: float | None = None,
        backgrounds: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Mix the camera into a project's cut screen with the full effect stack. The cut +
    matte are CACHED per (project, edl) under <project>/camera/; pass rematte=True to force
    a rebuild. Requires render(project_id, edl_id) first. `keyframes` = camera animation
    (default static bottom-right PIP). `overlays` = animated graphics (arrow/ring/box/label/
    inline-svg/image PNG). `blurs` = animated screen blur/focus (focus = invert). `screen_
    keyframes` = screen crop+zoom over time. `bg_visible_until` = seconds (cut timeline) to
    keep the FULL camera background visible before it drops to a cutout. Effects/cutout run
    on the GPU (RVM+NVENC) when CUDA is present; CPU rembg fallback otherwise. All effect
    times are cut-timeline seconds (see get_cut_transcript)."""
    pdir = project.require_project(project_id)
    slug = project.slugify(edl_id)
    cut_screen = pdir / "renders" / f"{slug}.mp4"
    if not cut_screen.exists():
        raise FileNotFoundError(
            f"cut screen not found at {cut_screen}. Run render(project_id, edl_id='{edl_id}') first.")
    if not os.path.isfile(camera_path):
        raise FileNotFoundError(f"camera file not found: {camera_path}")

    cam_dir = pdir / "camera"
    cam_dir.mkdir(exist_ok=True)
    cut_cam = cam_dir / f"{slug}_cut.mp4"
    if rematte or not cut_cam.exists():
        cut_video_only(camera_path, _resolve_segments(project_id, edl_id), str(cut_cam))

    if not keyframes:
        keyframes = [{"t": 0, "preset": "bottom-right"}]
    # validate image-mode background windows up front (clear error before the GPU pass)
    for w in (backgrounds or []):
        m = w.get("mode", "real")
        if m not in ("real", "camera", "screen") and not os.path.isfile(m):
            raise FileNotFoundError(f"background window image not found: {m}")
    out = output_path or str(pdir / "renders" / f"{slug}_mixed.mp4")
    effects = bool(overlays or blurs or screen_keyframes or bg_visible_until is not None or backgrounds)

    cached = False
    backend = "raw-inset"
    if (remove_background or effects) and _cuda_available():
        # GPU path: RVM matte (cached fgr/pha) + NVENC composite with the full effect stack.
        fgr = cam_dir / f"{slug}_fgr.mp4"
        pha = cam_dir / f"{slug}_pha.mp4"
        if rematte or not (fgr.exists() and pha.exists()):
            rvm_matte(str(cut_cam), str(fgr), str(pha))
        else:
            cached = True
        try:
            info = composite_gpu(str(cut_screen), str(fgr), str(pha), keyframes, out,
                                 cut_cam_path=str(cut_cam), overlays=overlays, blurs=blurs,
                                 screen_keyframes=screen_keyframes, bg_visible_until=bg_visible_until,
                                 backgrounds=backgrounds)
            backend = "rvm-gpu"
        except Exception:  # noqa: BLE001
            # CPU fallback keeps overlays/blur (compositor.compose) but not screen-zoom/bg-toggle.
            from . import compositor
            info = compositor.compose(str(cut_screen), str(cut_cam), keyframes, out,
                                      remove_background=True, overlays=overlays, blurs=blurs,
                                      bg_visible_until=bg_visible_until)
            backend = "rvm-gpu-failed+cpu-compose"
    elif remove_background or effects:
        # No CUDA: matte + overlays/blur on CPU via rembg/moviepy (slower).
        from . import compositor
        info = compositor.compose(str(cut_screen), str(cut_cam), keyframes, out,
                                  remove_background=remove_background, overlays=overlays,
                                  blurs=blurs, bg_visible_until=bg_visible_until)
        backend = "rembg-cpu"
    else:
        info = composite(str(cut_screen), str(cut_cam), None, keyframes, out)
    info.update({"project_id": project_id, "edl_id": edl_id,
                 "background_removed": remove_background, "matte_cached": cached,
                 "matte_backend": backend})
    return info


def _trim(src: str, t0: float, dur: float, out: str, audio: bool = False) -> None:
    """Trim a clip to [t0, t0+dur]. Uses media.run_ff (stdin=DEVNULL) so it never
    consumes the MCP server's stdio channel."""
    from .. import media
    cmd = [config.FFMPEG, "-y", "-loglevel", "error", "-ss", str(t0), "-t", str(dur),
           "-i", src, "-c:v", "libx264", "-preset", "veryfast", "-crf", "16"]
    cmd += (["-c:a", "aac"] if audio else ["-an"])
    cmd.append(out)
    media.run_ff(cmd)


def write_cut_transcript(project_id: str, edl_id: str) -> tuple[str, str, int]:
    """Derive the CUT-timeline transcript (the final spoken words after cleanup + cut,
    re-indexed 0..N with cut-timeline timestamps) and write .json + readable .txt next to
    the render. Shorts are designed against THESE indices. Returns (json, txt, n_words)."""
    import json

    pdir = project.project_dir(project_id)
    slug = project.slugify(edl_id)
    transcript = project.read_transcript(project_id)
    edl_obj = project.read_edl(project_id, edl_id)
    cleaned = edl_obj
    if edl_obj.get("cleanup"):
        cleaned, _ = edlmod.apply_cleanup(edl_obj, transcript)
    resolved = edlmod.resolve_edl(cleaned, transcript)
    words = edlmod.remap_to_output_timeline(resolved, transcript)
    for i, w in enumerate(words):
        w["i"] = i

    cj = pdir / "renders" / f"{slug}.cut_transcript.json"
    cj.parent.mkdir(parents=True, exist_ok=True)
    cj.write_text(json.dumps({"words": words}), encoding="utf-8")

    lines = [f"# Cut transcript for project '{project_id}' edl '{edl_id}' | words: {len(words)}",
             "# [mm:ss.xxx] (#word_index) text  -- design shorts by these CUT-timeline indices", ""]
    cur: list[str] = []
    ci, ct0 = 0, 0.0
    for w in words:
        if not cur:
            ci, ct0 = w["i"], float(w["start"])
        cur.append(w["w"])
        if len(cur) >= 16:
            lines.append(f"[{int(ct0 // 60):02d}:{ct0 - 60 * int(ct0 // 60):06.3f}] (#{ci}) " + " ".join(cur))
            cur = []
    if cur:
        lines.append(f"[{int(ct0 // 60):02d}:{ct0 - 60 * int(ct0 // 60):06.3f}] (#{ci}) " + " ".join(cur))
    ct = pdir / "renders" / f"{slug}.cut_transcript.txt"
    ct.write_text("\n".join(lines), encoding="utf-8")
    return str(cj), str(ct), len(words)


def vertical_clip(screen_cut: str, fgr: str, pha: str, transcript_json: str,
                  from_word: int, to_word: int, hook_title: str, out_path: str,
                  caption_preset: str = "karaoke-bold", title_hold: float = 2.5,
                  screen_top_y: int = 280, person_h: int = 1180,
                  screen_keyframes: list[dict[str, Any]] | None = None,
                  overlays: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Render a 9:16 SHORT-FORM clip: hook + karaoke captions in the TOP band, the screen
    in the MIDDLE (per-frame GPU crop+zoom from `screen_keyframes` so it can zoom into
    what's being discussed), and the background-removed person BIG at the BOTTOM over a
    blurred-screen backdrop. `screen_keyframes`: [{t, zoom, focus:[nx,ny], ease}] (zoom
    1.0 = full overview, 2.0 = 2x into focus). None = static full screen. GPU/NVENC."""
    import json
    import tempfile

    import cv2
    import torch

    from .. import captions, media

    words = json.load(open(transcript_json, encoding="utf-8"))["words"]
    t0 = float(words[from_word]["start"])
    dur = float(words[to_word]["end"]) - t0
    local = [{"w": words[i]["w"],
              "start": round(float(words[i]["start"]) - t0, 3),
              "end": round(float(words[i]["end"]) - t0, 3)}
             for i in range(from_word, to_word + 1)]
    ass = captions.build_ass(local, 1080, 1920, preset=caption_preset, position="top",
                             margin_v=40, title=hook_title, title_hold=title_hold)
    ass_path = os.path.splitext(out_path)[0] + ".ass"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(ass)
    skf_list, screen_track_spec = _split_track(screen_keyframes or [])
    skfs = timeline.normalize_screen_keyframes(skf_list)

    tmp = tempfile.mkdtemp()
    cs, cf, cp = os.path.join(tmp, "s.mp4"), os.path.join(tmp, "f.mp4"), os.path.join(tmp, "p.mp4")
    _trim(screen_cut, t0, dur, cs, audio=True)
    _trim(fgr, t0, dur, cf)
    _trim(pha, t0, dur, cp)
    # tracks for a short are computed on the TRIMMED clip (normalized to its own frame).
    cache_dir = os.path.dirname(os.path.abspath(out_path))
    screen_track = _compute_track(screen_track_spec, cs, cf, cache_dir)

    cap_s, cap_f, cap_p = cv2.VideoCapture(cs), cv2.VideoCapture(cf), cv2.VideoCapture(cp)
    fps = cap_s.get(cv2.CAP_PROP_FPS) or 30.0
    sw, sh = int(cap_s.get(3)), int(cap_s.get(4))
    pfw, pfh = int(cap_f.get(3)), int(cap_f.get(4))
    CW, CH = 1080, 1920
    disp_w, disp_h = 1080, int(round(1080 * sh / sw))   # screen display zone (e.g. 1080x607)
    pw = int(round(pfw * person_h / pfh))               # person scaled to person_h tall
    prepped = _prep_overlays(overlays, CW, screen_path=cs, cam_path=cf, cache_dir=cache_dir)

    sub = (f"subtitles=filename='{media.escape_filter_path(ass_path)}'"
           f":fontsdir='{media.escape_filter_path(str(captions.FONTS_DIR))}'")
    proc = subprocess.Popen(
        [config.FFMPEG, "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{CW}x{CH}", "-r", f"{fps:.4f}",
         "-i", "pipe:0", "-i", cs,
         "-filter_complex", f"[0:v]{sub}[v]", "-map", "[v]", "-map", "1:a:0?",
         "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "21", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", out_path],
        stdin=subprocess.PIPE)

    dev = "cuda"
    idx = 0
    try:
        with torch.no_grad():
            while True:
                ok_s, s = cap_s.read()
                if not ok_s:
                    break
                ok_f, f = cap_f.read()
                ok_p, p = cap_p.read()
                if not (ok_f and ok_p):
                    break
                t = idx / fps
                idx += 1
                screen = torch.from_numpy(cv2.cvtColor(s, cv2.COLOR_BGR2RGB)).to(dev).float()
                # background: blurred + darkened full screen stretched to canvas
                base = screen.permute(2, 0, 1).unsqueeze(0)
                small = torch.nn.functional.interpolate(base, size=(96, 54), mode="bilinear", align_corners=False)
                bg = torch.nn.functional.interpolate(small, size=(CH, CW), mode="bilinear", align_corners=False)[0].permute(1, 2, 0) * 0.45
                canvas = bg.clone()
                # middle: zoomed screen crop -> display zone
                cx, cy, cw, ch = timeline.sample_screen(skfs, t, sw, sh)
                if screen_track:
                    cx, cy, cw, ch = tracking.apply_track_to_rect((cx, cy, cw, ch), screen_track, t, sw, sh, clamp=True)
                crop = screen[cy:cy + ch, cx:cx + cw, :].permute(2, 0, 1).unsqueeze(0)
                disp = torch.nn.functional.interpolate(crop, size=(disp_h, disp_w), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
                canvas[screen_top_y:screen_top_y + disp_h, 0:disp_w, :] = disp
                # bottom: big background-removed person, bottom-centre
                fgr_t = torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).to(dev).float().permute(2, 0, 1).unsqueeze(0)
                pha_t = torch.from_numpy(cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)).to(dev).float().div(255.0).unsqueeze(0).unsqueeze(0)
                fr = torch.nn.functional.interpolate(fgr_t, size=(person_h, pw), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
                ar = torch.nn.functional.interpolate(pha_t, size=(person_h, pw), mode="bilinear", align_corners=False)[0, 0].unsqueeze(-1)
                px, py = (CW - pw) // 2, CH - person_h
                x0, y0 = max(0, px), max(0, py)
                x1, y1 = min(CW, px + pw), min(CH, py + person_h)
                fx0, fy0 = x0 - px, y0 - py
                a = ar[fy0:fy0 + (y1 - y0), fx0:fx0 + (x1 - x0), :]
                fg = fr[fy0:fy0 + (y1 - y0), fx0:fx0 + (x1 - x0), :]
                canvas[y0:y1, x0:x1, :] = fg * a + canvas[y0:y1, x0:x1, :] * (1 - a)
                if prepped:
                    canvas = _paste_overlays_gpu(canvas, prepped, t, CW, CH, dur,
                                                 cam_rect=(px, py, pw, person_h))
                proc.stdin.write(canvas.clamp(0, 255).byte().cpu().numpy().tobytes())
    finally:
        cap_s.release(); cap_f.release(); cap_p.release()
        proc.stdin.close()
        rc = proc.wait()
        for fp in (cs, cf, cp):
            try:
                os.remove(fp)
            except OSError:
                pass
    if rc != 0:
        raise RuntimeError(f"vertical_clip nvenc failed (rc={rc})")
    return {"output": out_path, "size": [CW, CH], "duration": round(dur, 2),
            "hook": hook_title, "zoomed": bool(skfs)}


def short_clip(project_id: str, edl_id: str, from_word: int, to_word: int,
               hook_title: str, screen_keyframes: list[dict[str, Any]] | None = None,
               caption_preset: str = "karaoke-bold", output_path: str = "",
               overlays: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Render a 9:16 short-form clip (vertical stacked layout + optional screen zoom +
    optional graphic overlays) from a project's cut screen + cached camera matte. Requires
    render() (cut screen) and mix_camera(remove_background=True) (matte) to have run for
    `edl_id`. `from_word`/`to_word` index the CUT transcript (see get_cut_transcript)."""
    pdir = project.require_project(project_id)
    slug = project.slugify(edl_id)
    cut_screen = pdir / "renders" / f"{slug}.mp4"
    fgr = pdir / "camera" / f"{slug}_fgr.mp4"
    pha = pdir / "camera" / f"{slug}_pha.mp4"
    if not cut_screen.exists():
        raise FileNotFoundError(f"cut screen not found ({cut_screen}); run render(edl_id='{edl_id}') first")
    if not (fgr.exists() and pha.exists()):
        raise FileNotFoundError(
            "camera matte not found; run mix_camera(remove_background=True) first (GPU matte)")
    cj, _, _ = write_cut_transcript(project_id, edl_id)
    out = output_path or str(pdir / "renders" / f"{slug}_short_{from_word}-{to_word}.mp4")
    return vertical_clip(str(cut_screen), str(fgr), str(pha), cj, from_word, to_word,
                         hook_title, out, caption_preset=caption_preset,
                         screen_keyframes=screen_keyframes, overlays=overlays)


# --- background replacement (cutout over a provided still image, per time window) --------

def _cover_to(img, w: int, h: int):
    """Resize a BGR image to COVER (w,h) (no distortion) and centre-crop to exactly (w,h)."""
    import cv2
    ih, iw = img.shape[:2]
    scale = max(w / iw, h / ih)
    nw, nh = max(w, int(round(iw * scale))), max(h, int(round(ih * scale)))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    x0, y0 = (nw - w) // 2, (nh - h) // 2
    return r[y0:y0 + h, x0:x0 + w]


def _segment_at(segs: list[dict[str, Any]], t: float):
    """The background of the LAST segment covering time t (cut-timeline seconds); 'camera'
    (the real recorded background, passthrough) if no segment covers t. Pure."""
    active = "camera"
    for s in segs:
        if s["start"] <= t < s["end"]:
            active = s["background"]
    return active


def replace_background(project_id: str, edl_id: str, segments: list[dict[str, Any]],
                       output_path: str = "", max_duration: float | None = None) -> dict[str, Any]:
    """Re-background the talking head: composite the cached cutout (matte) over a DIFFERENT
    background per time window. `segments` = [{start, end, background}] in cut-timeline seconds;
    `background` is either "camera" (show the REAL recorded room, no cutout — the original
    frames) or a path to a still IMAGE the cutout is composited over (cover-fit to the frame).
    The background image is produced however you like (for us: the Gemini image toolkit to remove
    the person and modify the office); this op is provider-agnostic — it just composites. Reuses
    the cached cut camera + RVM matte, so run mix_camera(edl_id, remove_background=True) first.
    Audio comes from the cut screen. GPU compose + NVENC when CUDA is present. `max_duration`
    caps the render (for a quick sample)."""
    import cv2
    import numpy as np
    import torch

    pdir = project.require_project(project_id)
    slug = project.slugify(edl_id)
    cam_dir = pdir / "camera"
    cut_cam, fgr, pha = cam_dir / f"{slug}_cut.mp4", cam_dir / f"{slug}_fgr.mp4", cam_dir / f"{slug}_pha.mp4"
    screen = pdir / "renders" / f"{slug}.mp4"
    for pth, what in ((cut_cam, "cut camera"), (fgr, "camera foreground matte"), (pha, "camera alpha matte")):
        if not pth.exists():
            raise FileNotFoundError(
                f"{what} not found at {pth}; run mix_camera(edl_id='{edl_id}', "
                "remove_background=True) first to build the cached matte")
    if not segments:
        raise ValueError("replace_background needs at least one segment {start, end, background}")
    segs = []
    for s in segments:
        bg = s["background"]
        if bg != "camera" and not os.path.isfile(bg):
            raise FileNotFoundError(f"background image not found: {bg}")
        segs.append({"start": float(s.get("start", 0.0)), "end": float(s.get("end", 1e9)), "background": bg})
    segs.sort(key=lambda x: x["start"])

    cap_c, cap_f, cap_p = cv2.VideoCapture(str(cut_cam)), cv2.VideoCapture(str(fgr)), cv2.VideoCapture(str(pha))
    fps = cap_c.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap_c.get(3)), int(cap_c.get(4))
    dev = "cuda" if _cuda_available() else "cpu"

    bg_cache: dict[str, Any] = {}
    for s in segs:
        bg = s["background"]
        if bg != "camera" and bg not in bg_cache:
            img = cv2.imread(bg, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"could not read background image: {bg}")
            bg_cache[bg] = torch.from_numpy(
                cv2.cvtColor(_cover_to(img, w, h), cv2.COLOR_BGR2RGB)).to(dev).float()

    out = output_path or str(pdir / "renders" / f"{slug}_bgreplace.mp4")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    enc = (["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "21"] if _nvenc_ok()
           else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"])
    proc = subprocess.Popen(
        [config.FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", f"{fps:.4f}", "-i", "pipe:0", "-i", str(screen),
         "-map", "0:v:0", "-map", "1:a:0?", *enc, "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-shortest", out],
        stdin=subprocess.PIPE)

    idx = 0
    used: set = set()
    try:
        with torch.no_grad():
            while True:
                ok_c, cf = cap_c.read()
                if not ok_c:
                    break
                ok_f, f = cap_f.read()
                ok_p, p = cap_p.read()
                if not (ok_f and ok_p):
                    break
                t = idx / fps
                idx += 1
                if max_duration is not None and t >= float(max_duration):
                    break
                bg = _segment_at(segs, t)
                used.add(bg)
                if bg == "camera":
                    proc.stdin.write(np.ascontiguousarray(cv2.cvtColor(cf, cv2.COLOR_BGR2RGB)).tobytes())
                    continue
                base = bg_cache[bg]
                fgr_t = torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).to(dev).float()
                a = torch.from_numpy(cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)).to(dev).float().div(255.0).unsqueeze(-1)
                comp = base * (1 - a) + fgr_t * a
                proc.stdin.write(comp.clamp(0, 255).byte().cpu().numpy().tobytes())
    finally:
        cap_c.release(); cap_f.release(); cap_p.release()
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"replace_background ffmpeg failed (rc={rc})")
    return {"output": out, "size": [w, h], "duration": round(idx / fps, 2),
            "segments": len(segs), "backgrounds_used": sorted(used),
            "matte_backend": "rvm-gpu" if dev == "cuda" else "cpu"}


# --- pause montage (the "millennial pause" gag) ---------------------------------------

def _pick_pauses(silences: list[dict[str, Any]], count: int, min_dur: float) -> list[dict[str, Any]]:
    """Genuine dead-air pauses (a silence gap of at least `min_dur`s — no words spoken), spread
    EVENLY across the timeline, up to `count`. Even spread (not the first N) so the montage samples
    the whole video. The clip is taken from INSIDE the gap at extraction time, so it is pure dead
    air (breathing / room tone), never the surrounding speech. Pure."""
    elig = [s for s in silences if s["dur"] >= min_dur]
    if count <= 0 or len(elig) <= count:
        return elig
    step = len(elig) / count
    return [elig[int(i * step)] for i in range(count)]


def _norm_clip(src: str, start: float | None, dur: float | None, out: str,
               audio_src: str | None = None) -> None:
    """Trim + normalize a clip to identical CFR params (1080p30 h264 / 48k stereo aac) so a
    list of them concat cleanly with -c copy. -ss before -i = fast keyframe seek. If `audio_src`
    is given, the VIDEO comes from `src` and the AUDIO from `audio_src` at the same window — used
    so a camera pause clip (whose own track is silent) carries the synced mic room tone/breathing."""
    from .. import media
    cmd = [config.FFMPEG, "-y", "-loglevel", "error"]
    if start is not None:
        cmd += ["-ss", f"{start}"]
    if dur is not None:
        cmd += ["-t", f"{dur}"]
    cmd += ["-i", src]
    amap = "0:a?"
    if audio_src:
        if start is not None:
            cmd += ["-ss", f"{start}"]
        if dur is not None:
            cmd += ["-t", f"{dur}"]
        cmd += ["-i", audio_src]
        amap = "1:a?"
    cmd += ["-map", "0:v:0", "-map", amap,
            "-vf", "scale=1920:1080:force_original_aspect_ratio=disable,fps=30,setsar=1",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-video_track_timescale", "30000",
            "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "192k", out]
    media.run_ff(cmd)


def _concat_files(parts: list, out: str) -> None:
    """Concat identically-encoded clips via the concat demuxer (-c copy). Paths use forward
    slashes so the demuxer list parses on Windows."""
    from .. import media
    lst = str(Path(out).with_suffix(".concat.txt"))
    Path(lst).write_text(
        "".join(f"file '{str(p).replace(chr(92), '/')}'\n" for p in parts), encoding="utf-8")
    media.run_ff([config.FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                  "-i", lst, "-c", "copy", str(out)])


def _frame_montage(montage: str, frame_image: str, screen_rect: list, out: str,
                   frame_chroma: str | None = None) -> None:
    """Wrap the montage inside a decorative frame image (e.g. a Pause-O-Vision TV bezel): the
    frame is scaled to 1080 tall and centered on a black 1920x1080 canvas. `screen_rect` =
    [x,y,w,h] (final-canvas px) is the inner screen. Without `frame_chroma` the montage is drawn
    ON TOP of the frame at screen_rect. With `frame_chroma` (the screen's solid color, e.g.
    '0x221b25') the frame's screen is keyed transparent and the montage is composited BEHIND the
    frame (cover-filling the screen), so the bezel + title stay ON TOP and nothing is cut off."""
    from .. import media
    x, y, w, h = (int(v) for v in screen_rect)
    if frame_chroma:
        fc = (f"[0:v]scale=-2:1080,colorkey={frame_chroma}:0.26:0.12[frm];"
              f"[1:v]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}[m];"
              f"color=black:s=1920x1080:r=30[bg];"
              f"[bg][m]overlay={x}:{y}:shortest=1[base];"
              f"[base][frm]overlay=(W-w)/2:0[v]")
    else:
        fc = (f"[0:v]scale=-2:1080[frm];[1:v]scale={w}:{h}[m];"
              f"color=black:s=1920x1080:r=30[bg];[bg][frm]overlay=(W-w)/2:0[t];"
              f"[t][m]overlay={x}:{y}:shortest=1[v]")
    media.run_ff([config.FFMPEG, "-y", "-loglevel", "error", "-i", frame_image, "-i", montage,
                  "-filter_complex", fc, "-map", "[v]", "-map", "1:a?",
                  "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                  "-video_track_timescale", "30000", "-c:a", "aac", "-ar", "48000", "-ac", "2", out])


def pause_montage(project_id: str, edl_id: str, camera_path: str, at: float,
                  count: int = 12, min_dur: float = 0.6, max_dur: float = 1.2,
                  frame_image: str = "", screen_rect: list | None = None,
                  frame_chroma: str | None = None, video_path: str = "",
                  output_path: str = "") -> dict[str, Any]:
    """Build a "millennial pause" montage of pure DEAD AIR and splice it into the composite at
    cut-time `at`. Uses the transcript silence map: every gap of at least `min_dur`s where NO
    words are spoken, spread across the video, up to `count`. Each clip is taken from INSIDE the
    gap (word-boundary margins trimmed) so it is only you pausing — breathing / room tone, never
    speech — capped at `max_dur`s. Audio is the raw camera's own track at that moment. Optionally
    wrap the montage inside `frame_image` (a decorative bezel like a Pause-O-Vision TV): it's
    centered on a black 1920x1080 canvas and the pauses play inside `screen_rect` = [x,y,w,h]
    (final-canvas px). Inserted at `at` in the camera-mixed master (default <edl>_mixed.mp4).
    Returns the spliced video + `montage_dur` — feed at/montage_dur to burn_captions to stay synced."""
    import shutil
    from .. import media
    pdir = project.require_project(project_id)
    slug = project.slugify(edl_id)
    t = project.read_transcript(project_id)
    picks = _pick_pauses(t.get("silences", []), int(count), float(min_dur))
    if not picks:
        raise ValueError(f"no silence gaps >= {min_dur}s to build a montage")
    if not os.path.isfile(camera_path):
        raise FileNotFoundError(f"camera file not found: {camera_path}")
    vid = video_path or str(pdir / "renders" / f"{slug}_mixed.mp4")
    if not os.path.isfile(vid):
        raise FileNotFoundError(f"composite not found: {vid} — run mix_camera first")
    # the camera capture's own audio track is silent (Source Record), so pull the montage audio
    # (room tone / breathing during the pause) from the synced screen source if it has audio.
    screen_src = project.read_meta(project_id).get("source")
    audio_src = screen_src if (screen_src and os.path.isfile(screen_src)) else None
    work = pdir / "renders" / "_montage"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    parts = []
    margin = 0.06   # stay clear of the word boundaries on both sides of the gap
    for i, s in enumerate(picks):
        start = float(s["start"]) + margin
        dur = min(float(max_dur), float(s["dur"]) - 2 * margin)
        if dur < 0.15:
            continue
        cp = work / f"p{i:02d}.mp4"
        _norm_clip(camera_path, start, dur, str(cp), audio_src=audio_src)
        parts.append(cp)
    if not parts:
        raise ValueError("no usable dead-air clips after trimming word-boundary margins")
    montage = work / "montage.mp4"
    _concat_files(parts, str(montage))
    if frame_image:
        if not os.path.isfile(frame_image):
            raise FileNotFoundError(f"frame image not found: {frame_image}")
        framed = work / "framed.mp4"
        _frame_montage(str(montage), frame_image,
                       screen_rect or [484, 140, 951, 724], str(framed),
                       frame_chroma=frame_chroma)
        montage = framed
    montage_dur = media.probe(str(montage))["duration"]
    pre, post = work / "pre.mp4", work / "post.mp4"
    _norm_clip(vid, 0.0, float(at), str(pre))
    _norm_clip(vid, float(at), None, str(post))
    out = output_path or str(pdir / "renders" / f"{slug}_montage.mp4")
    _concat_files([pre, montage, post], out)
    return {"project_id": project_id, "edl_id": edl_id, "output": out,
            "inserted_at": round(float(at), 3), "montage_dur": round(float(montage_dur), 3),
            "clips_used": len(parts)}
