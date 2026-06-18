"""ffmpeg/ffprobe wrappers: probe, extract audio, render an EDL, style, frames, thumbs.

The single ffmpeg runner lives here and is reused by the audio DSP subsystem too.
"""

import json
import subprocess
from pathlib import Path

from . import config


def run_ff(cmd: list[str]) -> str:
    """Run an ffmpeg/ffprobe command and return stdout, raising on non-zero exit.

    stdin=DEVNULL is REQUIRED when this runs inside an MCP stdio server: otherwise
    ffmpeg inherits the server's stdin pipe (the agent <-> server channel) and
    blocks/consumes it, hanging the whole server. Harmless in a shell.
    """
    proc = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd[:2])}...\n{proc.stderr[-4000:]}"
        )
    return proc.stdout


# Back-compat private alias used within this module.
_run = run_ff


def _parse_fps(rate: str) -> float:
    if not rate or rate in ("0/0", "N/A"):
        return 0.0
    if "/" in rate:
        num, den = rate.split("/")
        den_f = float(den)
        return float(num) / den_f if den_f else 0.0
    return float(rate)


def probe(src: str) -> dict:
    """Return {duration, fps, width, height, has_audio} for a media file."""
    out = _run([
        config.FFPROBE, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", src,
    ])
    info = json.loads(out)
    duration = float(info.get("format", {}).get("duration", 0.0))
    width = height = 0
    fps = 0.0
    has_audio = False
    for st in info.get("streams", []):
        if st.get("codec_type") == "video" and width == 0:
            width = int(st.get("width", 0))
            height = int(st.get("height", 0))
            fps = _parse_fps(st.get("avg_frame_rate") or st.get("r_frame_rate") or "0")
        if st.get("codec_type") == "audio":
            has_audio = True
    return {
        "duration": round(duration, 3),
        "fps": round(fps, 3),
        "width": width,
        "height": height,
        "has_audio": has_audio,
    }


def extract_audio(src: str, out_wav: Path) -> None:
    """Extract 16kHz mono PCM wav (what the STT backends expect)."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    _run([
        config.FFMPEG, "-y", "-i", src,
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(out_wav),
    ])


def _build_filtergraph(segments: list[tuple[float, float]], crossfade_ms: int) -> str:
    """trim/atrim each segment, then concat in order. One re-encode, frame-accurate;
    reordering and non-contiguous reuse both work since every segment reads from [0]."""
    parts: list[str] = []
    concat_inputs: list[str] = []
    fade = max(0, crossfade_ms) / 1000.0
    for i, (s, e) in enumerate(segments):
        dur = max(0.0, e - s)
        parts.append(f"[0:v]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS[v{i}]")
        afilters = [f"atrim=start={s:.3f}:end={e:.3f}", "asetpts=PTS-STARTPTS"]
        if fade > 0 and dur > 0:
            f = min(fade, dur / 2)
            afilters.append(f"afade=t=in:st=0:d={f:.3f}")
            afilters.append(f"afade=t=out:st={max(0.0, dur - f):.3f}:d={f:.3f}")
        parts.append(f"[0:a]{','.join(afilters)}[a{i}]")
        concat_inputs.append(f"[v{i}][a{i}]")
    n = len(segments)
    parts.append(f"{''.join(concat_inputs)}concat=n={n}:v=1:a=1[outv][outa]")
    return ";".join(parts)


def build_filtergraph(segments: list[tuple[float, float]], crossfade_ms: int) -> str:
    """Public accessor — useful for tests and for showing the graph in the editor."""
    return _build_filtergraph(segments, crossfade_ms)


def render(
    src: str,
    segments: list[tuple[float, float]],
    out_mp4: Path,
    filtergraph_file: Path,
    crossfade_ms: int = config.CROSSFADE_MS,
) -> None:
    """Render the ordered list of (start, end) source-time segments to out_mp4.

    Uses -filter_complex_script (graph written to a file) to avoid Windows
    command-length limits when an EDL has many segments.
    """
    if not segments:
        raise ValueError("no segments to render")
    graph = _build_filtergraph(segments, crossfade_ms)
    filtergraph_file.parent.mkdir(parents=True, exist_ok=True)
    filtergraph_file.write_text(graph, encoding="utf-8")
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    _run([
        config.FFMPEG, "-y", "-i", src,
        "-filter_complex_script", str(filtergraph_file),
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        str(out_mp4),
    ])


# --- aspect targets, styling encode, frames, thumbnail ---

ASPECT_DIMS = {
    "source": None,
    "9:16": (1080, 1920),
    "4:5": (1080, 1350),
    "1:1": (1080, 1080),
    "16:9": (1920, 1080),
}


def aspect_dims(aspect: str, src_w: int, src_h: int) -> tuple[int, int]:
    if aspect == "source" or aspect not in ASPECT_DIMS or ASPECT_DIMS[aspect] is None:
        return src_w, src_h
    return ASPECT_DIMS[aspect]


def escape_filter_path(p) -> str:
    """Escape a Windows path for use inside an ffmpeg filter argument."""
    return str(p).replace("\\", "/").replace(":", "\\:")


def aspect_filter(src_w: int, src_h: int, tw: int, th: int, mode: str) -> str:
    """Video-filter chain converting src to tw x th.
    mode: 'crop' (center-crop to fill) or 'pad' (fit + blurred bars)."""
    if (src_w, src_h) == (tw, th):
        return ""
    if mode == "pad":
        return (
            f"split=2[bg][fg];"
            f"[bg]scale={tw}:{th}:force_original_aspect_ratio=increase,crop={tw}:{th},boxblur=20:2[bgb];"
            f"[fg]scale={tw}:{th}:force_original_aspect_ratio=decrease[fgs];"
            f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2"
        )
    return f"scale={tw}:{th}:force_original_aspect_ratio=increase,crop={tw}:{th}"


def style_encode(
    in_video: Path,
    out_video: Path,
    target_w: int,
    target_h: int,
    src_w: int,
    src_h: int,
    aspect_mode: str = "crop",
    ass_path: Path | None = None,
    loudnorm: bool = False,
    fonts_dir: Path | None = None,
) -> None:
    """Single-encode styling pass over a cut clip: aspect transform + burned ASS
    captions + optional loudness normalization. (Reframe uses its own frame-pipe encode.)"""
    chains = []
    af = aspect_filter(src_w, src_h, target_w, target_h, aspect_mode)
    if af:
        chains.append(af)
    if ass_path is not None:
        sub = f"subtitles=filename='{escape_filter_path(ass_path)}'"
        if fonts_dir is not None:
            sub += f":fontsdir='{escape_filter_path(fonts_dir)}'"
        chains.append(sub)
    vf = ",".join(chains) if chains else "null"

    cmd = [config.FFMPEG, "-y", "-i", str(in_video), "-vf", vf]
    if loudnorm:
        cmd += ["-af", "loudnorm=I=-14:TP=-1.5:LRA=11"]
    cmd += [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", str(out_video),
    ]
    _run(cmd)


def grab_frame(in_video, at_t: float, out_png: Path, grid: bool = True, divisions: int = 10) -> tuple[int, int]:
    """Extract a single frame at at_t seconds; optionally overlay a labeled pixel grid so the
    agent/editor can read source coordinates for focus shots. Returns (width, height)."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    _run([config.FFMPEG, "-y", "-ss", f"{at_t:.3f}", "-i", str(in_video),
          "-frames:v", "1", str(out_png)])
    from PIL import Image, ImageDraw
    img = Image.open(out_png).convert("RGB")
    w, h = img.size
    if grid:
        d = ImageDraw.Draw(img)
        for k in range(1, divisions):
            gx = round(w * k / divisions)
            gy = round(h * k / divisions)
            d.line([(gx, 0), (gx, h)], fill=(255, 0, 0), width=1)
            d.line([(0, gy), (w, gy)], fill=(255, 0, 0), width=1)
            d.text((gx + 2, 2), str(gx), fill=(255, 255, 0))
            d.text((2, gy + 2), str(gy), fill=(255, 255, 0))
        d.text((2, h - 14), f"{w}x{h}", fill=(0, 255, 0))
        img.save(out_png)
    return w, h


def extract_thumbnail(
    in_video: Path,
    out_image: Path,
    at_t: float,
    target_w: int,
    target_h: int,
    src_w: int,
    src_h: int,
    aspect_mode: str = "crop",
    text: str | None = None,
    font_file: Path | None = None,
) -> None:
    chains = []
    af = aspect_filter(src_w, src_h, target_w, target_h, aspect_mode)
    if af:
        chains.append(af)
    if text:
        safe = text.replace("\\", "").replace(":", "\\:").replace("'", "’")
        ff = f":fontfile='{escape_filter_path(font_file)}'" if font_file else ""
        chains.append(
            f"drawtext=text='{safe}'{ff}:fontcolor=white:fontsize={int(target_h*0.06)}:"
            f"borderw=4:bordercolor=black:x=(w-tw)/2:y=h*0.12"
        )
    vf = ",".join(chains) if chains else "null"
    out_image.parent.mkdir(parents=True, exist_ok=True)
    _run([
        config.FFMPEG, "-y", "-ss", f"{at_t:.3f}", "-i", str(in_video),
        "-frames:v", "1", "-vf", vf, str(out_image),
    ])
