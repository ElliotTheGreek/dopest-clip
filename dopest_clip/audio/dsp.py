"""Local ffmpeg audio DSP.

Each public function takes an input path and writes an output, returning a dict with at
least {"out": <str path>, ...}. The output is resolved one of two ways:
  * pass an explicit `out` path, OR
  * pass `project_id=` (and a `name`) and the result lands in the project's audio slot via
    project.audio_out_path(project_id, name, ext).

ffmpeg is invoked ONLY through media.run_ff. The ffmpeg argument list for every operation
is built by a small pure helper (`_*_cmd(...) -> list[str]`) so tests can assert the exact
filter strings/flags without running ffmpeg. There are no silent fallbacks: a request for a
denoise method or a missing duration that cannot be satisfied raises a clear error.
"""

from __future__ import annotations

from pathlib import Path

from .. import config, media, project


# --- output resolution ------------------------------------------------------------

def _resolve_out(out, project_id, name, ext: str) -> Path:
    """Pick the destination path. Exactly one of `out` or (`project_id` + `name`) must be
    given. Returns a Path; never silently invents a location."""
    if out is not None and project_id is not None:
        raise ValueError("pass either `out` or `project_id`, not both")
    if out is not None:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    if project_id is not None:
        if not name:
            raise ValueError("`name` is required when writing into a project")
        project.require_project(project_id)
        return project.audio_out_path(project_id, name, ext)
    raise ValueError("an output is required: pass `out=` or `project_id=` + `name=`")


def _ext_of(path: Path) -> str:
    return path.suffix.lstrip(".") or "wav"


# --- pure command builders --------------------------------------------------------

def _af_cmd(src: str, out: str, af: str) -> list[str]:
    """Generic single audio-filter encode: -i src -af <af> out."""
    return [config.FFMPEG, "-y", "-i", str(src), "-af", af, str(out)]


def _normalize_cmd(src: str, out: str) -> list[str]:
    return _af_cmd(src, out, "loudnorm=I=-14:TP=-1.5:LRA=11")


def _denoise_cmd(src: str, out: str, method: str = "afftdn") -> list[str]:
    if method == "afftdn":
        af = "afftdn"
    elif method == "arnndn":
        af = "arnndn"
    else:
        raise ValueError(f"unknown denoise method {method!r}; expected 'afftdn' or 'arnndn'")
    return _af_cmd(src, out, af)


def _trim_silence_cmd(
    src: str, out: str, threshold_db: float = -50.0, min_silence_s: float = 0.5
) -> list[str]:
    """silenceremove trimming leading AND trailing silence. stop_periods=-1 removes every
    trailing silent run; the leading run is removed by the start_* params."""
    af = (
        f"silenceremove="
        f"start_periods=1:start_duration={min_silence_s:g}:start_threshold={threshold_db:g}dB:"
        f"stop_periods=-1:stop_duration={min_silence_s:g}:stop_threshold={threshold_db:g}dB"
    )
    return _af_cmd(src, out, af)


def _gain_cmd(src: str, out: str, db: float) -> list[str]:
    return _af_cmd(src, out, f"volume={db:g}dB")


def _fade_cmd(
    src: str, out: str, duration: float, fade_in_s: float = 0.0, fade_out_s: float = 0.0
) -> list[str]:
    """afade in at t=0 and/or out ending at the clip end. `duration` is the source
    duration (from media.probe) needed to place the out-fade start."""
    parts: list[str] = []
    if fade_in_s and fade_in_s > 0:
        parts.append(f"afade=t=in:st=0:d={fade_in_s:g}")
    if fade_out_s and fade_out_s > 0:
        start = max(0.0, duration - fade_out_s)
        parts.append(f"afade=t=out:st={start:g}:d={fade_out_s:g}")
    if not parts:
        raise ValueError("fade requires fade_in_s > 0 and/or fade_out_s > 0")
    return _af_cmd(src, out, ",".join(parts))


def _mix_cmd(srcs: list[str], out: str, weights: list[float] | None = None) -> list[str]:
    """amix N inputs into one. Each src is its own -i input; amix=inputs=N. Optional
    per-input weights map to amix's `weights` (space-separated). duration=longest keeps the
    full mix; normalize=0 so explicit weights aren't auto-rescaled."""
    if not srcs:
        raise ValueError("mix requires at least one input")
    if weights is not None and len(weights) != len(srcs):
        raise ValueError(
            f"weights length ({len(weights)}) must match number of inputs ({len(srcs)})"
        )
    cmd = [config.FFMPEG, "-y"]
    for s in srcs:
        cmd += ["-i", str(s)]
    n = len(srcs)
    af = f"amix=inputs={n}:duration=longest:normalize=0"
    if weights is not None:
        af += ":weights=" + " ".join(f"{w:g}" for w in weights)
    cmd += ["-filter_complex", af, str(out)]
    return cmd


def _enhance_cmd(
    src: str, out: str, *, target_lufs: float = -16.0, denoise_db: float = 12.0,
    presence_db: float = 3.0, copy_video: bool = False,
) -> list[str]:
    """Speech clarity + consistency chain: high-pass (kill rumble/handling) -> spectral denoise
    -> compressor (even out loud/quiet) -> de-mud + presence EQ -> EBU R128 loudnorm (consistent
    loudness). `copy_video` keeps a video stream bit-exact and only re-encodes the audio."""
    chain = (
        "highpass=f=80,"
        f"afftdn=nr={denoise_db:g}:nf=-25,"
        "acompressor=threshold=-20dB:ratio=3:attack=5:release=150,"
        "equalizer=f=300:t=q:w=1.2:g=-2,"
        f"equalizer=f=3200:t=q:w=2:g={presence_db:g},"
        f"loudnorm=I={target_lufs:g}:TP=-1.5:LRA=11"
    )
    cmd = [config.FFMPEG, "-y", "-i", str(src)]
    if copy_video:
        cmd += ["-c:v", "copy"]
    cmd += ["-af", chain, "-c:a", "aac", "-b:a", "192k", str(out)]
    return cmd


def _convert_cmd(
    src: str,
    out: str,
    fmt: str | None = None,
    sample_rate: int | None = None,
    channels: int | None = None,
) -> list[str]:
    """Format / sample-rate / channel conversion. The container/codec is inferred from the
    output extension; -f is set only when an explicit `fmt` is given."""
    cmd = [config.FFMPEG, "-y", "-i", str(src)]
    if sample_rate is not None:
        cmd += ["-ar", str(sample_rate)]
    if channels is not None:
        cmd += ["-ac", str(channels)]
    if fmt is not None:
        cmd += ["-f", fmt]
    cmd += [str(out)]
    return cmd


# --- public operations ------------------------------------------------------------

def normalize(src, out=None, *, project_id=None, name="normalized") -> dict:
    """EBU R128 loudness normalization to I=-14:TP=-1.5:LRA=11."""
    dst = _resolve_out(out, project_id, name, _ext_of(Path(out)) if out else "wav")
    cmd = _normalize_cmd(str(src), str(dst))
    media.run_ff(cmd)
    return {"out": str(dst), "filter": "loudnorm=I=-14:TP=-1.5:LRA=11"}


def denoise(src, out=None, method="afftdn", *, project_id=None, name="denoised") -> dict:
    """Spectral denoise via `afftdn` (default) or RNN denoise via `arnndn`."""
    dst = _resolve_out(out, project_id, name, _ext_of(Path(out)) if out else "wav")
    cmd = _denoise_cmd(str(src), str(dst), method)
    media.run_ff(cmd)
    return {"out": str(dst), "method": method}


def trim_silence(
    src, out=None, threshold_db=-50.0, min_silence_s=0.5, *, project_id=None, name="trimmed"
) -> dict:
    """Remove leading and trailing silence (silenceremove)."""
    dst = _resolve_out(out, project_id, name, _ext_of(Path(out)) if out else "wav")
    cmd = _trim_silence_cmd(str(src), str(dst), threshold_db, min_silence_s)
    media.run_ff(cmd)
    return {"out": str(dst), "threshold_db": threshold_db, "min_silence_s": min_silence_s}


def gain(src, out=None, db=0.0, *, project_id=None, name="gain") -> dict:
    """Apply a fixed gain in dB (volume filter)."""
    dst = _resolve_out(out, project_id, name, _ext_of(Path(out)) if out else "wav")
    cmd = _gain_cmd(str(src), str(dst), db)
    media.run_ff(cmd)
    return {"out": str(dst), "db": db}


def fade(src, out=None, fade_in_s=0.0, fade_out_s=0.0, *, project_id=None, name="faded") -> dict:
    """Apply an in- and/or out-fade. Reads source duration via media.probe to place the
    out-fade."""
    dst = _resolve_out(out, project_id, name, _ext_of(Path(out)) if out else "wav")
    duration = float(media.probe(str(src)).get("duration", 0.0))
    cmd = _fade_cmd(str(src), str(dst), duration, fade_in_s, fade_out_s)
    media.run_ff(cmd)
    return {"out": str(dst), "duration": duration, "fade_in_s": fade_in_s, "fade_out_s": fade_out_s}


def mix(srcs, out=None, weights=None, *, project_id=None, name="mixed") -> dict:
    """Mix several inputs into one track (amix), with optional per-input weights."""
    srcs = [str(s) for s in srcs]
    dst = _resolve_out(out, project_id, name, _ext_of(Path(out)) if out else "wav")
    cmd = _mix_cmd(srcs, str(dst), weights)
    media.run_ff(cmd)
    return {"out": str(dst), "inputs": srcs, "weights": weights}


def enhance(
    src, out=None, target_lufs=-16.0, denoise_db=12.0, presence_db=3.0,
    *, project_id=None, name="enhanced",
) -> dict:
    """Make voice audio clearer + more consistent in one pass (high-pass -> denoise -> compress
    -> presence EQ -> loudnorm). Works on an audio file OR a finished VIDEO — if the input has a
    video stream it's copied bit-exact and only the audio is re-encoded. Tune `target_lufs`
    (loudness), `denoise_db` (noise reduction), `presence_db` (clarity boost)."""
    dst = _resolve_out(out, project_id, name, _ext_of(Path(out)) if out else "wav")
    has_video = bool(media.probe(str(src)).get("width"))
    cmd = _enhance_cmd(str(src), str(dst), target_lufs=target_lufs, denoise_db=denoise_db,
                       presence_db=presence_db, copy_video=has_video)
    media.run_ff(cmd)
    return {"out": str(dst), "target_lufs": target_lufs, "denoise_db": denoise_db,
            "presence_db": presence_db, "video_copied": has_video}


def convert(
    src, out=None, fmt=None, sample_rate=None, channels=None, *, project_id=None, name="converted"
) -> dict:
    """Convert format / sample rate / channel count. When writing into a project the output
    extension comes from `fmt` (defaulting to wav)."""
    if out is not None:
        dst = _resolve_out(out, project_id, None, _ext_of(Path(out)))
    else:
        dst = _resolve_out(None, project_id, name, fmt or "wav")
    cmd = _convert_cmd(str(src), str(dst), fmt, sample_rate, channels)
    media.run_ff(cmd)
    return {"out": str(dst), "fmt": fmt, "sample_rate": sample_rate, "channels": channels}
