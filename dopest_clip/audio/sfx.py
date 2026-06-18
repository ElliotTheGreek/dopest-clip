"""Sound-effect generation, routed through the provider registry ("sfx" capability).

sound_effect() resolves the active, configured SFX provider, calls
provider.sfx(prompt, duration, **opts) -> bytes, and persists the bytes. Destination is the
project audio slot when `project_id` is given, else the explicit `out` path. No silent
fallback — registry.get raises if no provider is configured.

SFX providers (ElevenLabs) return mp3, so the default output extension is mp3.
"""

from __future__ import annotations

from pathlib import Path

from .. import project
from ..providers import registry


def sound_effect(
    prompt: str,
    project_id: str | None = None,
    out=None,
    duration: float | None = None,
    fmt: str = "mp3",
    name: str = "sfx",
    **opts,
) -> dict:
    """Generate a sound effect from `prompt` and write the audio to disk.

    Returns {"out": <path>, "provider": <name>, "duration": duration, "bytes": n}.
    """
    if out is not None and project_id is not None:
        raise ValueError("pass either `out` or `project_id`, not both")

    provider = registry.get("sfx")
    audio = provider.sfx(prompt, duration=duration, **opts)

    if out is not None:
        dst = Path(out)
        dst.parent.mkdir(parents=True, exist_ok=True)
    elif project_id is not None:
        project.require_project(project_id)
        dst = project.audio_out_path(project_id, name, fmt)
    else:
        raise ValueError("an output is required: pass `out=` or `project_id=`")

    dst.write_bytes(audio)
    return {
        "out": str(dst),
        "provider": provider.name,
        "duration": duration,
        "bytes": len(audio),
    }
