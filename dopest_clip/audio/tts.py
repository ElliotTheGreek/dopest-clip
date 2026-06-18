"""Text-to-speech, routed through the provider registry ("tts" capability).

synthesize() asks the registry for the active, configured TTS provider, calls
provider.tts(text, voice, fmt, **opts) -> bytes, and persists the bytes to disk. The
destination is the project's audio slot when `project_id` is given, otherwise the explicit
`out` path. No silent fallback: a missing/unconfigured provider raises ProviderError out of
registry.get.
"""

from __future__ import annotations

from pathlib import Path

from .. import project
from ..providers import registry


def synthesize(
    text: str,
    project_id: str | None = None,
    out=None,
    voice: str | None = None,
    fmt: str = "mp3",
    name: str = "tts",
    **opts,
) -> dict:
    """Synthesize `text` to speech and write the audio to disk.

    Returns {"out": <path>, "provider": <name>, "fmt": fmt, "voice": voice, "bytes": n}.
    """
    if out is not None and project_id is not None:
        raise ValueError("pass either `out` or `project_id`, not both")

    provider = registry.get("tts")
    audio = provider.tts(text, voice=voice, fmt=fmt, **opts)

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
        "fmt": fmt,
        "voice": voice,
        "bytes": len(audio),
    }
