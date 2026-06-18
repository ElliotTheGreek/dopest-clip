"""Audio quality-check, routed through the provider registry ("audio_qa" capability).

quality_check() resolves the active, configured audio_qa provider and calls
provider.audio_qa(audio_path, prompt, model) -> {"text": str}. No persistence — the report
is returned to the caller. No silent fallback (registry.get raises if unconfigured).
"""

from __future__ import annotations

from ..providers import registry


def quality_check(audio_path, prompt: str | None = None, model: str | None = None) -> dict:
    """Run a QA analysis over `audio_path`.

    Returns {"text": <report>, "provider": <name>}.
    """
    provider = registry.get("audio_qa")
    result = provider.audio_qa(str(audio_path), prompt=prompt, model=model)
    out = {"text": result.get("text", "")}
    out["provider"] = provider.name
    return out
