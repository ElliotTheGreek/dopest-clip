"""Transcription (ASR) with an explicit engine choice — never a silent guess.

Two homes for transcription:
  * the registry "stt" capability — a cloud provider (Fish ASR) whose method is
    `provider.asr(audio_path_or_bytes, **opts) -> {"text", "segments"}`, and
  * dopest_clip.stt.get_backend() — the local/cloud STT engines (whisperx / openai) whose
    method is `backend.transcribe(audio_wav, model=None, language=None) -> {...words...}`.

`engine` selects which:
  * "registry" — use the registry STT provider (raises if none configured),
  * "local"    — use dopest_clip.stt.get_backend(),
  * "auto"     — prefer the registry STT provider IF one is configured (validate ok),
                 otherwise fall back to the local backend. The choice is reported back in
                 the result's "engine" field so it is never hidden.
"""

from __future__ import annotations

from ..providers import registry


_STT_CAPABILITY = "stt"  # the registry capability name for cloud ASR (Fish)


def _registry_stt_configured() -> bool:
    """True iff the active registry STT provider validates ok. Never raises (mirrors the
    registry's list_providers contract)."""
    try:
        info = registry.list_providers().get(_STT_CAPABILITY, {})
    except Exception:
        return False
    active = info.get("active")
    if not active:
        return False
    return bool(info.get("providers", {}).get(active, {}).get("configured"))


def transcribe_audio(src, engine: str = "auto", **opts) -> dict:
    """Transcribe `src` (a path; bytes also accepted by the registry engine).

    Returns the engine's native dict plus {"engine": "registry"|"local", "provider": <name>}.
    """
    if engine not in ("auto", "registry", "local"):
        raise ValueError(f"unknown engine {engine!r}; expected 'auto', 'registry', or 'local'")

    use_registry: bool
    if engine == "registry":
        use_registry = True
    elif engine == "local":
        use_registry = False
    else:  # auto: explicit, reported choice — prefer registry only when it is configured
        use_registry = _registry_stt_configured()

    if use_registry:
        provider = registry.get(_STT_CAPABILITY)
        result = dict(provider.asr(src, **opts))
        result["engine"] = "registry"
        result["provider"] = provider.name
        return result

    from ..stt import get_backend

    backend = get_backend(opts.pop("backend", None))
    result = dict(backend.transcribe(
        str(src),
        model=opts.pop("model", None),
        language=opts.pop("language", None),
    ))
    result["engine"] = "local"
    result["provider"] = getattr(backend, "name", backend.__class__.__name__)
    return result
