"""Speech-to-text subsystem.

A backend turns a 16kHz mono wav into a contiguously-indexed word list with tight
timings plus a derived silence map. The single contract every backend honors:

    backend.transcribe(audio_wav, model=None, language=None) -> {
        "language": str,
        "words": [{"i": int, "w": str, "start": float, "end": float}, ...],
        "silences": [{"start": float, "end": float, "dur": float}, ...],
        "untimed_tokens_dropped": int,
    }

Word indices are contiguous 0..N-1: tokens the aligner could not place in time are
dropped (they can never be a cut boundary) and the index only increments for timed
words. `ops.transcribe` depends on this exact shape.

Heavy deps (whisperx/torch, openai/httpx/certifi) are imported LAZILY inside the
backends, so importing this package costs nothing and the light test venv works.
"""

from .. import config


def get_backend(name: str | None = None):
    """Return an STT backend instance. `name` defaults to config.STT_BACKEND.

    The backend classes are imported lazily so this factory (and importing this
    module) never requires the heavy ML/cloud deps until a backend is actually used.
    """
    name = (name or config.STT_BACKEND).strip().lower()
    if name == "whisperx":
        from .whisperx_backend import WhisperXBackend
        return WhisperXBackend()
    if name == "openai":
        from .openai_backend import OpenAIBackend
        return OpenAIBackend()
    raise ValueError(f"unknown STT backend '{name}' (expected 'whisperx' or 'openai')")


def compute_silences(words: list[dict], min_silence: float = config.MIN_SILENCE) -> list[dict]:
    """Derive silence gaps from word timings — backend-agnostic, so the same silence
    map is available regardless of which STT produced the words.

    A gap between the running max end-time and the next word's start is a silence iff
    it is >= min_silence. Overlapping/out-of-order word boundaries cannot produce a
    negative gap because we track the running maximum end. Times are rounded to ms.
    """
    silences: list[dict] = []
    prev_end = 0.0
    for w in words:
        start = float(w["start"])
        gap = start - prev_end
        if gap >= min_silence:
            silences.append({
                "start": round(prev_end, 3),
                "end": round(start, 3),
                "dur": round(gap, 3),
            })
        prev_end = max(prev_end, float(w["end"]))
    return silences
