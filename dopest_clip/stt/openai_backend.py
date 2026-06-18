"""OpenAI Whisper backend (whisper-1).

Alternate / verifier backend. Uses the file-transcription API with word-level
timestamp granularity. The OpenAI file API caps uploads at 25 MB, so long source
audio is split at silence boundaries near ~9-min marks (16kHz mono s16le is ~32 kB/s,
so 25 MB ~= 13 min; ~9 min leaves a safe margin) and the per-chunk word timings are
shifted by the chunk offset and re-indexed so the merged list stays contiguous.

The openai SDK (plus httpx/certifi) is imported lazily inside transcribe(), and the
API key is read from the OPENAI_API_KEY env var at CALL time (raising a clear error
when absent) — so importing this module never requires the dep or a configured key.
"""

import os
import subprocess
import tempfile

from . import compute_silences
from .. import config

_MAX_BYTES = 25 * 1024 * 1024
# 16kHz mono s16le = ~32 kB/s, so 25 MB ~= 13 min. Target ~9 min for a safe margin.
_CHUNK_TARGET_S = 540.0


def _probe_duration(path: str) -> float:
    out = subprocess.run(
        [config.FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
    ).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def _silence_intervals(path: str) -> list[tuple[float, float]]:
    err = subprocess.run(
        [config.FFMPEG, "-i", path, "-af", "silencedetect=noise=-30dB:d=0.3",
         "-f", "null", "-"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
    ).stderr
    starts: list[float] = []
    out: list[tuple[float, float]] = []
    for line in err.splitlines():
        if "silence_start:" in line:
            starts.append(float(line.split("silence_start:")[1].strip()))
        elif "silence_end:" in line and starts:
            end = float(line.split("silence_end:")[1].split("|")[0].strip())
            out.append((starts.pop(0), end))
    return out


def _split_points(path: str, duration: float) -> list[tuple[float, float]]:
    """Segment [0, duration] into <=~9-min pieces, cutting at a silence midpoint near
    each boundary so no word is split; falls back to a hard cut if no silence is near."""
    sil = _silence_intervals(path)
    points = [0.0]
    target = _CHUNK_TARGET_S
    while target < duration - 30:
        cands = [(a + b) / 2 for (a, b) in sil
                 if abs((a + b) / 2 - target) < 90 and (a + b) / 2 > points[-1] + 30]
        pt = min(cands, key=lambda m: abs(m - target)) if cands else target
        if pt <= points[-1]:
            pt = points[-1] + _CHUNK_TARGET_S
        points.append(min(round(pt, 3), duration))
        target = points[-1] + _CHUNK_TARGET_S
    points.append(duration)
    pts = sorted({p for p in points if 0 <= p <= duration})
    return list(zip(pts[:-1], pts[1:]))


def _extract_wav(path: str, start: float, end: float) -> str:
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    subprocess.run(
        [config.FFMPEG, "-y", "-loglevel", "error", "-ss", str(start), "-to", str(end),
         "-i", path, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", tmp],
        check=True, stdin=subprocess.DEVNULL,
    )
    return tmp


class OpenAIBackend:
    def __init__(self):
        # Defer client construction (and the API-key requirement) to transcribe(), so
        # get_backend("openai") and importing this module never need a key or the SDK.
        self._client = None

    def _build_client(self):
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set; cannot use the 'openai' STT backend"
            )
        try:
            import certifi
            import httpx
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover - exercised only without the dep
            raise RuntimeError(
                "the 'openai' STT backend requires the optional 'stt' extra "
                "(openai + httpx + certifi). Install with: pip install dopest-clip[stt]"
            ) from e

        # An EMPTY OPENAI_BASE_URL env var (as set by some launchers) is otherwise read
        # by the SDK as the base URL, producing scheme-less request URLs that fail as a
        # bare "Connection error." Coerce empty -> None and fall back to the public
        # endpoint. The pinned certifi client + trust_env=False keep TLS/proxy behavior
        # deterministic across launch environments.
        base_url = (os.environ.get("OPENAI_BASE_URL") or "").strip() or "https://api.openai.com/v1"
        http_client = httpx.Client(verify=certifi.where(), trust_env=False, timeout=120)
        return OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)

    def _client_lazy(self):
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _api_words(self, wav_path, language, offset, start_index):
        """One whisper-1 call on a (sub-)file; words shifted by `offset` seconds and
        re-indexed from `start_index`."""
        client = self._client_lazy()
        with open(wav_path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model="whisper-1", file=f, response_format="verbose_json",
                timestamp_granularities=["word"],
                **({"language": language} if language else {}),
            )
        words = []
        for w in (getattr(resp, "words", None) or []):
            words.append({
                "i": start_index + len(words),
                "w": str(w.word).strip(),
                "start": round(float(w.start) + offset, 3),
                "end": round(float(w.end) + offset, 3),
            })
        return words, (getattr(resp, "language", None) or language)

    def transcribe(self, audio_wav, model: str | None = None, language: str | None = None) -> dict:
        # model is accepted for contract symmetry; whisper-1 is the only OpenAI STT
        # model with word-level timestamps, so it is fixed.
        path = str(audio_wav)
        size = os.path.getsize(path)
        duration = _probe_duration(path)

        if size <= _MAX_BYTES:
            words, lang = self._api_words(path, language, 0.0, 0)
        else:
            # Split into <25MB chunks at silence boundaries near ~9-min marks, then
            # transcribe each and merge with cumulative time offsets. Always pass the
            # ORIGINAL `language` arg (the API wants ISO-639-1 or nothing; a detected
            # name like "english" is invalid as an input param).
            words, lang = [], None
            for (s, e) in _split_points(path, duration):
                chunk = _extract_wav(path, s, e)
                try:
                    w, detected = self._api_words(chunk, language, s, len(words))
                finally:
                    try:
                        os.remove(chunk)
                    except OSError:
                        pass
                lang = lang or detected
                words.extend(w)

        silences = compute_silences(words, config.MIN_SILENCE)
        return {
            "language": lang,
            "words": words,
            "silences": silences,
            "untimed_tokens_dropped": 0,
        }
