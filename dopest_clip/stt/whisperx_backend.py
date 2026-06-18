"""WhisperX backend: faster-whisper transcription + wav2vec2 forced alignment.

The free, local, GPU(/CPU) default. Forced alignment gives tight word-level
boundaries, which is what silence-aware cutting depends on. Device and compute type
come from config (DEVICE / COMPUTE_TYPE), model from config.WHISPERX_MODEL unless an
explicit model is passed to transcribe().

whisperx (and its torch dependency) is imported lazily so that merely importing this
module — e.g. for get_backend() class resolution — never requires torch.
"""

from . import compute_silences
from .. import config


class WhisperXBackend:
    def __init__(self):
        self.default_model = config.WHISPERX_MODEL
        self.device = config.DEVICE
        self.compute_type = config.COMPUTE_TYPE

    def transcribe(self, audio_wav, model: str | None = None, language: str | None = None) -> dict:
        try:
            import whisperx
        except ImportError as e:  # pragma: no cover - exercised only without the dep
            raise RuntimeError(
                "the 'whisperx' STT backend requires the optional 'stt' extra "
                "(whisperx + torch). Install with: pip install dopest-clip[stt]"
            ) from e

        model_name = model or self.default_model
        audio = whisperx.load_audio(str(audio_wav))

        asr = whisperx.load_model(
            model_name, self.device, compute_type=self.compute_type, language=language,
        )
        result = asr.transcribe(audio, batch_size=16)
        lang = result.get("language", language)

        align_model, metadata = whisperx.load_align_model(language_code=lang, device=self.device)
        aligned = whisperx.align(
            result["segments"], align_model, metadata, audio, self.device,
            return_char_alignments=False,
        )

        words: list[dict] = []
        dropped = 0
        for seg in aligned.get("segments", []):
            for w in seg.get("words", []):
                if w.get("start") is None or w.get("end") is None:
                    # A token (often a numeral/symbol) the aligner couldn't place in
                    # time. It cannot be a cut boundary, so drop it and keep indices
                    # contiguous (index only increments for timed words).
                    dropped += 1
                    continue
                words.append({
                    "i": len(words),
                    "w": str(w.get("word", "")).strip(),
                    "start": round(float(w["start"]), 3),
                    "end": round(float(w["end"]), 3),
                })

        silences = compute_silences(words, config.MIN_SILENCE)
        return {
            "language": lang,
            "words": words,
            "silences": silences,
            "untimed_tokens_dropped": dropped,
        }
