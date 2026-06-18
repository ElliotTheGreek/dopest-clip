"""Audio subsystem: local ffmpeg DSP + cloud audio routed through the provider registry.

Two halves:
  * dsp.py    — purely local ffmpeg signal processing (normalize, denoise, trim silence,
                gain, fade, mix, convert). Every operation runs through media.run_ff and
                the ffmpeg arg list is built by a small pure helper so it can be asserted
                without invoking ffmpeg.
  * tts.py / asr.py / sfx.py / qa.py — cloud audio. tts/sfx/qa resolve their provider via
                registry.get(<capability>); asr can use the registry STT provider or the
                local dopest_clip.stt backends, the choice made explicit via an `engine` arg.

Self-contained: imports only dopest_clip.*, stdlib, and declared deps. No silent fallbacks.
"""

from . import dsp, tts, asr, sfx, qa  # noqa: F401

__all__ = ["dsp", "tts", "asr", "sfx", "qa"]
