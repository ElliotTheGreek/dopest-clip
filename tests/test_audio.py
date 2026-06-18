"""Audio subsystem tests — light venv, NO provider keys, NO network, ffmpeg may be ABSENT.

Two layers:
  * DSP command-builders are pure (build an ffmpeg arg list, run nothing) and asserted
    directly: loudnorm string, afftdn/arnndn, silenceremove params, volume, afade, amix
    inputs, convert flags.
  * tts/sfx/qa/asr routing is verified by monkeypatching registry.get to return a FAKE
    provider, asserting the bytes/dict are persisted to the expected audio_out_path.

A single real-ffmpeg smoke test is marked @pytest.mark.needs_ffmpeg.
"""

from __future__ import annotations

import shutil

import pytest

from dopest_clip import config, project
from dopest_clip.audio import dsp, tts, asr, sfx, qa
from dopest_clip.providers import registry


# --- helpers ----------------------------------------------------------------------

def _make_project(projects_root, project_id="proj-1"):
    """Create a minimal valid project (meta.json present) so require_project passes."""
    project.ensure_project(project_id)
    project.write_meta(project_id, {"source": "x.mp4", "duration": 1.0})
    return project_id


# --- DSP command builders (pure) --------------------------------------------------

def test_normalize_cmd():
    cmd = dsp._normalize_cmd("in.wav", "out.wav")
    assert cmd[0] == config.FFMPEG
    assert "-af" in cmd
    af = cmd[cmd.index("-af") + 1]
    assert af == "loudnorm=I=-14:TP=-1.5:LRA=11"
    assert cmd[-1] == "out.wav"


def test_denoise_cmd_afftdn():
    cmd = dsp._denoise_cmd("in.wav", "out.wav", "afftdn")
    assert cmd[cmd.index("-af") + 1] == "afftdn"


def test_denoise_cmd_arnndn():
    cmd = dsp._denoise_cmd("in.wav", "out.wav", "arnndn")
    assert cmd[cmd.index("-af") + 1] == "arnndn"


def test_denoise_cmd_bad_method_raises():
    with pytest.raises(ValueError):
        dsp._denoise_cmd("in.wav", "out.wav", "nope")


def test_trim_silence_cmd_params():
    cmd = dsp._trim_silence_cmd("in.wav", "out.wav", threshold_db=-40, min_silence_s=0.25)
    af = cmd[cmd.index("-af") + 1]
    assert af.startswith("silenceremove=")
    assert "start_periods=1" in af
    assert "start_duration=0.25" in af
    assert "start_threshold=-40dB" in af
    assert "stop_periods=-1" in af
    assert "stop_duration=0.25" in af
    assert "stop_threshold=-40dB" in af


def test_gain_cmd():
    cmd = dsp._gain_cmd("in.wav", "out.wav", -6.0)
    assert cmd[cmd.index("-af") + 1] == "volume=-6dB"
    cmd2 = dsp._gain_cmd("in.wav", "out.wav", 3.5)
    assert cmd2[cmd2.index("-af") + 1] == "volume=3.5dB"


def test_fade_cmd_in_and_out():
    cmd = dsp._fade_cmd("in.wav", "out.wav", duration=10.0, fade_in_s=1.0, fade_out_s=2.0)
    af = cmd[cmd.index("-af") + 1]
    assert "afade=t=in:st=0:d=1" in af
    # out fade starts at duration - fade_out_s = 8.0
    assert "afade=t=out:st=8:d=2" in af


def test_fade_cmd_out_only():
    cmd = dsp._fade_cmd("in.wav", "out.wav", duration=5.0, fade_in_s=0.0, fade_out_s=1.5)
    af = cmd[cmd.index("-af") + 1]
    assert af == "afade=t=out:st=3.5:d=1.5"


def test_fade_cmd_requires_a_fade():
    with pytest.raises(ValueError):
        dsp._fade_cmd("in.wav", "out.wav", duration=5.0, fade_in_s=0.0, fade_out_s=0.0)


def test_mix_cmd_inputs_and_weights():
    cmd = dsp._mix_cmd(["a.wav", "b.wav", "c.wav"], "out.wav", weights=[1.0, 0.5, 0.25])
    # three -i inputs in order
    assert cmd.count("-i") == 3
    assert cmd[cmd.index("-i") + 1] == "a.wav"
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "amix=inputs=3" in fc
    assert "duration=longest" in fc
    assert "normalize=0" in fc
    assert "weights=1 0.5 0.25" in fc


def test_mix_cmd_no_weights():
    cmd = dsp._mix_cmd(["a.wav", "b.wav"], "out.wav")
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "amix=inputs=2" in fc
    assert "weights" not in fc


def test_mix_cmd_weight_length_mismatch_raises():
    with pytest.raises(ValueError):
        dsp._mix_cmd(["a.wav", "b.wav"], "out.wav", weights=[1.0])


def test_mix_cmd_empty_raises():
    with pytest.raises(ValueError):
        dsp._mix_cmd([], "out.wav")


def test_convert_cmd_all_flags():
    cmd = dsp._convert_cmd("in.mp3", "out.wav", fmt="wav", sample_rate=44100, channels=2)
    assert cmd[cmd.index("-ar") + 1] == "44100"
    assert cmd[cmd.index("-ac") + 1] == "2"
    assert cmd[cmd.index("-f") + 1] == "wav"
    assert cmd[-1] == "out.wav"


def test_convert_cmd_minimal():
    cmd = dsp._convert_cmd("in.mp3", "out.wav")
    assert "-ar" not in cmd
    assert "-ac" not in cmd
    assert "-f" not in cmd


# --- DSP output resolution --------------------------------------------------------

def test_resolve_out_explicit(tmp_path):
    p = dsp._resolve_out(str(tmp_path / "sub" / "o.wav"), None, None, "wav")
    assert str(p).endswith("o.wav")
    assert p.parent.exists()


def test_resolve_out_project(projects_root):
    pid = _make_project(projects_root)
    p = dsp._resolve_out(None, pid, "thing", "wav")
    assert p == project.audio_out_path(pid, "thing", "wav")


def test_resolve_out_both_raises(projects_root, tmp_path):
    pid = _make_project(projects_root)
    with pytest.raises(ValueError):
        dsp._resolve_out(str(tmp_path / "o.wav"), pid, "x", "wav")


def test_resolve_out_neither_raises():
    with pytest.raises(ValueError):
        dsp._resolve_out(None, None, None, "wav")


# --- DSP op writes via monkeypatched run_ff (no ffmpeg) ---------------------------

def test_normalize_op_calls_run_ff(monkeypatch, tmp_path):
    captured = {}

    def fake_run_ff(cmd):
        captured["cmd"] = cmd
        return ""

    monkeypatch.setattr(dsp.media, "run_ff", fake_run_ff)
    out = tmp_path / "norm.wav"
    res = dsp.normalize("in.wav", out=str(out))
    assert res["out"] == str(out)
    assert captured["cmd"][captured["cmd"].index("-af") + 1] == "loudnorm=I=-14:TP=-1.5:LRA=11"


def test_fade_op_probes_duration(monkeypatch, tmp_path):
    monkeypatch.setattr(dsp.media, "probe", lambda src: {"duration": 12.0})
    captured = {}
    monkeypatch.setattr(dsp.media, "run_ff", lambda cmd: captured.setdefault("cmd", cmd) or "")
    out = tmp_path / "f.wav"
    res = dsp.fade("in.wav", out=str(out), fade_out_s=2.0)
    assert res["duration"] == 12.0
    af = captured["cmd"][captured["cmd"].index("-af") + 1]
    assert "afade=t=out:st=10:d=2" in af


def test_mix_op_into_project(monkeypatch, projects_root):
    pid = _make_project(projects_root)
    captured = {}
    monkeypatch.setattr(dsp.media, "run_ff", lambda cmd: captured.setdefault("cmd", cmd) or "")
    res = dsp.mix(["a.wav", "b.wav"], project_id=pid, name="bed")
    assert res["out"] == str(project.audio_out_path(pid, "bed", "wav"))
    assert "-filter_complex" in captured["cmd"]


# --- cloud routing: tts / sfx / qa / asr ------------------------------------------

class FakeTTS:
    name = "faketts"

    def tts(self, text, voice=None, fmt="mp3", **opts):
        self.last = {"text": text, "voice": voice, "fmt": fmt, "opts": opts}
        return b"TTSBYTES"


class FakeSFX:
    name = "fakesfx"

    def sfx(self, prompt, duration=None, **opts):
        self.last = {"prompt": prompt, "duration": duration, "opts": opts}
        return b"SFXBYTES"


class FakeQA:
    name = "fakeqa"

    def audio_qa(self, audio_path, prompt=None, model=None):
        self.last = {"audio_path": audio_path, "prompt": prompt, "model": model}
        return {"text": "PASS 9/10"}


class FakeASR:
    name = "fakeasr"

    def asr(self, src, **opts):
        self.last = {"src": src, "opts": opts}
        return {"text": "hello world", "segments": [{"start": 0, "end": 1, "text": "hi"}]}


def test_tts_routes_and_persists(monkeypatch, projects_root):
    pid = _make_project(projects_root)
    fake = FakeTTS()
    monkeypatch.setattr(registry, "get", lambda cap: fake if cap == "tts" else (_ for _ in ()).throw(AssertionError(cap)))
    res = tts.synthesize("read this", project_id=pid, voice="v1", fmt="mp3", name="line1")
    expected = project.audio_out_path(pid, "line1", "mp3")
    assert res["out"] == str(expected)
    assert res["provider"] == "faketts"
    assert res["bytes"] == len(b"TTSBYTES")
    assert expected.read_bytes() == b"TTSBYTES"
    assert fake.last["voice"] == "v1"


def test_tts_explicit_out(monkeypatch, tmp_path):
    fake = FakeTTS()
    monkeypatch.setattr(registry, "get", lambda cap: fake)
    out = tmp_path / "deep" / "speech.mp3"
    res = tts.synthesize("hi", out=str(out))
    assert out.read_bytes() == b"TTSBYTES"
    assert res["out"] == str(out)


def test_tts_both_targets_raise(monkeypatch, projects_root, tmp_path):
    pid = _make_project(projects_root)
    monkeypatch.setattr(registry, "get", lambda cap: FakeTTS())
    with pytest.raises(ValueError):
        tts.synthesize("hi", project_id=pid, out=str(tmp_path / "o.mp3"))


def test_tts_no_target_raises(monkeypatch):
    monkeypatch.setattr(registry, "get", lambda cap: FakeTTS())
    with pytest.raises(ValueError):
        tts.synthesize("hi")


def test_sfx_routes_and_persists(monkeypatch, projects_root):
    pid = _make_project(projects_root)
    fake = FakeSFX()
    monkeypatch.setattr(registry, "get", lambda cap: fake)
    res = sfx.sound_effect("laser zap", project_id=pid, duration=3.0, name="zap")
    expected = project.audio_out_path(pid, "zap", "mp3")
    assert res["out"] == str(expected)
    assert res["provider"] == "fakesfx"
    assert expected.read_bytes() == b"SFXBYTES"
    assert fake.last["duration"] == 3.0


def test_qa_routes(monkeypatch, tmp_path):
    fake = FakeQA()
    monkeypatch.setattr(registry, "get", lambda cap: fake)
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"x")
    res = qa.quality_check(str(audio), prompt="loop clean?")
    assert res["text"] == "PASS 9/10"
    assert res["provider"] == "fakeqa"
    assert fake.last["prompt"] == "loop clean?"


def test_asr_engine_registry(monkeypatch, tmp_path):
    fake = FakeASR()
    monkeypatch.setattr(registry, "get", lambda cap: fake)
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    res = asr.transcribe_audio(str(audio), engine="registry", language="en")
    assert res["text"] == "hello world"
    assert res["engine"] == "registry"
    assert res["provider"] == "fakeasr"
    assert fake.last["opts"]["language"] == "en"


def test_asr_engine_local(monkeypatch, tmp_path):
    class FakeBackend:
        name = "fakelocal"

        def transcribe(self, audio_wav, model=None, language=None):
            return {"language": "en", "words": [{"i": 0, "w": "hi", "start": 0.0, "end": 0.5}]}

    import dopest_clip.stt as stt_mod
    monkeypatch.setattr(stt_mod, "get_backend", lambda name=None: FakeBackend())
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    res = asr.transcribe_audio(str(audio), engine="local")
    assert res["engine"] == "local"
    assert res["provider"] == "fakelocal"
    assert res["words"][0]["w"] == "hi"


def test_asr_auto_prefers_registry_when_configured(monkeypatch, tmp_path):
    fake = FakeASR()
    monkeypatch.setattr(asr, "_registry_stt_configured", lambda: True)
    monkeypatch.setattr(registry, "get", lambda cap: fake)
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    res = asr.transcribe_audio(str(audio), engine="auto")
    assert res["engine"] == "registry"


def test_asr_auto_falls_back_to_local(monkeypatch, tmp_path):
    class FakeBackend:
        name = "fakelocal"

        def transcribe(self, audio_wav, model=None, language=None):
            return {"words": []}

    monkeypatch.setattr(asr, "_registry_stt_configured", lambda: False)
    import dopest_clip.stt as stt_mod
    monkeypatch.setattr(stt_mod, "get_backend", lambda name=None: FakeBackend())
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    res = asr.transcribe_audio(str(audio), engine="auto")
    assert res["engine"] == "local"


def test_asr_bad_engine_raises():
    with pytest.raises(ValueError):
        asr.transcribe_audio("a.wav", engine="nope")


# --- real ffmpeg smoke test (skipped if ffmpeg absent) ----------------------------

needs_ffmpeg = pytest.mark.skipif(
    not (shutil.which(config.FFMPEG) and shutil.which(config.FFPROBE)),
    reason="ffmpeg/ffprobe not on PATH",
)


@needs_ffmpeg
def test_normalize_real_ffmpeg(tmp_path):
    """Generate a tone, normalize it, and confirm a non-empty output is produced."""
    from dopest_clip import media

    src = tmp_path / "tone.wav"
    media.run_ff([
        config.FFMPEG, "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-ar", "16000", "-ac", "1", str(src),
    ])
    out = tmp_path / "norm.wav"
    res = dsp.normalize(str(src), out=str(out))
    assert out.exists() and out.stat().st_size > 0
    assert res["out"] == str(out)
