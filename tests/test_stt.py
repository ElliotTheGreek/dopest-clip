"""STT subsystem tests.

These run in the LIGHT venv (pytest, pytest-mock, Pillow, requests, mcp — NO
torch/whisperx/openai). So they never import the heavy libs: the factory is tested at
the class level, and the OpenAI backend is exercised against a FAKE openai/httpx/
certifi injected into sys.modules — no network, no real package, no API key needed.
"""

import sys
import types

import pytest

from dopest_clip import config
from dopest_clip.stt import compute_silences, get_backend
from dopest_clip.stt.openai_backend import OpenAIBackend
from dopest_clip.stt.whisperx_backend import WhisperXBackend


# --- compute_silences pure logic ---

def test_compute_silences_gap_above_threshold_is_silence():
    # 0.5s gap between word 0 (ends 0.5) and word 1 (starts 1.0) >= default 0.15.
    words = [
        {"i": 0, "w": "a", "start": 0.0, "end": 0.5},
        {"i": 1, "w": "b", "start": 1.0, "end": 1.5},
    ]
    sil = compute_silences(words, min_silence=0.15)
    assert sil == [{"start": 0.5, "end": 1.0, "dur": 0.5}]


def test_compute_silences_adjacent_below_threshold_not_silence():
    # 0.1s gap < 0.15 -> no silence. Leading gap before word 0 (0.0) is also none.
    words = [
        {"i": 0, "w": "a", "start": 0.0, "end": 0.5},
        {"i": 1, "w": "b", "start": 0.6, "end": 1.0},
    ]
    assert compute_silences(words, min_silence=0.15) == []


def test_compute_silences_leading_silence_counted():
    words = [{"i": 0, "w": "a", "start": 0.4, "end": 0.8}]
    sil = compute_silences(words, min_silence=0.15)
    assert sil == [{"start": 0.0, "end": 0.4, "dur": 0.4}]


def test_compute_silences_uses_running_max_end_no_negative_gap():
    # Overlapping word boundaries must not produce a spurious/negative silence.
    words = [
        {"i": 0, "w": "a", "start": 0.0, "end": 2.0},
        {"i": 1, "w": "b", "start": 0.5, "end": 1.0},  # nested inside word 0
        {"i": 2, "w": "c", "start": 2.05, "end": 2.5},  # gap 0.05 < threshold
    ]
    assert compute_silences(words, min_silence=0.15) == []


def test_compute_silences_default_min_silence_from_config():
    # Gap exactly equal to config.MIN_SILENCE counts (>=).
    g = config.MIN_SILENCE
    words = [
        {"i": 0, "w": "a", "start": 0.0, "end": 0.5},
        {"i": 1, "w": "b", "start": round(0.5 + g, 3), "end": 1.5},
    ]
    sil = compute_silences(words)
    assert len(sil) == 1
    assert sil[0]["start"] == 0.5


def test_compute_silences_no_trailing_silence():
    # Contract's compute_silences has no duration arg, so it never appends a trailing
    # silence past the last word.
    words = [{"i": 0, "w": "a", "start": 0.0, "end": 0.5}]
    assert compute_silences(words) == []


# --- factory returns right class without importing heavy libs ---

def test_get_backend_whisperx_class():
    b = get_backend("whisperx")
    assert isinstance(b, WhisperXBackend)


def test_get_backend_openai_class():
    # No OPENAI_API_KEY needed: construction defers the key requirement to transcribe().
    b = get_backend("openai")
    assert isinstance(b, OpenAIBackend)


def test_get_backend_default_uses_config(monkeypatch):
    monkeypatch.setattr(config, "STT_BACKEND", "whisperx")
    assert isinstance(get_backend(), WhisperXBackend)
    monkeypatch.setattr(config, "STT_BACKEND", "openai")
    assert isinstance(get_backend(), OpenAIBackend)


def test_get_backend_unknown_raises():
    with pytest.raises(ValueError):
        get_backend("nope")


def test_no_heavy_libs_imported_at_module_load():
    # Importing the STT package must not pull torch/whisperx/openai. Checked in a FRESH
    # interpreter so a full-install venv (or a prior test in the session) can't pollute
    # sys.modules and make this pass/fail spuriously.
    import subprocess
    code = (
        "import importlib, sys\n"
        "importlib.import_module('dopest_clip.stt')\n"
        "bad=[m for m in ('torch','whisperx','openai') if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"heavy deps imported by STT: {r.stdout}{r.stderr}"


# --- OpenAI backend: normalization, word-index, empty base_url coercion ---

class _FakeWord:
    def __init__(self, word, start, end):
        self.word = word
        self.start = start
        self.end = end


class _FakeResp:
    def __init__(self, words, language="english"):
        self.words = words
        self.language = language


class _FakeTranscriptions:
    def __init__(self, resp):
        self._resp = resp
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._resp


class _FakeAudio:
    def __init__(self, resp):
        self.transcriptions = _FakeTranscriptions(resp)


class _FakeOpenAIClient:
    last_kwargs = None

    def __init__(self, resp):
        self.audio = _FakeAudio(resp)


class _DummyFile:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patch_open(monkeypatch):
    """Stub builtins.open in the backend so _api_words can run without a real wav file;
    the fake OpenAI client never touches the returned handle."""
    import dopest_clip.stt.openai_backend as ob
    monkeypatch.setattr("builtins.open", lambda *a, **k: _DummyFile())
    return ob


@pytest.fixture
def fake_openai_env(monkeypatch):
    """Inject fake openai/httpx/certifi modules and capture the OpenAI() ctor kwargs."""
    captured = {}
    resp = _FakeResp([
        _FakeWord(" Hello ", 0.0, 0.5),
        _FakeWord("world", 1.0, 1.4),
    ])
    client = _FakeOpenAIClient(resp)

    def _openai_ctor(**kwargs):
        captured["kwargs"] = kwargs
        return client

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = _openai_ctor

    fake_httpx = types.ModuleType("httpx")
    fake_httpx.Client = lambda **kw: object()

    fake_certifi = types.ModuleType("certifi")
    fake_certifi.where = lambda: "/fake/cacert.pem"

    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)
    monkeypatch.setitem(sys.modules, "certifi", fake_certifi)
    return captured, client


def test_openai_missing_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    b = OpenAIBackend()
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        b._build_client()


def test_openai_empty_base_url_coerced_to_public(monkeypatch, fake_openai_env):
    captured, _ = fake_openai_env
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "")  # the launcher bug: empty string
    OpenAIBackend()._build_client()
    assert captured["kwargs"]["base_url"] == "https://api.openai.com/v1"


def test_openai_explicit_base_url_respected(monkeypatch, fake_openai_env):
    captured, _ = fake_openai_env
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.local/v1")
    OpenAIBackend()._build_client()
    assert captured["kwargs"]["base_url"] == "https://proxy.local/v1"
    assert captured["kwargs"]["api_key"] == "sk-test"


def test_openai_word_normalization_and_indexing(monkeypatch, fake_openai_env):
    _, client = fake_openai_env
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    # Small file path -> single API call (avoid ffprobe/ffmpeg + real file IO).
    import dopest_clip.stt.openai_backend as ob
    monkeypatch.setattr(ob.os.path, "getsize", lambda p: 1000)
    monkeypatch.setattr(ob, "_probe_duration", lambda p: 5.0)
    _patch_open(monkeypatch)

    out = OpenAIBackend().transcribe("dummy.wav")

    assert out["language"] == "english"
    assert out["untimed_tokens_dropped"] == 0
    words = out["words"]
    assert [w["i"] for w in words] == [0, 1]          # contiguous indices
    assert [w["w"] for w in words] == ["Hello", "world"]  # stripped
    assert words[0] == {"i": 0, "w": "Hello", "start": 0.0, "end": 0.5}
    # gap 1.0 - 0.5 = 0.5 >= MIN_SILENCE -> one silence
    assert out["silences"] == [{"start": 0.5, "end": 1.0, "dur": 0.5}]
    # whisper-1 with word granularity was requested
    assert client.audio.transcriptions.calls[0]["model"] == "whisper-1"
    assert client.audio.transcriptions.calls[0]["timestamp_granularities"] == ["word"]


def test_openai_chunk_offset_reindexing(monkeypatch, fake_openai_env):
    """Large file -> chunked path: each chunk's words are shifted by the chunk offset
    and re-indexed so the merged list stays contiguous."""
    _, client = fake_openai_env
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    import dopest_clip.stt.openai_backend as ob
    monkeypatch.setattr(ob.os.path, "getsize", lambda p: ob._MAX_BYTES + 1)
    monkeypatch.setattr(ob, "_probe_duration", lambda p: 1200.0)
    # Two chunks at fixed offsets; avoid ffmpeg entirely.
    monkeypatch.setattr(ob, "_split_points", lambda p, d: [(0.0, 600.0), (600.0, 1200.0)])
    monkeypatch.setattr(ob, "_extract_wav", lambda p, s, e: "chunk.wav")
    monkeypatch.setattr(ob.os, "remove", lambda p: None)
    _patch_open(monkeypatch)

    out = OpenAIBackend().transcribe("big.wav")
    words = out["words"]
    # two words per chunk * 2 chunks = 4, contiguous indices
    assert [w["i"] for w in words] == [0, 1, 2, 3]
    # second chunk's words are offset by 600s
    assert words[2]["start"] == 600.0
    assert words[3]["start"] == 601.0


# --- a real-model test would need GPU/network: mark + skip ---

@pytest.mark.needs_gpu
def test_whisperx_real_alignment_smoke():  # pragma: no cover
    pytest.skip("requires CUDA + whisperx model download; not run in the light venv")
