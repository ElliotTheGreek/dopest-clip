"""Provider registry tests — run in the light venv with NO provider keys and NO network.

Strategy:
  * Import-time + list_providers / validate behavior is checked against a key-less env.
  * Each provider's HTTP call is verified by monkeypatching requests.post / requests.get
    with a fake that records the URL/headers/payload and returns a canned response, so we
    assert the wire shape without ever leaving the process.
  * Live tests would be marked @pytest.mark.needs_network; we have none that hit the
    network unmarked.
"""

from __future__ import annotations

import base64
import json

import pytest

import dopest_clip.providers.registry as reg  # the module (functions, ProviderError, env_str)
from dopest_clip.providers import (
    openai as openai_mod,
    fish as fish_mod,
    elevenlabs as el_mod,
    gemini as gemini_mod,
    openrouter as orouter_mod,
    flowdot as flowdot_mod,
)


# --- env scrubbing --------------------------------------------------------------

PROVIDER_ENV = [
    "OPENAI_API_KEY", "OPENAI_BASE_URL",
    "FISH_AUDIO_API_KEY", "FISH_AUDIO_BASE_URL", "FISH_AUDIO_CACHE_DIR",
    "ELEVENLABS_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "OPENROUTER_REFERER", "OPENROUTER_TITLE",
    "FLOWDOT_API_KEY", "FLOWDOT_BASE_URL",
] + [f"DOPEST_PROVIDER_{c.upper()}" for c in reg.CAPABILITIES]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    """Strip every provider key/selection env var and isolate PROVIDERS_TOML + caches,
    then reset the registry's in-memory state. Runs for every test."""
    for name in PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    # Point PROVIDERS_TOML at an isolated (nonexistent-by-default) temp path.
    from dopest_clip import config
    monkeypatch.setattr(config, "PROVIDERS_TOML", tmp_path / "providers.toml")
    # Fish cache to temp so no real disk cache is touched.
    monkeypatch.setenv("FISH_AUDIO_CACHE_DIR", str(tmp_path / "fishcache"))
    reg._reset_for_tests()
    yield
    reg._reset_for_tests()


class FakeResp:
    def __init__(self, *, status=200, json_body=None, content=b"", text=""):
        self.status_code = status
        self.ok = 200 <= status < 300
        self._json = json_body
        self.content = content
        self.text = text or (json.dumps(json_body) if json_body is not None else "")

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


# --- env_str gotcha -------------------------------------------------------------

def test_env_str_empty_is_none(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "   ")
    assert reg.env_str("OPENAI_BASE_URL") is None
    assert openai_mod._base_url() == "https://api.openai.com/v1"


# --- list_providers / validate --------------------------------------------------

def test_list_providers_shape_and_unconfigured():
    info = reg.list_providers()
    assert set(info) == set(reg.CAPABILITIES)
    # llm has openai/openrouter/flowdot, all unconfigured, openai active by default
    llm = info["llm"]
    assert llm["active"] == "openai"
    assert llm["providers"]["openai"]["active"] is True
    for cap, capinfo in info.items():
        for name, pinfo in capinfo["providers"].items():
            assert pinfo["configured"] is False, f"{cap}/{name} should be unconfigured"


@pytest.mark.parametrize("provider", [
    openai_mod.OpenAIProvider(),
    fish_mod.FishProvider(),
    el_mod.ElevenLabsProvider(),
    gemini_mod.GeminiProvider(),
    orouter_mod.OpenRouterProvider(),
    flowdot_mod.FlowDotProvider(),
])
def test_validate_no_key_never_raises(provider):
    status = provider.validate()  # must not raise
    assert status["ok"] is False
    assert isinstance(status["detail"], str) and status["detail"]


# --- get() resolution -----------------------------------------------------------

def test_get_raises_when_unconfigured():
    with pytest.raises(reg.ProviderError) as ei:
        reg.get("llm")
    assert "openai" in str(ei.value) and "not usable" in str(ei.value)


def test_get_unknown_capability_raises():
    with pytest.raises(reg.ProviderError):
        reg.get("nope")


def test_get_works_once_configured(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    p = reg.get("llm")
    assert p.name == "openai"


def test_set_provider_selects_and_get_uses_it(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    reg.set_provider("llm", "openrouter")
    assert reg.get("llm").name == "openrouter"
    # selection persisted to the temp toml
    from dopest_clip import config
    assert config.PROVIDERS_TOML.exists()
    text = config.PROVIDERS_TOML.read_text(encoding="utf-8")
    assert "[active]" in text and 'llm = "openrouter"' in text


def test_set_provider_unknown_raises():
    with pytest.raises(reg.ProviderError):
        reg.set_provider("llm", "bogus")


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("DOPEST_PROVIDER_LLM", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    assert reg.get("llm").name == "openrouter"


def test_toml_active_table_read(monkeypatch):
    from dopest_clip import config
    config.PROVIDERS_TOML.write_text('[active]\nllm = "openrouter"\n', encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    # in-memory override is empty, so toml wins over the code default
    assert reg.get("llm").name == "openrouter"


# --- fish.tts wire shape + cache ------------------------------------------------

def test_fish_tts_builds_request_and_caches(monkeypatch):
    monkeypatch.setenv("FISH_AUDIO_API_KEY", "fk-test")
    calls = []

    def fake_post(url, **kw):
        calls.append((url, kw))
        return FakeResp(status=200, content=b"AUDIOBYTES")

    monkeypatch.setattr(fish_mod.requests, "post", fake_post)
    p = fish_mod.FishProvider()
    out = p.tts("hello world", voice="vid123", fmt="mp3", prosody={"speed": 1.1})
    assert out == b"AUDIOBYTES"
    url, kw = calls[0]
    assert url == "https://api.fish.audio/v1/tts"
    assert kw["headers"]["Authorization"] == "Bearer fk-test"
    assert kw["headers"]["model"] == "speech-1.6"
    body = kw["json"]
    assert body["text"] == "hello world"
    assert body["reference_id"] == "vid123"
    assert body["format"] == "mp3"
    assert body["prosody"] == {"speed": 1.1}

    # second identical call hits the disk cache -> no new POST
    out2 = p.tts("hello world", voice="vid123", fmt="mp3", prosody={"speed": 1.1})
    assert out2 == b"AUDIOBYTES"
    assert len(calls) == 1, "cached call should not re-POST"

    # changing prosody busts the cache (gotcha: key includes prosody)
    p.tts("hello world", voice="vid123", fmt="mp3", prosody={"speed": 2.0})
    assert len(calls) == 2


def test_fish_cache_key_includes_all_fields():
    k1 = fish_mod._LRUDiskCache.generate_key("t", "v", "mp3", None)
    k2 = fish_mod._LRUDiskCache.generate_key("t", "v", "wav", None)
    k3 = fish_mod._LRUDiskCache.generate_key("t", "v2", "mp3", None)
    k4 = fish_mod._LRUDiskCache.generate_key("t", "v", "mp3", {"speed": 1.1})
    assert len({k1, k2, k3, k4}) == 4


def test_fish_asr_multipart(monkeypatch):
    monkeypatch.setenv("FISH_AUDIO_API_KEY", "fk-test")
    captured = {}

    def fake_post(url, **kw):
        captured["url"] = url
        captured["files"] = kw.get("files")
        captured["headers"] = kw.get("headers")
        return FakeResp(status=200, json_body={"text": "transcribed", "segments": [{"start": 0, "end": 1, "text": "hi"}]})

    monkeypatch.setattr(fish_mod.requests, "post", fake_post)
    p = fish_mod.FishProvider()
    res = p.asr(b"\x00\x01wavbytes", language="en")
    assert res["text"] == "transcribed"
    assert res["segments"][0]["text"] == "hi"
    assert captured["url"] == "https://api.fish.audio/v1/asr"
    assert "audio" in captured["files"]
    assert captured["headers"]["Authorization"] == "Bearer fk-test"


# --- elevenlabs.sfx wire shape --------------------------------------------------

def test_elevenlabs_sfx_request(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-test")
    captured = {}

    def fake_post(url, **kw):
        captured["url"] = url
        captured["headers"] = kw["headers"]
        captured["json"] = kw["json"]
        return FakeResp(status=200, content=b"SFXBYTES")

    monkeypatch.setattr(el_mod.requests, "post", fake_post)
    p = el_mod.ElevenLabsProvider()
    out = p.sfx("laser zap", duration=3.0, prompt_influence=0.5)
    assert out == b"SFXBYTES"
    assert captured["url"].startswith("https://api.elevenlabs.io/v1/sound-generation?output_format=")
    assert captured["headers"]["xi-api-key"] == "el-test"
    assert captured["headers"]["Accept"] == "audio/mpeg"
    assert captured["json"]["text"] == "laser zap"
    assert captured["json"]["model_id"] == "eleven_text_to_sound_v2"
    assert captured["json"]["duration_seconds"] == 3.0
    assert captured["json"]["prompt_influence"] == 0.5


# --- openai.audio_qa wire shape -------------------------------------------------

def test_openai_audio_qa_request(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"ID3rawmp3bytes")
    captured = {}

    def fake_post(url, **kw):
        captured["url"] = url
        captured["headers"] = kw["headers"]
        captured["json"] = kw["json"]
        return FakeResp(status=200, json_body={"choices": [{"message": {"content": "PASS 9/10"}}]})

    monkeypatch.setattr(openai_mod.requests, "post", fake_post)
    p = openai_mod.OpenAIProvider()
    res = p.audio_qa(str(audio), prompt="is the loop clean?")
    assert res["text"] == "PASS 9/10"
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    body = captured["json"]
    assert body["model"] == "gpt-4o-audio-preview"
    assert body["modalities"] == ["text"]
    user = body["messages"][1]["content"]
    audio_part = next(p for p in user if p["type"] == "input_audio")
    assert audio_part["input_audio"]["format"] == "mp3"
    # base64 of the file bytes
    assert audio_part["input_audio"]["data"] == base64.b64encode(b"ID3rawmp3bytes").decode()


def test_openai_audio_qa_rejects_bad_format(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    bad = tmp_path / "clip.xyz"
    bad.write_bytes(b"x")
    p = openai_mod.OpenAIProvider()
    with pytest.raises(ValueError):
        p.audio_qa(str(bad))


# --- gemini.generate_image wire shape -------------------------------------------

def test_gemini_generate_image_request(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    img_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\nPIXELS").decode()
    captured = {}

    def fake_post(url, **kw):
        captured["url"] = url
        captured["params"] = kw.get("params")
        captured["json"] = kw["json"]
        return FakeResp(status=200, json_body={
            "candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": img_b64}}]}}]
        })

    monkeypatch.setattr(gemini_mod.requests, "post", fake_post)
    p = gemini_mod.GeminiProvider()
    out = p.generate_image("a red cube", model="gemini-2.5-flash-image")
    assert out == b"\x89PNG\r\n\x1a\nPIXELS"
    assert captured["url"] == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent"
    assert captured["params"] == {"key": "g-test"}
    cfg = captured["json"]["generationConfig"]
    assert "IMAGE" in cfg["responseModalities"]
    assert captured["json"]["contents"][0]["parts"][0]["text"] == "a red cube"


def test_gemini_analyze_image_text(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")

    def fake_post(url, **kw):
        return FakeResp(status=200, json_body={
            "candidates": [{"content": {"parts": [{"text": "a red cube on white"}]}}]
        })

    monkeypatch.setattr(gemini_mod.requests, "post", fake_post)
    p = gemini_mod.GeminiProvider()
    res = p.analyze_image(b"\x89PNG\r\n\x1a\n...", "what is this?", model="gemini-2.5-flash")
    assert res == {"text": "a red cube on white"}


# --- openrouter.complete wire shape ---------------------------------------------

def test_openrouter_complete_request(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    captured = {}

    def fake_post(url, **kw):
        captured["url"] = url
        captured["headers"] = kw["headers"]
        captured["json"] = kw["json"]
        return FakeResp(status=200, json_body={"choices": [{"message": {"content": "hi there"}}]})

    monkeypatch.setattr(orouter_mod.requests, "post", fake_post)
    p = orouter_mod.OpenRouterProvider()
    msgs = [{"role": "user", "content": "hi"}]
    res = p.complete(msgs, model="openai/gpt-4o-mini", temperature=0.2)
    assert res["text"] == "hi there"
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer or-test"
    assert captured["json"]["model"] == "openai/gpt-4o-mini"
    assert captured["json"]["messages"] == msgs
    assert captured["json"]["temperature"] == 0.2


# --- flowdot unverified caps ----------------------------------------------------

def test_flowdot_llm_requires_base_url(monkeypatch):
    monkeypatch.setenv("FLOWDOT_API_KEY", "fd-test")
    p = flowdot_mod.FlowDotProvider()
    with pytest.raises(RuntimeError) as ei:
        p.complete([{"role": "user", "content": "hi"}], model="x")
    assert "FLOWDOT_BASE_URL" in str(ei.value)


def test_flowdot_image_not_implemented(monkeypatch):
    monkeypatch.setenv("FLOWDOT_API_KEY", "fd-test")
    monkeypatch.setenv("FLOWDOT_BASE_URL", "https://example.test/v1")
    p = flowdot_mod.FlowDotProvider()
    with pytest.raises(NotImplementedError):
        p.generate_image("x", model="m")


def test_flowdot_llm_with_base_url(monkeypatch):
    monkeypatch.setenv("FLOWDOT_API_KEY", "fd-test")
    monkeypatch.setenv("FLOWDOT_BASE_URL", "https://example.test/v1")
    captured = {}

    def fake_post(url, **kw):
        captured["url"] = url
        return FakeResp(status=200, json_body={"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(flowdot_mod.requests, "post", fake_post)
    p = flowdot_mod.FlowDotProvider()
    res = p.complete([{"role": "user", "content": "hi"}], model="m")
    assert res["text"] == "ok"
    assert captured["url"] == "https://example.test/v1/chat/completions"


# --- a marked live test that is skipped by default ------------------------------

@pytest.mark.needs_network
def test_live_openai_validate():  # pragma: no cover - requires a real key + network
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("no OPENAI_API_KEY")
    assert openai_mod.OpenAIProvider().validate()["ok"] is True
