"""Full coverage of the MCP control surface (api.py): every tool wrapper delegates to the
right implementation with the right arguments. Implementations are stubbed so this is pure
control-flow coverage (no ffmpeg/torch/network)."""

import dopest_clip.api as api
from dopest_clip.obs import camera_mix as cmx
from dopest_clip.obs import client as obc
from dopest_clip.obs import compositor as compc


def _spy(monkeypatch, obj, name, ret):
    """Replace obj.name with a spy returning ret; return a dict that captures the call."""
    cap = {}

    def f(*a, **k):
        cap["args"], cap["kwargs"] = a, k
        return ret
    monkeypatch.setattr(obj, name, f)
    return cap


# --- audio: local DSP -----------------------------------------------------------------

def test_audio_dsp_wrappers_delegate(monkeypatch):
    cases = [
        (lambda: api.audio_normalize("s.wav"), "normalize"),
        (lambda: api.audio_denoise("s.wav"), "denoise"),
        (lambda: api.audio_trim_silence("s.wav"), "trim_silence"),
        (lambda: api.audio_gain("s.wav", db=3), "gain"),
        (lambda: api.audio_fade("s.wav", fade_in_s=1), "fade"),
        (lambda: api.audio_mix(["a.wav", "b.wav"]), "mix"),
        (lambda: api.audio_convert("s.wav", fmt="mp3"), "convert"),
    ]
    for call, dele in cases:
        _spy(monkeypatch, api._dsp, dele, {"ok": dele})
        assert call() == {"ok": dele}


def test_cloud_audio_wrappers_delegate(monkeypatch):
    _spy(monkeypatch, api._tts, "synthesize", {"ok": "tts"})
    assert api.tts("hi") == {"ok": "tts"}
    _spy(monkeypatch, api._asr, "transcribe_audio", {"ok": "asr"})
    assert api.asr("a.wav") == {"ok": "asr"}
    _spy(monkeypatch, api._sfx, "sound_effect", {"ok": "sfx"})
    assert api.sfx("boom") == {"ok": "sfx"}
    _spy(monkeypatch, api._qa, "quality_check", {"ok": "qa"})
    assert api.audio_qa("a.wav") == {"ok": "qa"}


# --- image ----------------------------------------------------------------------------

def test_image_provider_wrappers_delegate(monkeypatch):
    _spy(monkeypatch, api._img_gen, "generate", {"ok": "gen"})
    assert api.image_generate("p", "model") == {"ok": "gen"}
    _spy(monkeypatch, api._img_gen, "edit", {"ok": "edit"})
    assert api.image_edit("i.png", "inst", "model") == {"ok": "edit"}
    _spy(monkeypatch, api._img_gen, "compose", {"ok": "comp"})
    assert api.image_compose(["a.png", "b.png"], "inst", "model") == {"ok": "comp"}
    _spy(monkeypatch, api._img_gen, "analyze", {"ok": "an"})
    assert api.image_analyze("i.png", "inst", "model") == {"ok": "an"}


def test_image_local_ops_wrappers_delegate(monkeypatch):
    cases = [
        (lambda: api.image_crop("s", "o", 0, 0, 1, 1), "crop"),
        (lambda: api.image_resize("s", "o", width=10), "resize"),
        (lambda: api.image_pad("s", "o"), "pad"),
        (lambda: api.image_square_canvas("s", "o"), "square_canvas"),
        (lambda: api.image_invert("s", "o"), "invert_colors"),
        (lambda: api.image_remove_background("s", "o"), "remove_background"),
        (lambda: api.image_svg_to_png("s", "o"), "svg_to_png"),
        (lambda: api.image_info("s"), "get_image_info"),
        (lambda: api.image_icon_set("s", "d"), "generate_icon_set"),
    ]
    for call, dele in cases:
        _spy(monkeypatch, api._img_ops, dele, {"ok": dele})
        assert call() == {"ok": dele}


# --- providers ------------------------------------------------------------------------

def test_provider_wrappers(monkeypatch):
    _spy(monkeypatch, api.registry, "list_providers", {"image": {"active": "gemini"}})
    assert api.list_providers() == {"image": {"active": "gemini"}}
    assert api.validate_provider("image") == {"image": {"active": "gemini"}}
    _spy(monkeypatch, api.registry, "set_provider", None)
    assert api.set_provider("image", "gemini") == {"capability": "image", "active": "gemini"}


# --- recording (lazy-imported obs.* delegates) ----------------------------------------

def test_recording_wrappers_delegate(monkeypatch):
    for fn_name, dele in [("list_devices", "list_devices"), ("start_recording", "start_recording"),
                          ("stop_recording", "stop_recording"), ("recording_status", "recording_status")]:
        _spy(monkeypatch, obc, dele, {"ok": dele})
        assert getattr(api, fn_name)() == {"ok": dele}
    _spy(monkeypatch, obc, "setup_scene", {"ok": "scene"})
    assert api.setup_scene("mon", "cam", "mic") == {"ok": "scene"}


def test_compose_and_mix_wrappers_delegate(monkeypatch):
    _spy(monkeypatch, compc, "compose", {"ok": "compose"})
    assert api.compose_camera("scr", "cam", [], "out") == {"ok": "compose"}
    cap = _spy(monkeypatch, cmx, "mix", {"ok": "mix"})
    assert api.mix_camera("p", "e", "cam", overlays=[{"x": 1}], bg_visible_until=3.0) == {"ok": "mix"}
    assert cap["kwargs"]["overlays"] == [{"x": 1}] and cap["kwargs"]["bg_visible_until"] == 3.0
    cap2 = _spy(monkeypatch, cmx, "short_clip", {"ok": "short"})
    assert api.make_short("p", "e", 0, 5, "Hook", overlays=[{"y": 2}]) == {"ok": "short"}
    assert cap2["kwargs"]["overlays"] == [{"y": 2}]


def test_get_cut_transcript_reads_text(monkeypatch, tmp_path):
    txt = tmp_path / "cut.txt"
    txt.write_text("hello cut transcript", encoding="utf-8")
    monkeypatch.setattr(cmx, "write_cut_transcript", lambda p, e: (str(tmp_path / "cut.json"), str(txt), 7))
    out = api.get_cut_transcript("p", "e")
    assert out["word_count"] == 7 and out["text"] == "hello cut transcript"
    assert out["cut_transcript_txt"] == str(txt)


def test_get_cut_transcript_missing_text_is_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(cmx, "write_cut_transcript", lambda p, e: (str(tmp_path / "x.json"), str(tmp_path / "missing.txt"), 0))
    out = api.get_cut_transcript("p", "e")
    assert out["text"] == "" and out["word_count"] == 0


def test_list_graphics_catalog_shape():
    g = api.list_graphics()
    assert set(g["kinds"]) == {"arrow", "ring", "box", "label"} and "custom" in g


def test_preview_track_explicit_video_path_delegates(monkeypatch):
    from dopest_clip.obs import tracking as trk
    cap = _spy(monkeypatch, trk, "preview", {"ok": "track"})
    out = api.preview_track("p", "cursor", video_path="C:/v.mp4", at=1.5)
    assert out == {"ok": "track"}
    assert cap["args"][0] == "C:/v.mp4" and cap["args"][1] == "cursor"
    assert cap["kwargs"]["at"] == 1.5


def test_preview_track_resolves_screen_and_camera_paths(monkeypatch, tmp_path):
    from dopest_clip import project
    from dopest_clip.obs import tracking as trk
    monkeypatch.setattr(project, "require_project", lambda pid: tmp_path)
    monkeypatch.setattr(project, "slugify", lambda e: "demo")
    cap = _spy(monkeypatch, trk, "preview", {"ok": "track"})
    api.preview_track("p", "face", edl_id="demo", source="screen")
    assert cap["args"][0] == str(tmp_path / "renders" / "demo.mp4")
    api.preview_track("p", "face", edl_id="demo", source="camera")
    assert cap["args"][0] == str(tmp_path / "camera" / "demo_cut.mp4")


# --- jobs (async render control) ------------------------------------------------------

def test_render_status_and_list_via_api():
    import time
    r = api.start_render("render", {"project_id": "p", "edl_obj_or_id": "e"})  # render will fail (no project)
    jid = r["job_id"]
    end = time.time() + 3.0
    while time.time() < end and api.render_status(jid).get("status") == "running":
        time.sleep(0.02)
    st = api.render_status(jid)
    assert st["status"] in ("done", "error")          # it ran in the background either way
    assert any(j["job_id"] == jid for j in api.list_render_jobs()["jobs"])


def test_render_status_unknown_job():
    assert "error" in api.render_status("job_nope")


def test_start_render_allowed_op_missing_from_registry(monkeypatch):
    # operation is whitelisted but absent from OPERATIONS -> the fn-None guard
    monkeypatch.setattr(api, "_RENDER_OPS", api._RENDER_OPS | {"ghost_render"})
    r = api.start_render("ghost_render", {})
    assert "error" in r and "unknown operation" in r["error"]
