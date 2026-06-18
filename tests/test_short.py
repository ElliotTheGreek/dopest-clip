"""Short-form / camera-mix additions.

The GPU/NVENC render paths (rvm_matte, composite_gpu, vertical_clip) are not run here —
there is no deterministic GPU render to assert on. Instead this covers the pure logic
(list_graphics, write_cut_transcript), the precondition errors (short_clip without a cut
or a matte), and the mix() BACKEND ROUTING (GPU vs CPU-rembg vs raw inset) by stubbing
the heavy callables and asserting which one ran.
"""

import sys
import types

import pytest

from dopest_clip import api, project
from dopest_clip.obs import camera_mix


def _seed(projects_root, pid, transcript, edl_obj):
    """Write a project tree (meta + transcript + edl) directly — no ffmpeg."""
    project.ensure_project(pid)
    src = str(projects_root / "fake.mp4")
    project.write_meta(pid, {"project_id": pid, "source": src, "duration": 10.0,
                             "fps": 30.0, "width": 1920, "height": 1080, "has_audio": True})
    full = {"project_id": pid, "source": src, "duration": 10.0, "fps": 30.0,
            "width": 1920, "height": 1080, "language": transcript["language"],
            "words": transcript["words"], "silences": transcript["silences"]}
    project.write_json(project.transcript_json_path(pid), full)
    project.write_json(project.edl_path(pid, edl_obj["edl_id"]), edl_obj)


# --- list_graphics (pure catalog) -----------------------------------------------------

def test_list_graphics_catalog():
    g = api.list_graphics()
    assert set(g["kinds"]) == {"arrow", "ring", "box", "label"}
    assert g["kinds"]["arrow"]["anchor"].startswith("tip")
    assert "svg" in g["custom"] and "image" in g["custom"]
    assert "keyframes" in g["animation"]
    assert g["used_by"].startswith("compose_camera")


# --- _cuda_available routing helper ---------------------------------------------------

def test_cuda_available_true_when_torch_reports_device(monkeypatch):
    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert camera_mix._cuda_available() is True


def test_cuda_available_false_when_no_device(monkeypatch):
    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert camera_mix._cuda_available() is False


def test_cuda_available_false_on_import_error(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert camera_mix._cuda_available() is False


# --- write_cut_transcript (pure: cleanup + resolve + remap + reindex) ------------------

def test_write_cut_transcript_reindexes_and_writes(projects_root, synthetic_transcript):
    edl = {"edl_id": "e1", "segments": [{"from_word": 0, "to_word": 5, "label": "all"}]}
    _seed(projects_root, "p1", synthetic_transcript, edl)
    cj, ct, n = camera_mix.write_cut_transcript("p1", "e1")
    assert n == 6
    import json
    words = json.loads(open(cj, encoding="utf-8").read())["words"]
    assert [w["i"] for w in words] == [0, 1, 2, 3, 4, 5]      # contiguous, re-indexed
    assert words[0]["start"] < 0.5                            # re-based onto the cut timeline
    assert words == sorted(words, key=lambda w: w["start"])   # monotonic cut-timeline times
    txt = open(ct, encoding="utf-8").read()
    assert "hello world this is a test" in txt
    assert "(#0)" in txt


def test_get_cut_transcript_op_returns_text(projects_root, synthetic_transcript):
    edl = {"edl_id": "e1", "segments": [{"from_word": 0, "to_word": 2, "label": "h"}]}
    _seed(projects_root, "p2", synthetic_transcript, edl)
    r = api.get_cut_transcript("p2", "e1")
    assert r["word_count"] == 3
    assert "hello world this" in r["text"]
    assert r["cut_transcript_txt"].endswith(".cut_transcript.txt")


# --- short_clip precondition errors (no GPU needed) -----------------------------------

def test_short_clip_errors_without_cut_screen(projects_root, synthetic_transcript):
    edl = {"edl_id": "e1", "segments": [{"from_word": 0, "to_word": 5, "label": "a"}]}
    _seed(projects_root, "p3", synthetic_transcript, edl)
    with pytest.raises(FileNotFoundError, match="cut screen not found"):
        camera_mix.short_clip("p3", "e1", 0, 3, "Hook")


def test_short_clip_errors_without_matte(projects_root, synthetic_transcript):
    edl = {"edl_id": "e1", "segments": [{"from_word": 0, "to_word": 5, "label": "a"}]}
    _seed(projects_root, "p4", synthetic_transcript, edl)
    # create the cut screen so the first check passes, but no matte fgr/pha
    cut = project.render_path("p4", "e1")
    cut.parent.mkdir(parents=True, exist_ok=True)
    cut.write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="camera matte not found"):
        camera_mix.short_clip("p4", "e1", 0, 3, "Hook")


# --- mix() backend routing (stub the heavy callables, assert which path ran) ----------

@pytest.fixture
def mix_project(projects_root, synthetic_transcript):
    edl = {"edl_id": "e1", "segments": [{"from_word": 0, "to_word": 5, "label": "a"}]}
    _seed(projects_root, "pmix", synthetic_transcript, edl)
    cut = project.render_path("pmix", "e1")
    cut.parent.mkdir(parents=True, exist_ok=True)
    cut.write_bytes(b"x")
    cam = projects_root / "camera.mkv"
    cam.write_bytes(b"x")
    return "pmix", "e1", str(cam)


def _stub_heavy(monkeypatch, compositor_mod):
    """Stub every heavy callable mix() can reach so no ffmpeg/torch runs."""
    monkeypatch.setattr(camera_mix, "_resolve_segments", lambda p, e: [(0.0, 1.0)])
    monkeypatch.setattr(camera_mix, "cut_video_only", lambda *a, **k: None)
    monkeypatch.setattr(camera_mix, "rvm_matte", lambda *a, **k: None)
    monkeypatch.setattr(camera_mix, "composite_gpu",
                        lambda *a, **k: {"output": "o", "size": [1, 1], "duration": 1.0})
    monkeypatch.setattr(camera_mix, "composite",
                        lambda *a, **k: {"output": "o", "size": [1, 1], "duration": 1.0})
    monkeypatch.setattr(compositor_mod, "compose",
                        lambda *a, **k: {"output": "o", "size": [1, 1], "duration": 1.0})


def test_mix_uses_gpu_when_cuda_available(monkeypatch, mix_project):
    from dopest_clip.obs import compositor
    pid, eid, cam = mix_project
    _stub_heavy(monkeypatch, compositor)
    monkeypatch.setattr(camera_mix, "_cuda_available", lambda: True)
    info = camera_mix.mix(pid, eid, cam, remove_background=True)
    assert info["matte_backend"] == "rvm-gpu"
    assert info["background_removed"] is True


def test_mix_falls_back_to_rembg_cpu_without_cuda(monkeypatch, mix_project):
    from dopest_clip.obs import compositor
    pid, eid, cam = mix_project
    _stub_heavy(monkeypatch, compositor)
    monkeypatch.setattr(camera_mix, "_cuda_available", lambda: False)
    info = camera_mix.mix(pid, eid, cam, remove_background=True)
    assert info["matte_backend"] == "rembg-cpu"


def test_mix_raw_inset_when_no_background_removal(monkeypatch, mix_project):
    from dopest_clip.obs import compositor
    pid, eid, cam = mix_project
    _stub_heavy(monkeypatch, compositor)
    monkeypatch.setattr(camera_mix, "_cuda_available", lambda: True)
    info = camera_mix.mix(pid, eid, cam, remove_background=False)
    assert info["matte_backend"] == "raw-inset"
    assert info["background_removed"] is False


# --- unified GPU compose: effect params route to composite_gpu ------------------------

def test_mix_routes_full_effect_stack_to_gpu_composite(monkeypatch, mix_project):
    """overlays / blurs / screen_keyframes / bg_visible_until all flow through to the GPU
    composite, with the un-matted cut camera passed for the bg-visible phase."""
    pid, eid, cam = mix_project
    captured = {}
    monkeypatch.setattr(camera_mix, "_resolve_segments", lambda p, e: [(0.0, 1.0)])
    monkeypatch.setattr(camera_mix, "cut_video_only", lambda *a, **k: None)
    monkeypatch.setattr(camera_mix, "rvm_matte", lambda *a, **k: None)
    monkeypatch.setattr(camera_mix, "composite_gpu",
                        lambda *a, **k: captured.update(k) or {"output": "o", "size": [1, 1], "duration": 1.0})
    monkeypatch.setattr(camera_mix, "_cuda_available", lambda: True)

    info = camera_mix.mix(
        pid, eid, cam, remove_background=True,
        overlays=[{"kind": "ring", "keyframes": [{"t": 0, "pos": [0.5, 0.5], "scale": 0.1}]}],
        blurs=[{"shape": "circle", "keyframes": [{"t": 0, "pos": [0.5, 0.5], "scale": 0.3}]}],
        screen_keyframes=[{"t": 0, "zoom": 1.0, "focus": [0.5, 0.5]}],
        bg_visible_until=3.0)
    assert info["matte_backend"] == "rvm-gpu"
    assert captured["overlays"] and captured["blurs"] and captured["screen_keyframes"]
    assert captured["bg_visible_until"] == 3.0
    assert captured["cut_cam_path"]   # un-matted cut camera passed for the bg-visible phase


def test_mix_bg_toggle_alone_uses_gpu(monkeypatch, mix_project):
    pid, eid, cam = mix_project
    seen = {}
    monkeypatch.setattr(camera_mix, "_resolve_segments", lambda p, e: [(0.0, 1.0)])
    monkeypatch.setattr(camera_mix, "cut_video_only", lambda *a, **k: None)
    monkeypatch.setattr(camera_mix, "rvm_matte", lambda *a, **k: None)
    monkeypatch.setattr(camera_mix, "composite_gpu",
                        lambda *a, **k: seen.update(k) or {"output": "o", "size": [1, 1], "duration": 1.0})
    monkeypatch.setattr(camera_mix, "_cuda_available", lambda: True)
    info = camera_mix.mix(pid, eid, cam, remove_background=False, bg_visible_until=2.0)
    assert info["matte_backend"] == "rvm-gpu"   # an effect forces GPU compose even w/o remove_background
    assert seen["bg_visible_until"] == 2.0


def test_prep_overlays_rasterizes_kind_and_defaults(projects_root):
    prepped = camera_mix._prep_overlays(
        [{"kind": "ring", "color": "#ff0000", "keyframes": [{"t": 0, "pos": [0.5, 0.5], "scale": 0.1}]}], 1920)
    assert len(prepped) == 1
    o = prepped[0]
    assert o["arr"].ndim == 3 and o["arr"].shape[2] == 4   # HxW RGBA
    assert o["t_out"] is None and o["anchor"] == [0.5, 0.5]


def test_prep_overlays_empty_is_noop():
    assert camera_mix._prep_overlays(None, 1920) == []
    assert camera_mix._prep_overlays([], 1920) == []
