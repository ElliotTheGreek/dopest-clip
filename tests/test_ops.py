"""ops orchestration — pure-logic paths (validate_edl, slicing, catalogs, suggest brief).

ffmpeg/STT/reframe paths are not exercised here (no torch/cv2/ffmpeg in the light venv);
those are marked needs_ffmpeg / needs_gpu where present. Importing ops must not pull cv2/
ultralytics/torch — see test_imports_have_no_heavy_deps.
"""

import importlib.util
import sys

import pytest

from dopest_clip import ops, project

_HAS_ULTRALYTICS = importlib.util.find_spec("ultralytics") is not None


def _seed_project(projects_root, pid, transcript):
    """Write a project tree directly: dirs + meta.json + transcript.json (no ffmpeg)."""
    project.ensure_project(pid)
    meta = {
        "project_id": pid,
        "source": str(projects_root / "fake.mp4"),
        "duration": 10.0, "fps": 30.0, "width": 1920, "height": 1080, "has_audio": True,
    }
    project.write_meta(pid, meta)
    full = {
        "project_id": pid, "source": meta["source"], "duration": meta["duration"],
        "fps": meta["fps"], "width": meta["width"], "height": meta["height"],
        "language": transcript["language"], "words": transcript["words"],
        "silences": transcript["silences"],
    }
    project.write_json(project.transcript_json_path(pid), full)
    return meta


# --- validate_edl end-to-end (pure logic) ---------------------------------------------

def test_validate_edl_resolves_and_saves(projects_root, synthetic_transcript):
    _seed_project(projects_root, "proj1", synthetic_transcript)
    edl_obj = {"edl_id": "clip-a", "segments": [{"from_word": 0, "to_word": 2, "label": "h"}]}
    res = ops.validate_edl("proj1", edl_obj)
    assert "error" not in res
    assert res["reconstructed_text"] == "hello world this"
    assert res["edl_id"] == "clip-a"
    assert res["warnings"] == []
    # it was actually saved to disk
    assert project.edl_path("proj1", "clip-a").exists()
    assert "clip-a" in project.list_edl_ids("proj1")


def test_validate_edl_reports_cleanup(projects_root):
    transcript = {
        "language": "en",
        "words": [
            {"i": 0, "w": "so", "start": 0.0, "end": 0.3},
            {"i": 1, "w": "um", "start": 0.4, "end": 0.6},
            {"i": 2, "w": "yeah", "start": 0.7, "end": 1.0},
        ],
        "silences": [],
    }
    _seed_project(projects_root, "proj-clean", transcript)
    edl_obj = {"edl_id": "c", "segments": [{"from_word": 0, "to_word": 2, "label": "s"}],
               "cleanup": {"remove_fillers": True}}
    res = ops.validate_edl("proj-clean", edl_obj)
    assert "cleanup" in res
    dropped = {d["w"] for d in res["cleanup"]["removed_fillers"]}
    assert "um" in dropped


def test_validate_edl_no_transcript_errors(projects_root):
    project.ensure_project("empty")
    project.write_meta("empty", {"project_id": "empty", "source": "x", "duration": 1.0,
                                 "fps": 30.0, "width": 100, "height": 100, "has_audio": True})
    res = ops.validate_edl("empty", {"edl_id": "x", "segments": [{"from_word": 0, "to_word": 0}]})
    assert "error" in res


def test_validate_edl_out_of_range_returns_error(projects_root, synthetic_transcript):
    _seed_project(projects_root, "p2", synthetic_transcript)
    res = ops.validate_edl("p2", {"edl_id": "bad", "segments": [{"from_word": 0, "to_word": 99}]})
    assert "error" in res


# --- list/get project -----------------------------------------------------------------

def test_list_and_get_project(projects_root, synthetic_transcript):
    _seed_project(projects_root, "alpha", synthetic_transcript)
    _seed_project(projects_root, "beta", synthetic_transcript)
    lst = ops.list_projects()
    assert lst["count"] == 2
    ids = {p["project_id"] for p in lst["projects"]}
    assert ids == {"alpha", "beta"}
    assert all(p["transcribed"] for p in lst["projects"])

    got = ops.get_project("alpha")
    assert got["project_id"] == "alpha"
    assert got["transcribed"] is True
    assert got["edls"] == []


def test_get_project_unknown(projects_root):
    assert "error" in ops.get_project("nope")


# --- get_transcript slicing -----------------------------------------------------------

def test_get_transcript_word_range(projects_root, synthetic_transcript):
    _seed_project(projects_root, "slice", synthetic_transcript)
    res = ops.get_transcript("slice", from_word=1, to_word=3)
    assert res["text"] == "world this is"
    assert res["first_word"] == 1
    assert res["last_word"] == 3
    assert res["count"] == 3


def test_get_transcript_time_range_and_json(projects_root, synthetic_transcript):
    _seed_project(projects_root, "slice2", synthetic_transcript)
    # word 0 spans 0.5-1.0, word 1 spans 1.5-2.0 ...
    res = ops.get_transcript("slice2", from_time=1.4, to_time=2.1, fmt="json")
    assert "words" in res
    assert [w["w"] for w in res["words"]] == ["world"]


def test_get_transcript_no_transcript(projects_root):
    project.ensure_project("nt")
    project.write_meta("nt", {"project_id": "nt", "source": "x", "duration": 1.0,
                              "fps": 30.0, "width": 1, "height": 1, "has_audio": True})
    assert "error" in ops.get_transcript("nt")


# --- catalogs -------------------------------------------------------------------------

def test_list_caption_presets():
    res = ops.list_caption_presets()
    assert res["default"] == "karaoke-bold"
    assert set(res["presets"]) == {"karaoke-bold", "lower-third", "minimal-top"}


def test_list_reframe_modes():
    res = ops.list_reframe_modes()
    assert "track" in res["modes"]
    assert "full" in res["modes"]
    assert "9:16" in res["aspects"]
    assert "source" in res["aspects"]


# --- suggest_clips (design brief only) ------------------------------------------------

def test_suggest_clips_is_design_brief(projects_root, synthetic_transcript):
    _seed_project(projects_root, "sg", synthetic_transcript)
    res = ops.suggest_clips("sg", n=2, instructions="punchy")
    assert res["mode"] == "design_brief"
    assert res["n"] == 2
    assert res["instructions"] == "punchy"
    assert "rubric" in res and "edl_schema" in res
    assert "[#0]" in res["indexed_transcript"]
    # NO llm/clips/score keys — this phase never calls a provider
    assert "clips" not in res
    assert "model" not in res


def test_suggest_clips_no_transcript(projects_root):
    project.ensure_project("sg2")
    project.write_meta("sg2", {"project_id": "sg2", "source": "x", "duration": 1.0,
                               "fps": 30.0, "width": 1, "height": 1, "has_audio": True})
    assert "error" in ops.suggest_clips("sg2")


# --- import hygiene -------------------------------------------------------------------

def test_imports_have_no_heavy_deps():
    """Importing the editing modules must not pull torch/cv2/ultralytics. Checked in a
    FRESH interpreter so the result is independent of what other tests (or a full-install
    venv) have already loaded into this session's sys.modules."""
    import subprocess
    code = (
        "import importlib, sys\n"
        "for m in ('dopest_clip.ops','dopest_clip.reframe','dopest_clip.captions','dopest_clip.verify'):\n"
        "    importlib.import_module(m)\n"
        "bad=[h for h in ('torch','cv2','ultralytics') if h in sys.modules]\n"
        "assert not bad, bad\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"heavy deps imported at load: {r.stdout}{r.stderr}"


# --- pure reframe math (no model, no cv2) ---------------------------------------------

def test_one_euro_filter_passes_first_and_smooths():
    from dopest_clip.reframe import _OneEuro
    f = _OneEuro(freq=30.0, mincutoff=0.22, beta=0.018)
    # first sample passes through unchanged
    assert f(100.0) == 100.0
    # a jump is damped: output lands between previous and the new raw value
    out = f(200.0)
    assert 100.0 < out < 200.0


def test_base_crop_portrait_target_from_landscape():
    from dopest_clip.reframe import _base_crop
    # 9:16 portrait crop out of a 1920x1080 landscape -> full height, narrow width
    cw, ch = _base_crop(1920, 1080, 1080, 1920)
    assert ch == 1080
    assert cw == round(1080 * (1080 / 1920))  # height * target_ar


def test_shot_rect_full_is_whole_frame():
    from dopest_clip.reframe import _shot_rect
    rect = _shot_rect({"mode": "full"}, 0, [], 1920, 1080, 608, 1080)
    assert rect == [0.0, 0.0, 1920.0, 1080.0]


def test_shot_rect_focus_clamps_inside_frame():
    from dopest_clip.reframe import _shot_rect
    # focus near the left edge with zoom -> clamped so x >= 0 and width fits
    rect = _shot_rect({"mode": "focus", "x": 0, "y": 540, "zoom": 1.0}, 0, [],
                      1920, 1080, 608, 1080)
    x, y, w, h = rect
    assert x >= 0
    assert x + w <= 1920
    assert h <= 1080


def test_shot_rect_explicit_crop_wins():
    from dopest_clip.reframe import _shot_rect
    rect = _shot_rect({"crop": {"x": 100, "y": 50, "w": 400, "h": 300}}, 0, [],
                      1920, 1080, 608, 1080)
    assert rect == [100, 50, 400, 300]


def test_resolve_model_path_uses_bundled_default():
    """The bare default 'yolo11n.pt' resolves to the bundled asset (no ultralytics needed)."""
    from dopest_clip import config, reframe
    resolved = reframe._resolve_model_path()
    assert resolved == str(config.ASSETS_DIR / "yolo11n.pt")
    assert (config.ASSETS_DIR / "yolo11n.pt").exists()


@pytest.mark.needs_gpu
@pytest.mark.skipif(not _HAS_ULTRALYTICS, reason="ultralytics not installed")
def test_reframe_model_loads():
    """Requires CUDA + ultralytics; smoke-test the bundled model path resolves."""
    from dopest_clip import reframe
    reframe._load_model()
