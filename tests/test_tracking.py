"""Coverage for obs.tracking — the per-frame target tracker that lets any compose effect
FOLLOW a moving target instead of sitting at static keyframes.

The PURE logic (sample_track interpolation, the rect re-centre, normalization+smoothing,
target validation, the track/static split) is tested with no heavy deps. The actual
detectors + the full compute_track/preview loop run live against a synthetic video and are
guarded by importorskip('cv2') so the light venv still collects this file.
"""

import json

import pytest

from dopest_clip.obs import tracking


# --- sample_track: interpolation by time ----------------------------------------------

def _track():
    return [{"t": 0.0, "x": 0.0, "y": 0.0, "w": 0.1, "h": 0.2},
            {"t": 1.0, "x": 1.0, "y": 0.5, "w": 0.3, "h": 0.4}]


def test_sample_track_empty_is_none():
    assert tracking.sample_track([], 1.0) is None


def test_sample_track_clamps_before_start_and_after_end():
    tr = _track()
    assert tracking.sample_track(tr, -5.0) == (0.0, 0.0, 0.1, 0.2)
    assert tracking.sample_track(tr, 99.0) == (1.0, 0.5, 0.3, 0.4)


def test_sample_track_interpolates_midpoint():
    x, y, w, h = tracking.sample_track(_track(), 0.5)
    assert x == pytest.approx(0.5) and y == pytest.approx(0.25)
    assert w == pytest.approx(0.2) and h == pytest.approx(0.3)


def test_sample_track_exact_first_time_hits_start_clamp():
    tr = [{"t": 1.0, "x": 0.2, "y": 0.2, "w": 0.1, "h": 0.1},
          {"t": 2.0, "x": 0.8, "y": 0.8, "w": 0.1, "h": 0.1}]
    # t == first time is <= track[0]['t'], so the start clamp returns the first point
    assert tracking.sample_track(tr, 1.0) == (0.2, 0.2, 0.1, 0.1)


def test_sample_track_single_point():
    tr = [{"t": 2.0, "x": 0.4, "y": 0.6, "w": 0.1, "h": 0.1}]
    assert tracking.sample_track(tr, 0.0) == (0.4, 0.6, 0.1, 0.1)
    assert tracking.sample_track(tr, 5.0) == (0.4, 0.6, 0.1, 0.1)


# --- apply_track_to_rect: re-centre keeping size --------------------------------------

def test_apply_track_to_rect_empty_track_returns_rect_unchanged():
    rect = (10, 20, 30, 40)
    assert tracking.apply_track_to_rect(rect, [], 0.0, 100, 100) == rect


def test_apply_track_to_rect_centres_on_tracked_point():
    tr = [{"t": 0.0, "x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1}]
    # box 40x20 centred (anchor 0.5,0.5) on (0.5,0.5) of a 200x100 frame -> centre (100,50)
    x, y, w, h = tracking.apply_track_to_rect((0, 0, 40, 20), tr, 0.0, 200, 100)
    assert (x, y, w, h) == (80, 40, 40, 20)


def test_apply_track_to_rect_custom_anchor():
    tr = [{"t": 0.0, "x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1}]
    # anchor (0,0) = top-left lands on the tracked point
    x, y, w, h = tracking.apply_track_to_rect((0, 0, 40, 20), tr, 0.0, 200, 100, anchor=(0.0, 0.0))
    assert (x, y) == (100, 50)


def test_apply_track_to_rect_clamps_inside_frame():
    tr = [{"t": 0.0, "x": 1.0, "y": 1.0, "w": 0.1, "h": 0.1}]   # bottom-right corner
    x, y, w, h = tracking.apply_track_to_rect((0, 0, 40, 20), tr, 0.0, 200, 100, clamp=True)
    assert x == 200 - 40 and y == 100 - 20            # kept fully inside


def test_apply_track_to_rect_src_rect_maps_camera_coords_to_output():
    # face at camera-centre (0.5,0.5); camera composited as a 100x100 PIP at output (300,200)
    tr = [{"t": 0.0, "x": 0.5, "y": 0.5, "w": 0.2, "h": 0.4}]
    x, y, w, h = tracking.apply_track_to_rect((0, 0, 20, 20), tr, 0.0, 1000, 1000,
                                              src_rect=(300, 200, 100, 100))
    # camera-centre maps to output (300+50, 200+50)=(350,250); 20x20 box centred there
    assert (x, y) == (340, 240)


def test_apply_track_to_rect_offset_places_above_head():
    # head centre at (0.5,0.5), head box 0.0 x 0.2 of a 100x500 frame -> head height 100px
    tr = [{"t": 0.0, "x": 0.5, "y": 0.5, "w": 0.0, "h": 0.2}]
    # offset oy=-0.9 -> shift up 0.9*100 = 90px from centre (250) -> 160; anchor bottom (0,1)
    x, y, w, h = tracking.apply_track_to_rect((0, 0, 30, 40), tr, 0.0, 100, 500,
                                              anchor=(0.5, 1.0), offset=(0.0, -0.9))
    assert y == 160 - 40            # bottom of the box sits 90px above the head centre


def test_apply_track_to_rect_offset_with_src_rect_uses_camera_box_size():
    # face box 0.5 high in a 200px-tall camera PIP -> head height 100px in output
    tr = [{"t": 0.0, "x": 0.5, "y": 0.5, "w": 0.4, "h": 0.5}]
    x, y, w, h = tracking.apply_track_to_rect((0, 0, 10, 10), tr, 0.0, 1000, 1000,
                                              src_rect=(0, 0, 200, 200), offset=(0.0, -1.0))
    # centre = (100,100); offset up 1.0*(0.5*200)=100 -> y-centre 0; 10x10 centred -> y=-5
    assert y == -5 and x == 95


# --- _to_normalized: per-frame px -> smoothed normalized ------------------------------

def test_to_normalized_shapes_and_bounds():
    sw, sh, fps = 100, 50, 10.0
    per_frame = [(50.0, 25.0, 10.0, 8.0)] * 5
    out = tracking._to_normalized(per_frame, sw, sh, fps)
    assert len(out) == 5
    assert out[0]["t"] == 0.0 and out[1]["t"] == pytest.approx(0.1)
    for e in out:
        assert 0.0 <= e["x"] <= 1.0 and 0.0 <= e["y"] <= 1.0
        assert e["w"] == pytest.approx(0.1) and e["h"] == pytest.approx(0.16)


def test_to_normalized_negative_size_floored_to_zero():
    out = tracking._to_normalized([(10.0, 10.0, -5.0, -5.0)], 100, 100, 30.0)
    assert out[0]["w"] == 0.0 and out[0]["h"] == 0.0


# --- validate_target ------------------------------------------------------------------

@pytest.mark.parametrize("t", ["cursor", "face", "person", "cup", "laptop", "cell phone"])
def test_validate_target_accepts_known(t):
    tracking.validate_target(t)            # no raise


def test_validate_target_accepts_template_dict():
    tracking.validate_target({"template_at": 1.0, "region": [0, 0, 10, 10]})


def test_validate_target_rejects_unknown_string():
    with pytest.raises(ValueError, match="unknown track target"):
        tracking.validate_target("banana_split")


def test_validate_target_rejects_template_missing_keys():
    with pytest.raises(ValueError, match="template target needs"):
        tracking.validate_target({"region": [0, 0, 10, 10]})


def test_validate_target_rejects_wrong_type():
    with pytest.raises(ValueError, match="must be a string or a template dict"):
        tracking.validate_target(42)


# --- _target_key ----------------------------------------------------------------------

def test_target_key_string_and_template():
    assert tracking._target_key("cell phone") == "cell_phone"
    assert tracking._target_key({"template_at": 2.5, "region": [10, 20, 30, 40]}) == "tmpl_2.5_10_20_30_40"


# --- resolve_track --------------------------------------------------------------------

def test_resolve_track_none_spec_returns_none():
    assert tracking.resolve_track(None, "s.mp4", "c.mp4", "/cache") is None


def test_resolve_track_missing_source_video_returns_none():
    # source 'camera' but no camera path available
    assert tracking.resolve_track({"target": "face", "source": "camera"}, "s.mp4", None, "/cache") is None


def test_resolve_track_delegates_with_cache_path(monkeypatch):
    seen = {}

    def fake_compute(video, target, *, cache_path=None):
        seen.update(video=video, target=target, cache_path=cache_path)
        return [{"t": 0.0, "x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1}]

    monkeypatch.setattr(tracking, "compute_track", fake_compute)
    out = tracking.resolve_track({"target": "cursor"}, "scr.mp4", "cam.mp4", "/cache")
    assert out[0]["x"] == 0.5
    assert seen["video"] == "scr.mp4"           # default source is screen
    assert seen["cache_path"].endswith("track_screen_cursor.json")


def test_resolve_track_camera_source_uses_camera_video(monkeypatch):
    seen = {}
    monkeypatch.setattr(tracking, "compute_track",
                        lambda video, target, *, cache_path=None: seen.update(video=video) or [])
    tracking.resolve_track({"target": "face", "source": "camera"}, "scr.mp4", "cam.mp4", None)
    assert seen["video"] == "cam.mp4"


# --- compute_track cache hit (pure: returns before importing cv2) ---------------------

def test_compute_track_returns_cached_without_detecting(tmp_path):
    cache = tmp_path / "track.json"
    track = [{"t": 0.0, "x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}]
    cache.write_text(json.dumps({"target": "cursor", "track": track}), encoding="utf-8")
    # video path is bogus on purpose — a cache hit must not touch it
    out = tracking.compute_track("does-not-exist.mp4", "cursor", cache_path=str(cache))
    assert out == track


def test_compute_track_corrupt_cache_falls_through(tmp_path, monkeypatch):
    cache = tmp_path / "track.json"
    cache.write_text("{not json", encoding="utf-8")
    # past the bad cache it imports cv2 + opens the video; stub compute by faking cv2 absent
    import builtins
    real_import = builtins.__import__

    def no_cv2(name, *a, **k):
        if name == "cv2":
            raise ImportError("cv2 blocked for test")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_cv2)
    with pytest.raises(ImportError):
        tracking.compute_track("v.mp4", "cursor", cache_path=str(cache))


# --- _split_track (camera_mix glue, pure) ---------------------------------------------

def test_split_track_plain_list_has_no_track():
    from dopest_clip.obs import camera_mix
    kfs = [{"t": 0, "zoom": 2}]
    out_kfs, track = camera_mix._split_track(kfs)
    assert out_kfs == kfs and track is None


def test_split_track_lifts_track_off_keyframe_and_strips_it():
    from dopest_clip.obs import camera_mix
    kfs = [{"t": 0, "zoom": 2, "track": {"target": "cursor"}}, {"t": 3, "zoom": 1}]
    out_kfs, track = camera_mix._split_track(kfs)
    assert track == {"target": "cursor"}
    assert all("track" not in k for k in out_kfs)
    assert out_kfs[0]["zoom"] == 2          # other keys preserved


def test_split_track_wrapper_dict_form():
    from dopest_clip.obs import camera_mix
    out_kfs, track = camera_mix._split_track({"keyframes": [{"t": 0}], "track": {"target": "face"}})
    assert out_kfs == [{"t": 0}] and track == {"target": "face"}


# --- live: detectors + full loop against a synthetic video (needs cv2) ----------------

def _make_moving_square_video(path, n=30, w=160, h=120, sq=24):
    """A white square sliding left->right on black — a deterministic track target."""
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    assert vw.isOpened()
    for i in range(n):
        frame = np.zeros((h, w, 3), dtype="uint8")
        x = int(10 + (w - sq - 20) * i / (n - 1))
        y = h // 2 - sq // 2
        frame[y:y + sq, x:x + sq] = 255
        vw.write(frame)
    vw.release()


def test_compute_track_template_follows_moving_square(tmp_path):
    pytest.importorskip("cv2")
    vid = tmp_path / "sq.mp4"
    _make_moving_square_video(vid)
    target = {"template_at": 0.0, "region": [10, 48, 24, 24]}   # the square at frame 0
    track = tracking.compute_track(str(vid), target, every=1)
    assert len(track) >= 25
    # the square moves left->right, so the tracked x must increase end-vs-start
    assert track[-1]["x"] > track[0]["x"] + 0.2


def test_preview_draws_frame_and_downsamples(tmp_path):
    pytest.importorskip("cv2")
    vid = tmp_path / "sq.mp4"
    _make_moving_square_video(vid)
    out = tracking.preview(str(vid), {"template_at": 0.0, "region": [10, 48, 24, 24]},
                           at=0.5, max_points=5)
    assert out["points"] >= 25
    assert len(out["track"]) <= 5 + 1
    assert out["preview_frame"] and out["preview_frame"].endswith(".png")
    import os
    assert os.path.isfile(out["preview_frame"])
    assert out["at_point"] is not None


def test_preview_empty_track_returns_no_frame(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    monkeypatch.setattr(tracking, "compute_track", lambda *a, **k: [])
    out = tracking.preview(str(tmp_path / "x.mp4"), "cursor")
    assert out["points"] == 0 and out["preview_frame"] is None and out["track"] == []


def test_cursor_detector_missing_template_raises(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    from dopest_clip import config
    monkeypatch.setattr(config, "ASSETS_DIR", tmp_path)   # no cursor.png here
    with pytest.raises(FileNotFoundError, match="cursor template not found"):
        tracking._cursor_detector(100, 100)


def test_cursor_detector_finds_template(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    # build a tiny cursor template + a frame that contains it at a known spot
    assets = tmp_path / "assets"
    assets.mkdir()
    # a template with VARIANCE (left half white) — a uniform block makes matchTemplate degenerate
    tmpl = np.zeros((10, 10, 3), dtype="uint8")
    tmpl[:, :5] = 255
    cv2.imwrite(str(assets / "cursor.png"), tmpl)
    from dopest_clip import config
    monkeypatch.setattr(config, "ASSETS_DIR", assets)
    detect, every = tracking._cursor_detector(120, 90)
    assert every == 1
    frame = np.zeros((90, 120, 3), dtype="uint8")
    frame[40:50, 70:80, :] = 0
    frame[40:50, 70:75, :] = 255       # same left-half-white pattern at top-left (70,40)
    cx, cy, w, h = detect(frame)
    assert abs(cx - 75) <= 2 and abs(cy - 45) <= 2


def test_compute_track_writes_cache(tmp_path):
    pytest.importorskip("cv2")
    vid = tmp_path / "sq.mp4"
    _make_moving_square_video(vid)
    cache = tmp_path / "out" / "track.json"
    track = tracking.compute_track(str(vid), {"template_at": 0.0, "region": [10, 48, 24, 24]},
                                   every=1, cache_path=str(cache))
    import os
    assert os.path.isfile(cache)                       # cache written
    cached = json.loads(cache.read_text(encoding="utf-8"))
    assert cached["track"] == track


def test_template_detector_bad_seed_frame_raises(tmp_path):
    pytest.importorskip("cv2")
    with pytest.raises(ValueError, match="could not read the seed frame"):
        tracking._template_detector(str(tmp_path / "nope.mp4"), 0.0, [0, 0, 10, 10])


def test_face_detector_builds_and_misses_on_blank(monkeypatch):
    pytest.importorskip("cv2")
    import numpy as np
    detect, every = tracking._face_detector(120, 90)
    assert every == 2
    assert detect(np.zeros((90, 120, 3), dtype="uint8")) is None   # no face -> None


def test_build_detector_dispatches_each_target(monkeypatch):
    monkeypatch.setattr(tracking, "_cursor_detector", lambda sw, sh: ("CURSOR", 1))
    monkeypatch.setattr(tracking, "_face_detector", lambda sw, sh: ("FACE", 2))
    monkeypatch.setattr(tracking, "_yolo_detector", lambda name: (f"YOLO:{name}", 3))
    monkeypatch.setattr(tracking, "_template_detector", lambda v, t, r: ("TMPL", 2))
    assert tracking._build_detector("cursor", "v.mp4", 10, 10)[0] == "CURSOR"
    assert tracking._build_detector("face", "v.mp4", 10, 10)[0] == "FACE"
    assert tracking._build_detector("laptop", "v.mp4", 10, 10)[0] == "YOLO:laptop"
    assert tracking._build_detector({"template_at": 0, "region": [0, 0, 1, 1]}, "v.mp4", 10, 10)[0] == "TMPL"


def test_yolo_detector_builds_via_reframe(monkeypatch):
    # cover the YOLO detector wiring without loading a real model
    from dopest_clip import reframe
    monkeypatch.setattr(reframe, "_load_model", lambda: "MODEL")
    captured = {}

    def fake_box(model, frame, conf, classes, center_frac=0.5):
        captured.update(model=model, classes=classes, conf=conf, frac=center_frac)
        return (1.0, 2.0, 3.0, 4.0)

    monkeypatch.setattr(reframe, "_detect_box", fake_box)
    detect, every = tracking._yolo_detector("laptop")
    assert every == 3
    assert detect("frame") == (1.0, 2.0, 3.0, 4.0)
    assert captured["model"] == "MODEL" and captured["classes"] == [tracking.COCO["laptop"]]
