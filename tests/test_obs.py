"""OBS recording subsystem — pure-logic tests.

Runs in the LIGHT venv: pytest, pytest-mock, Pillow, requests, mcp. No
websocket-client / moviepy / torch / cv2 / resvg, and no running OBS. So we test:

- timeline easing + keyframe interpolation + clamping (pure math),
- graphics SVG string builders (no rasterization),
- the ws.py requestId correlation with a FAKE socket (monkeypatched),
- that every obs.* module imports with none of the heavy deps installed.

Anything needing a live OBS is marked needs_obs and skipped; matting/compose
needs_gpu and skipped.
"""

import json

import pytest

from dopest_clip.obs import graphics, timeline


# --------------------------------------------------------------------------- #
# import-without-heavy-deps                                                    #
# --------------------------------------------------------------------------- #

def test_all_obs_modules_import_without_heavy_deps():
    # None of these imports may require websocket/moviepy/torch/cv2/resvg.
    import dopest_clip.obs.blur  # noqa: F401
    import dopest_clip.obs.camera_mix  # noqa: F401
    import dopest_clip.obs.client  # noqa: F401
    import dopest_clip.obs.compositor  # noqa: F401
    import dopest_clip.obs.graphics  # noqa: F401
    import dopest_clip.obs.timeline  # noqa: F401
    import dopest_clip.obs.ws  # noqa: F401


# --------------------------------------------------------------------------- #
# timeline: easing                                                             #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("kind", ["linear", "in", "out", "inout", "unknown-defaults-to-inout"])
def test_easing_bounded_and_endpoints(kind):
    # f(0) == 0, f(1) == 1, and stays within [0,1] across the interval.
    assert timeline._ease(0.0, kind) == pytest.approx(0.0)
    assert timeline._ease(1.0, kind) == pytest.approx(1.0)
    for i in range(21):
        p = i / 20.0
        v = timeline._ease(p, kind)
        assert 0.0 <= v <= 1.0


@pytest.mark.parametrize("kind", ["linear", "in", "out", "inout"])
def test_easing_monotonic_nondecreasing(kind):
    prev = -1.0
    for i in range(101):
        v = timeline._ease(i / 100.0, kind)
        assert v >= prev - 1e-9, f"{kind} not monotonic at {i}"
        prev = v


def test_easing_clamps_out_of_range():
    assert timeline._ease(-5.0, "linear") == pytest.approx(0.0)
    assert timeline._ease(5.0, "linear") == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# timeline: keyframe interpolation + clamping                                  #
# --------------------------------------------------------------------------- #

def test_normalize_keyframes_expands_presets_and_sorts():
    kfs = timeline.normalize_keyframes([
        {"t": 2.0, "preset": "fullscreen"},
        {"t": 0.0, "preset": "bottom-right"},
    ])
    assert [k["t"] for k in kfs] == [0.0, 2.0]          # sorted by time
    assert kfs[0]["anchor"] == "bottom-right"
    assert kfs[0]["scale"] == pytest.approx(timeline.PIP_SCALE)
    assert kfs[1]["anchor"] == "center"
    assert kfs[1]["scale"] == pytest.approx(1.0)


def test_normalize_keyframes_validation():
    with pytest.raises(ValueError):
        timeline.normalize_keyframes([])
    with pytest.raises(ValueError):
        timeline.normalize_keyframes([{"scale": 0.5}])             # missing t
    with pytest.raises(ValueError):
        timeline.normalize_keyframes([{"t": 0, "preset": "nope"}])  # unknown preset
    with pytest.raises(ValueError):
        timeline.normalize_keyframes([{"t": 0, "anchor": "middle"}])  # bad anchor


def test_sample_interpolates_pos_linear_midpoint():
    # two pos keyframes, same scale, linear ease -> centre moves linearly.
    fw, fh = 1000, 1000
    kfs = timeline.normalize_keyframes([
        {"t": 0.0, "pos": [0.0, 0.0], "scale": 0.5, "ease": "linear"},
        {"t": 1.0, "pos": [1.0, 1.0], "scale": 0.5, "ease": "linear"},
    ])
    aspect = 1.0
    # camera box is 0.5*fh tall = 500 px; aspect 1 -> 500 wide.
    x0, y0, w0, h0 = timeline.sample(kfs, 0.0, fw, fh, aspect)
    x1, y1, w1, h1 = timeline.sample(kfs, 1.0, fw, fh, aspect)
    xm, ym, wm, hm = timeline.sample(kfs, 0.5, fw, fh, aspect)
    assert (w0, h0) == (500, 500) == (w1, h1) == (wm, hm)
    # centre at t=0 is (0,0) -> top-left of box at (-250,-250); at t=1 centre (1000,1000).
    # midpoint centre = (500,500) -> top-left (250,250).
    assert (x0, y0) == (-250, -250)
    assert (x1, y1) == (750, 750)
    assert (xm, ym) == (250, 250)


def test_sample_clamps_before_first_and_after_last():
    fw, fh = 800, 600
    kfs = timeline.normalize_keyframes([
        {"t": 1.0, "pos": [0.2, 0.2], "scale": 0.4, "ease": "linear"},
        {"t": 3.0, "pos": [0.8, 0.8], "scale": 0.4, "ease": "linear"},
    ])
    before = timeline.sample(kfs, 0.0, fw, fh, 1.0)
    at_first = timeline.sample(kfs, 1.0, fw, fh, 1.0)
    after = timeline.sample(kfs, 99.0, fw, fh, 1.0)
    at_last = timeline.sample(kfs, 3.0, fw, fh, 1.0)
    assert before == at_first      # clamp to first keyframe
    assert after == at_last        # clamp to last keyframe


def test_sample_easing_inout_midpoint_equals_geometry_midpoint():
    # smoothstep at p=0.5 is exactly 0.5, so the inout midpoint matches linear.
    fw = fh = 1000
    kfs = timeline.normalize_keyframes([
        {"t": 0.0, "pos": [0.0, 0.5], "scale": 0.5, "ease": "inout"},
        {"t": 1.0, "pos": [1.0, 0.5], "scale": 0.5, "ease": "inout"},
    ])
    xm = timeline.sample(kfs, 0.5, fw, fh, 1.0)[0]
    # centre x at mid = 500 -> left edge = 500 - 250 = 250
    assert xm == 250


def test_sample_screen_full_when_empty():
    assert timeline.sample_screen([], 0.0, 1920, 1080) == (0, 0, 1920, 1080)


def test_sample_screen_zoom_crops_centered():
    kfs = timeline.normalize_screen_keyframes([
        {"t": 0.0, "zoom": 2.0, "focus": [0.5, 0.5]},
    ])
    x, y, w, h = timeline.sample_screen(kfs, 0.0, 1000, 1000)
    assert (w, h) == (500, 500)
    assert (x, y) == (250, 250)   # centred crop


def test_sample_overlay_anchor_places_tip_on_pos():
    # an overlay whose anchor is its tip [0.5, 0.0] should land that point on pos.
    kfs = timeline.normalize_overlay_keyframes([
        {"t": 0.0, "pos": [0.5, 0.5], "scale": 0.2},
    ])
    fw = fh = 1000
    aspect = 1.0
    anchor = [0.5, 0.0]
    x, y, w, h = timeline.sample_overlay(kfs, 0.0, fw, fh, aspect, anchor)
    # draw_w = 0.2*1000 = 200, draw_h = 200; anchor x at pos 500 -> x = 500 - 0.5*200 = 400
    # anchor y at pos 500 -> y = 500 - 0.0*200 = 500
    assert (w, h) == (200, 200)
    assert (x, y) == (400, 500)


# --------------------------------------------------------------------------- #
# graphics: SVG string builders (no rasterization)                             #
# --------------------------------------------------------------------------- #

def test_arrow_svg_and_tip_anchor():
    svg, anchor = graphics.arrow(direction="right", color="#abcdef", stroke=10)
    assert svg.lstrip().startswith("<svg")
    assert "<polygon" in svg and "<line" in svg
    assert "#abcdef" in svg
    assert 'stroke-width="10"' in svg
    # right (0 deg): tip is at +45 in x from centre 50 -> 95/100 = 0.95, y = 0.5
    assert anchor[0] == pytest.approx(0.95, abs=1e-3)
    assert anchor[1] == pytest.approx(0.5, abs=1e-3)


def test_arrow_accepts_explicit_angle_string():
    svg, anchor = graphics.arrow(direction="90")  # straight down
    assert "<svg" in svg
    assert anchor[1] == pytest.approx(0.95, abs=1e-3)  # tip below centre


def test_ring_svg_centered_anchor():
    svg, anchor = graphics.ring(color="#00ff00", stroke=8)
    assert "<circle" in svg
    assert "#00ff00" in svg
    assert 'fill="none"' in svg
    assert anchor == [0.5, 0.5]


def test_box_svg_aspect_and_anchor():
    svg, anchor = graphics.box(color="#123456", aspect=2.0)
    assert "<rect" in svg
    assert "#123456" in svg
    assert "viewBox=\"0 0 100.0 50.0\"" in svg  # 100 wide / aspect 2 = 50 tall
    assert anchor == [0.5, 0.5]


def test_label_svg_contains_escaped_text():
    svg, anchor = graphics.label("A & B < C", color="#ff0000", text_color="#ffffff")
    assert "<text" in svg and "<rect" in svg
    assert "A &amp; B &lt; C" in svg     # xml-escaped
    assert "#ff0000" in svg and "#ffffff" in svg
    assert anchor == [0.5, 0.5]


def test_build_dispatch_and_unknown_kind():
    svg, anchor = graphics.build({"kind": "ring", "color": "#fff"})
    assert "<circle" in svg
    with pytest.raises(ValueError):
        graphics.build({"kind": "spiral"})
    # inline svg passthrough
    svg2, anchor2 = graphics.build({"svg": "<svg>x</svg>", "anchor": [0.1, 0.2]})
    assert svg2 == "<svg>x</svg>"
    assert anchor2 == [0.1, 0.2]


def test_build_strips_reserved_keys_before_calling_builder():
    # keyframes/t_in/etc must not be passed to the shape builder (would TypeError).
    svg, _ = graphics.build({
        "kind": "arrow", "direction": "up", "keyframes": [{"t": 0, "pos": [0, 0]}],
        "t_in": 1.0, "fade": 0.2, "opacity": 0.5,
    })
    assert "<svg" in svg


# --------------------------------------------------------------------------- #
# ws.py: requestId correlation with a FAKE socket (no real connection)         #
# --------------------------------------------------------------------------- #

class FakeSocket:
    """A scripted obs-websocket peer. ``feed`` queues frames the client will recv;
    ``sent`` records what the client sent. No network."""

    def __init__(self, frames):
        self._frames = list(frames)
        self.sent = []
        self.closed = False

    def recv(self):
        if not self._frames:
            raise AssertionError("client recv'd more frames than were queued")
        return self._frames.pop(0)

    def send(self, data):
        self.sent.append(json.loads(data))

    def close(self):
        self.closed = True


def _resp(request_id, request_type, data=None, result=True, code=100, comment=""):
    return json.dumps({
        "op": 7,
        "d": {
            "requestType": request_type,
            "requestId": request_id,
            "requestStatus": {"result": result, "code": code, "comment": comment},
            "responseData": data or {},
        },
    })


def _event(event_type="SomeEvent"):
    return json.dumps({"op": 5, "d": {"eventType": event_type, "eventData": {}}})


def _make_client_with_socket(fake):
    from dopest_clip.obs.ws import WSClient
    c = WSClient(host="x", port=1)
    c._ws = fake  # inject the fake, bypassing connect()/websocket import
    return c


def test_request_matches_response_with_same_request_id():
    # client will send r1; respond to r1.
    fake = FakeSocket([_resp("r1", "GetVersion", {"obsVersion": "32.1.2"})])
    c = _make_client_with_socket(fake)
    out = c.request("GetVersion")
    assert out == {"obsVersion": "32.1.2"}
    assert fake.sent[0]["d"]["requestId"] == "r1"
    assert fake.sent[0]["d"]["requestType"] == "GetVersion"


def test_request_skips_events_and_mismatched_ids_then_resolves_correct_one():
    # An event, then a response to a DIFFERENT request id, then the real r1 reply.
    # A naive blind-recv client would return the wrong frame; this one must skip both.
    fake = FakeSocket([
        _event("RecordStateChanged"),
        _resp("r999", "SomethingElse", {"wrong": True}),
        _resp("r1", "GetRecordStatus", {"outputActive": True}),
    ])
    c = _make_client_with_socket(fake)
    out = c.request("GetRecordStatus")
    assert out == {"outputActive": True}      # matched r1, not r999 / not the event


def test_request_raises_on_failed_status():
    from dopest_clip.obs.ws import OBSError
    fake = FakeSocket([_resp("r1", "RemoveInput", result=False, code=600, comment="no such input")])
    c = _make_client_with_socket(fake)
    with pytest.raises(OBSError) as ei:
        c.request("RemoveInput", {"inputName": "ghost"})
    assert "600" in str(ei.value)
    assert "no such input" in str(ei.value)


def test_request_id_increments_across_calls():
    fake = FakeSocket([
        _resp("r1", "A", {"n": 1}),
        _resp("r2", "B", {"n": 2}),
    ])
    c = _make_client_with_socket(fake)
    assert c.request("A") == {"n": 1}
    assert c.request("B") == {"n": 2}
    assert [s["d"]["requestId"] for s in fake.sent] == ["r1", "r2"]


def test_mismatched_id_does_not_resolve_wrong_waiter_eventually_raises():
    # Only a wrong-id frame is queued; the client must NOT accept it. It will then
    # recv past the end -> our FakeSocket raises AssertionError (proving it kept reading).
    fake = FakeSocket([_resp("r999", "Wrong", {"bad": True})])
    c = _make_client_with_socket(fake)
    with pytest.raises(AssertionError):
        c.request("GetVersion")


# --------------------------------------------------------------------------- #
# ws.py: auto-reconnect after a dead socket (OBS restart -> WinError 10053)     #
# --------------------------------------------------------------------------- #

def test_request_reconnects_after_dropped_socket(monkeypatch):
    """After OBS restarts, the cached socket is dead: send raises ConnectionAbortedError
    (WinError 10053, an OSError). request() must drop it, reconnect, and retry ONCE."""
    from dopest_clip.obs.ws import WSClient

    class DeadSocket:
        def send(self, data):
            raise ConnectionAbortedError(10053, "An established connection was aborted")
        def recv(self):
            raise AssertionError("must not recv on the dead socket")
        def close(self):
            pass

    good = FakeSocket([_resp("r2", "GetVersion", {"obsVersion": "32"})])  # retry uses rid r2
    c = WSClient(host="x", port=1)
    c._ws = DeadSocket()  # the stale socket left over from before the OBS restart
    monkeypatch.setattr(c, "connect", lambda: setattr(c, "_ws", good))  # reconnect -> good socket
    out = c.request("GetVersion")
    assert out == {"obsVersion": "32"}
    assert good.sent and good.sent[0]["d"]["requestType"] == "GetVersion"


def test_request_does_not_reconnect_on_logical_failure(monkeypatch):
    """A result:false response is an OBSError, not a transport drop -> NO reconnect/retry."""
    from dopest_clip.obs.ws import OBSError, WSClient
    fake = FakeSocket([_resp("r1", "RemoveInput", result=False, code=600, comment="nope")])
    c = WSClient(host="x", port=1)
    c._ws = fake
    reconnects: list[int] = []
    monkeypatch.setattr(c, "connect", lambda: reconnects.append(1))
    with pytest.raises(OBSError):
        c.request("RemoveInput", {"inputName": "ghost"})
    assert len(fake.sent) == 1   # sent exactly once
    assert reconnects == []       # never tried to reconnect


# --------------------------------------------------------------------------- #
# client.py — the three bugs fixed after live QA against real OBS hardware:     #
#   1. device-id resolution (camera bound by name -> never opened, 0x0)         #
#   2. camera-streaming verification (no more false "scene ready")              #
#   3. stop_recording polls for the finalized camera file (was a 1.5s race)     #
# Plus the OBS reliability fixes: best-effort probe teardown + orphan sweep.     #
# --------------------------------------------------------------------------- #

def _bare_client():
    from dopest_clip.obs import client as obc
    return obc.OBSClient.__new__(obc.OBSClient)  # bypass __init__ (no WSClient/connect)


def test_resolve_device_accepts_id_name_and_substring():
    from dopest_clip.obs import client as obc
    c = _bare_client()
    full = r"NexiGo N60 FHD Webcam:\\?\usb#vid_3443"
    c.list_devices = lambda kind: [
        obc.Device(name="NexiGo N60 FHD Webcam", device_id=full),
        obc.Device(name="OBS Virtual Camera", device_id="OBS Virtual Camera:"),
    ]
    assert c._resolve_device(obc.KIND_CAMERA, full) == full                    # exact device_id
    assert c._resolve_device(obc.KIND_CAMERA, "NexiGo N60 FHD Webcam") == full  # exact name -> id
    assert c._resolve_device(obc.KIND_CAMERA, "nexigo") == full                 # ci substring


def test_resolve_device_raises_on_unknown_and_empty():
    from dopest_clip.obs import client as obc
    c = _bare_client()
    c.list_devices = lambda kind: [obc.Device(name="Cam A", device_id="Cam A:id")]
    with pytest.raises(obc.OBSError):
        c._resolve_device(obc.KIND_CAMERA, "does-not-exist")
    with pytest.raises(obc.OBSError):
        c._resolve_device(obc.KIND_CAMERA, "")


def test_camera_dims_and_wait_streaming_detects_frames():
    c = _bare_client()
    seq = iter([
        {"sceneItems": [{"sourceName": "Camera", "sceneItemTransform": {"sourceWidth": 0, "sourceHeight": 0}}]},
        {"sceneItems": [{"sourceName": "Camera", "sceneItemTransform": {"sourceWidth": 1920, "sourceHeight": 1080}}]},
    ])
    c.req = lambda t, d=None: next(seq)
    assert c.camera_dims("S") == (0.0, 0.0)            # consumes the 0x0 frame
    assert c.wait_camera_streaming("S", timeout=5.0) is True  # next poll sees 1920x1080


def test_wait_camera_streaming_times_out_when_zero():
    c = _bare_client()
    c.req = lambda t, d=None: {"sceneItems": [
        {"sourceName": "Camera", "sceneItemTransform": {"sourceWidth": 0, "sourceHeight": 0}}]}
    assert c.wait_camera_streaming("S", timeout=1.0) is False


def test_stop_recording_polls_for_finalized_camera_file(tmp_path):
    c = _bare_client()
    c._camera_dir = str(tmp_path)
    c._cam_before = set()
    c.req = lambda t, d=None: {"outputPath": "C:/screen.mp4"}  # StopRecord
    cam = tmp_path / "camera_2026-06-17_21-53-34.mkv"
    cam.write_bytes(b"x" * 256)  # present + stable
    out = c.stop_recording(timeout=3.0)
    assert out["screen"] == "C:/screen.mp4"
    assert out["camera"] == str(cam)


def test_stop_recording_returns_null_camera_when_none_written(tmp_path):
    c = _bare_client()
    c._camera_dir = str(tmp_path)
    c._cam_before = set()
    c.req = lambda t, d=None: {"outputPath": "C:/screen.mp4"}
    out = c.stop_recording(timeout=1.0)  # no camera_* file appears
    assert out["camera"] is None


def test_ensure_absent_does_not_raise_on_slow_teardown():
    # An input that never disappears (slow display-capture teardown) must NOT raise --
    # this was the "did not tear down within 5.0s" hang that blocked setup_scene.
    c = _bare_client()
    c.req = lambda t, d=None: ({"inputs": [{"inputName": "stuck"}]} if t == "GetInputList" else {})
    assert c._ensure_absent("stuck", timeout=0.3) is None


def test_sweep_probes_removes_only_probe_inputs():
    c = _bare_client()
    removed: list[str] = []
    inputs = [{"inputName": "Screen"},
              {"inputName": "__dopestclip_probe_monitor_capture_3"},
              {"inputName": "Camera"}]

    def req(t, d=None):
        if t == "GetInputList":
            return {"inputs": inputs}
        if t == "RemoveInput":
            removed.append(d["inputName"])
            return {}
        return {}
    c.req = req
    c._sweep_probes()
    assert removed == ["__dopestclip_probe_monitor_capture_3"]


def test_ensure_input_recreates_when_input_missing_from_scene():
    # Input exists globally (right device) but is NOT in this scene (GetSceneItemId 600,
    # e.g. it lingered from a renamed old scene). Must remove + recreate, not crash on 600.
    from dopest_clip.obs.ws import OBSError
    c = _bare_client()
    present = {"Screen": True}
    calls: list[str] = []

    def req(t, d=None):
        calls.append(t)
        if t == "GetInputList":
            return {"inputs": [{"inputName": "Screen"}] if present["Screen"] else []}
        if t == "GetInputSettings":
            return {"inputSettings": {"monitor_id": "MON"}}
        if t == "GetSceneItemId":
            raise OBSError("GetSceneItemId failed [600]: no scene items")
        if t == "RemoveInput":
            present["Screen"] = False
            return {}
        if t == "CreateInput":
            return {"sceneItemId": 7}
        return {}
    c.req = req
    sid, created = c._ensure_input(
        "DopestClipRec", "Screen", "monitor_capture", "monitor_id", "MON", {"monitor_id": "MON"})
    assert (sid, created) == (7, True)
    assert "CreateInput" in calls   # recreated into this scene instead of 600-crashing


def test_ensure_input_reuses_when_in_scene_with_matching_device():
    # Already in this scene with the right device -> reuse, never recreate (no device reopen).
    c = _bare_client()
    calls: list[str] = []

    def req(t, d=None):
        calls.append(t)
        if t == "GetInputList":
            return {"inputs": [{"inputName": "Screen"}]}
        if t == "GetInputSettings":
            return {"inputSettings": {"monitor_id": "MON"}}
        if t == "GetSceneItemId":
            return {"sceneItemId": 3}
        return {}
    c.req = req
    sid, created = c._ensure_input(
        "DopestClipRec", "Screen", "monitor_capture", "monitor_id", "MON", {"monitor_id": "MON"})
    assert (sid, created) == (3, False)
    assert "CreateInput" not in calls and "RemoveInput" not in calls


# --------------------------------------------------------------------------- #
# live-OBS / GPU markers (skipped here)                                        #
# --------------------------------------------------------------------------- #

@pytest.mark.needs_obs
def test_live_setup_scene_and_record():  # pragma: no cover - needs a running OBS
    pytest.skip("requires a running OBS with obs-websocket + Source Record plugin")


@pytest.mark.needs_gpu
def test_gpu_camera_mix():  # pragma: no cover - needs CUDA + matting extra
    pytest.skip("requires CUDA + dopest-clip[matting]")
