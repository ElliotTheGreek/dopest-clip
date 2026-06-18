"""OBS control for track-separated recording, on a reliable correlated websocket
client (see ws.py for why obsws-python was dropped).

Protocol facts grounded against a real machine (OBS 32.1.2 / websocket 5.7.3):
- Input kinds (Windows): monitor capture = ``monitor_capture``, camera =
  ``dshow_input``, mic = ``wasapi_input_capture``.
- Device lists are list-property items on a live input, read via
  ``GetInputPropertiesListPropertyItems``.
- Separate clean camera file = Source Record plugin, filter kind
  ``source_record_filter``.
- The camera source must be ENABLED (a hidden dshow source produces no frames) but
  moved OFF-CANVAS so it never lands in the screen recording; Source Record taps the
  source's native frames regardless of scene placement.

Connection + scene config is read from the environment (see ``_env_*`` below):
    OBS_WS_HOST       (default localhost)
    OBS_WS_PORT       (default 4455)
    OBS_WS_PASSWORD   (default unset)
    OBS_SCENE_NAME    (default "DopestClipRec")
    OBS_CAMERA_DIR    (default <PROJECTS_ROOT parent>/obs_camera)

Importing this module needs NO extra installed — the websocket-client dep is loaded
lazily by ws.py. The plain functions at the bottom (list_devices/setup_scene/
start_recording/stop_recording/recording_status) are the public surface used by the
MCP server and the Electron sidecar.
"""

from __future__ import annotations

import dataclasses
import glob
import os
import time
from typing import Any

from .. import config
from .ws import OBSError, WSClient

KIND_MONITOR = "monitor_capture"
KIND_CAMERA = "dshow_input"
KIND_MIC = "wasapi_input_capture"

DEVICE_PROP = {
    KIND_MONITOR: "monitor_id",
    KIND_CAMERA: "video_device_id",
    KIND_MIC: "device_id",
}

SOURCE_RECORD_FILTER_KIND = "source_record_filter"


@dataclasses.dataclass
class Device:
    name: str
    device_id: str


class OBSClient:
    # a freshly created Source Record filter (plus a just-recreated camera device)
    # needs a moment to wire its recording-start hook; starting the recording before
    # then silently misses it and no camera file is produced. 3s was flaky at the
    # threshold; 6s held across repeated runs.
    FILTER_SETTLE_S = 6.0

    def __init__(self, host: str = "localhost", port: int = 4455, password: str | None = None):
        self._ws = WSClient(host=host, port=port, password=password)
        self._record_ready_at = 0.0
        self._camera_dir: str | None = None
        self._cam_before: set[str] = set()

    def req(self, t: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._ws.request(t, data)

    def close(self) -> None:
        self._ws.close()

    # -- info ---------------------------------------------------------------
    def version(self) -> dict[str, Any]:
        v = self.req("GetVersion")
        return {
            "obs_version": v.get("obsVersion"),
            "websocket_version": v.get("obsWebSocketVersion"),
            "platform": v.get("platform"),
        }

    def has_source_record(self) -> bool:
        kinds = self.req("GetSourceFilterKindList").get("sourceFilterKinds", [])
        return SOURCE_RECORD_FILTER_KIND in kinds

    # -- device enumeration -------------------------------------------------
    def list_devices(self, kind: str) -> list[Device]:
        prop = DEVICE_PROP[kind]
        tmp = f"__dopestclip_probe_{kind}"
        self._ensure_absent(tmp)
        self.req("CreateInput", {
            "sceneName": self._first_scene(), "inputName": tmp,
            "inputKind": kind, "inputSettings": {}, "sceneItemEnabled": False,
        })
        try:
            items = self.req("GetInputPropertiesListPropertyItems",
                             {"inputName": tmp, "propertyName": prop}).get("propertyItems", [])
        finally:
            self._ensure_absent(tmp)
        out: list[Device] = []
        for it in items:
            val = it.get("itemValue")
            if val:
                out.append(Device(name=it.get("itemName") or val, device_id=val))
        return out

    def list_monitors(self) -> list[Device]:
        return self.list_devices(KIND_MONITOR)

    def list_cameras(self) -> list[Device]:
        return self.list_devices(KIND_CAMERA)

    def list_mics(self) -> list[Device]:
        return self.list_devices(KIND_MIC)

    def _resolve_device(self, kind: str, identifier: str) -> str:
        """Resolve a device *identifier* — either a full device_id OR a friendly name —
        to the exact device_id OBS expects for this input kind.

        This is the fix for the "camera never opens" class of bug: a dshow camera's
        ``video_device_id`` must be the full ``"Friendly Name:\\\\?\\usb#..."`` string;
        storing just the friendly name leaves OBS unable to bind the device (it reports
        the source active but 0x0, so Source Record captures nothing). Accepting either
        form and resolving here makes setup_scene forgiving and, crucially, correct.
        Raises with the available device names if nothing matches — never silently
        stores an unusable id.
        """
        ident = (identifier or "").strip()
        if not ident:
            raise OBSError(f"no {kind} device specified")
        devices = self.list_devices(kind)
        for d in devices:  # exact device_id
            if d.device_id == ident:
                return d.device_id
        for d in devices:  # exact name
            if d.name == ident:
                return d.device_id
        low = ident.lower()
        for d in devices:  # case-insensitive exact, then substring
            if d.name.strip().lower() == low:
                return d.device_id
        for d in devices:
            if low in d.name.strip().lower():
                return d.device_id
        avail = ", ".join(repr(d.name) for d in devices) or "(none found)"
        raise OBSError(f"{kind} device {identifier!r} not found in OBS. Available: {avail}.")

    def camera_dims(self, scene_name: str, source: str = "Camera") -> tuple[float, float]:
        """Current (sourceWidth, sourceHeight) of the camera source. 0x0 means the
        device is not delivering frames (not bound / wrong format / held by another app)."""
        items = self.req("GetSceneItemList", {"sceneName": scene_name}).get("sceneItems", [])
        for it in items:
            if it.get("sourceName") == source:
                t = it.get("sceneItemTransform", {})
                return float(t.get("sourceWidth") or 0.0), float(t.get("sourceHeight") or 0.0)
        return 0.0, 0.0

    def wait_camera_streaming(self, scene_name: str, source: str = "Camera",
                              timeout: float = 10.0) -> bool:
        """Poll until the camera source actually produces frames (sourceWidth>0), or
        timeout. Catches the dead-camera case BEFORE a recording silently yields nothing."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            w, h = self.camera_dims(scene_name, source)
            if w > 0 and h > 0:
                return True
            time.sleep(0.5)
        return False

    # -- scene construction -------------------------------------------------
    def build_scene(self, scene_name: str, monitor_id: str, camera_id: str, mic_id: str,
                    camera_record_dir: str, mic_track: int = 1) -> dict[str, Any]:
        """Idempotent. Re-calling with the same devices does NOT tear down and reopen
        the camera -- rapid open/close of a USB webcam exhausts it (DShow 0x800705AA)
        and it stops producing frames. We only (re)create an input when it is missing
        or its device changed, so the camera opens at most once per device choice.
        """
        self._ensure_scene(scene_name)

        # Resolve each identifier (a friendly NAME or a device_id) to the exact device_id
        # OBS needs. Without this, a name passed for the camera is stored verbatim and the
        # webcam never opens (active but 0x0 -> empty camera file).
        monitor_id = self._resolve_device(KIND_MONITOR, monitor_id)
        camera_id = self._resolve_device(KIND_CAMERA, camera_id)
        mic_id = self._resolve_device(KIND_MIC, mic_id)

        churned = False

        churned |= self._ensure_input(
            scene_name, "Screen", KIND_MONITOR, "monitor_id", monitor_id,
            {"monitor_id": monitor_id, "capture_cursor": True})[1]

        # camera ENABLED (so the webcam streams frames) but pushed OFF-CANVAS so it
        # never appears in the screen recording. Source Record captures its native
        # frames directly, independent of scene placement.
        cam_item, cam_created = self._ensure_input(
            scene_name, "Camera", KIND_CAMERA, "video_device_id", camera_id,
            {"video_device_id": camera_id})
        if cam_created:
            v = self.req("GetVideoSettings")
            self.req("SetSceneItemTransform", {
                "sceneName": scene_name, "sceneItemId": cam_item,
                "sceneItemTransform": {"positionX": float(v["baseWidth"]) + 50.0,
                                       "positionY": 0.0},
            })
        churned |= cam_created
        churned |= self._ensure_source_record("Camera", camera_record_dir)

        churned |= self._ensure_input(
            scene_name, "Mic", KIND_MIC, "device_id", mic_id, {"device_id": mic_id})[1]
        # ensure the mic is on `mic_track` (default 1) so it lands in OBS's main
        # recording file -> the screen .mp4 carries the spoken audio for the
        # composite + transcription downstream.
        self.req("SetInputAudioTracks", {
            "inputName": "Mic", "inputAudioTracks": {str(mic_track): True}})

        # Prevent DOUBLED audio: OBS's global mic source (e.g. "Mic/Aux") often captures
        # the same physical device as our "Mic", so the voice records twice with a small
        # offset (echoey). Mute every other mic-type input so only our "Mic" is recorded.
        for inp in self.req("GetInputList").get("inputs", []):
            if inp["inputKind"] == KIND_MIC and inp["inputName"] != "Mic":
                self.req("SetInputMute", {"inputName": inp["inputName"], "inputMuted": True})

        self.req("SetCurrentProgramScene", {"sceneName": scene_name})
        self._camera_dir = camera_record_dir
        # only impose the settle when something was actually (re)created; an unchanged,
        # already-armed scene records immediately.
        if churned:
            self._record_ready_at = time.monotonic() + self.FILTER_SETTLE_S

        # Verify the camera is genuinely delivering frames before declaring the scene
        # ready. A dshow source can report active while producing 0x0 (wrong/blocked
        # device, no negotiated format), in which case Source Record writes nothing.
        # Fail loudly with an actionable message instead of a false "scene ready".
        streaming = self.wait_camera_streaming(scene_name)
        if not streaming:
            raise OBSError(
                "Camera source 'Camera' is active but delivering no frames (0x0) — the "
                "webcam is not capturing, so the isolated camera file would be empty. In "
                "OBS open the 'Camera' source Properties: set Video Format (e.g. MJPEG) and "
                "a supported Resolution, uncheck 'Deactivate when not showing', and make "
                "sure no other app is using the camera. Then call setup_scene again."
            )
        return {
            "scene": scene_name, "screen_source": "Screen", "camera_source": "Camera",
            "mic_source": "Mic", "mic_track": mic_track,
            "camera_record_dir": camera_record_dir, "rebuilt": churned,
            "camera_streaming": streaming,
        }

    def _ensure_input(self, scene: str, name: str, kind: str, dev_key: str,
                      dev_val: str, settings: dict[str, Any]) -> tuple[int, bool]:
        """Create the input, or reuse it untouched if it already exists with the same
        device. Returns (sceneItemId, created). Reuse avoids reopening the device."""
        if name in self._input_names():
            cur = self.req("GetInputSettings", {"inputName": name}).get("inputSettings", {})
            if cur.get(dev_key) == dev_val:
                sid = self.req("GetSceneItemId",
                               {"sceneName": scene, "sourceName": name})["sceneItemId"]
                return sid, False
            self._ensure_absent(name)  # device changed -> one controlled reopen
        r = self.req("CreateInput", {
            "sceneName": scene, "inputName": name, "inputKind": kind,
            "inputSettings": settings, "sceneItemEnabled": True})
        return r["sceneItemId"], True

    def _ensure_source_record(self, source_name: str, record_dir: str) -> bool:
        """Ensure the Source Record filter exists. Returns True if newly created (the
        new filter needs a settle before it will arm on recording start)."""
        if not self.has_source_record():
            raise OBSError(
                "Source Record plugin is not loaded in OBS. Install it from "
                "https://github.com/exeldro/obs-source-record/releases and restart OBS."
            )
        os.makedirs(record_dir, exist_ok=True)
        settings = {
            # record_mode 3 = "Recording": camera file starts/stops with the main
            # OBS recording (0 = "None" records nothing). rate_control MUST be
            # uppercase "CBR" -- the plugin default "cbr" silently fails nvenc init.
            "record_mode": 3, "path": record_dir,
            "filename_formatting": "camera_%CCYY-%MM-%DD_%hh-%mm-%ss",
            "rec_format": "mkv", "rate_control": "CBR", "scale_type": 3,
            # UNCAP the camera file: the plugin defaults to splitting at 900s/2048MB,
            # which would chop a 30-40 min take. 0 = no limit.
            "max_time_sec": 0, "max_size_mb": 0,
        }
        existing = self.req("GetSourceFilterList", {"sourceName": source_name}).get("filters", [])
        if any(f["filterName"] == "DopestClipSourceRecord" for f in existing):
            self.req("SetSourceFilterSettings", {
                "sourceName": source_name, "filterName": "DopestClipSourceRecord",
                "filterSettings": settings})
            return False
        self.req("CreateSourceFilter", {
            "sourceName": source_name, "filterName": "DopestClipSourceRecord",
            "filterKind": SOURCE_RECORD_FILTER_KIND, "filterSettings": settings})
        return True

    # -- recording ----------------------------------------------------------
    def start_recording(self) -> None:
        # block until the Source Record filter has settled, so the camera file is
        # never silently dropped by a build->record race
        wait = self._record_ready_at - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        # snapshot camera dir so stop_recording can identify THIS take's camera file
        if self._camera_dir:
            self._cam_before = set(glob.glob(os.path.join(self._camera_dir, "camera_*")))
        self.req("StartRecord")

    def verify_camera_capture(self, camera_record_dir: str, attempts: int = 5) -> bool:
        """Prove the isolated camera file actually records before the user relies on
        it. Source Record auto-start on a freshly built scene is intermittently racy,
        so we do throwaway test recordings (deleting both the test screen file and
        test camera file) until one produces a camera file. Returns True once proven.
        """
        main_dir = self.record_directory()
        for _ in range(attempts):
            cam_before = set(glob.glob(os.path.join(camera_record_dir, "camera_*")))
            main_before = set(glob.glob(os.path.join(main_dir, "*")))
            self.start_recording()
            time.sleep(2.0)
            self.req("StopRecord")
            time.sleep(1.5)
            cam_new = set(glob.glob(os.path.join(camera_record_dir, "camera_*"))) - cam_before
            main_new = set(glob.glob(os.path.join(main_dir, "*"))) - main_before
            for f in cam_new | main_new:  # remove throwaway test artifacts
                try:
                    os.remove(f)
                except OSError:
                    pass
            if cam_new:
                return True
            time.sleep(2.0)  # let the device/filter settle further, then retry
        return False

    def stop_recording(self, timeout: float = 15.0) -> dict[str, Any]:
        """Stop and return this take's file pair: the screen recording (OBS main
        output) and the isolated camera file produced during the same window.

        Source Record finalizes the camera .mkv asynchronously AFTER StopRecord — a
        large take can take several seconds to flush. A fixed short sleep raced that and
        reported camera=null even though the file existed. Instead we poll for the new
        camera_* file and wait until its size stops growing (finalized) before returning.
        """
        screen = self.req("StopRecord").get("outputPath")
        camera = None
        if self._camera_dir:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                new = set(glob.glob(os.path.join(self._camera_dir, "camera_*"))) - self._cam_before
                if new:
                    cand = max(new, key=os.path.getmtime)
                    size1 = os.path.getsize(cand)
                    time.sleep(0.5)
                    # finalized when the size has stopped growing
                    if os.path.exists(cand) and os.path.getsize(cand) == size1 and size1 > 0:
                        camera = cand
                        break
                time.sleep(0.5)
        return {"screen": screen, "camera": camera}

    def recording_status(self) -> dict[str, Any]:
        s = self.req("GetRecordStatus")
        return {
            "active": s.get("outputActive"),
            "timecode": s.get("outputTimecode"),
            "bytes": s.get("outputBytes"),
        }

    def record_directory(self) -> str:
        return self.req("GetRecordDirectory")["recordDirectory"]

    # -- helpers ------------------------------------------------------------
    def _scene_names(self) -> list[str]:
        return [s["sceneName"] for s in self.req("GetSceneList").get("scenes", [])]

    def _input_names(self) -> list[str]:
        return [i["inputName"] for i in self.req("GetInputList").get("inputs", [])]

    def _first_scene(self) -> str:
        names = self._scene_names()
        if names:
            return names[0]
        self.req("CreateScene", {"sceneName": "Scene"})
        return "Scene"

    def _ensure_scene(self, name: str) -> None:
        if name not in self._scene_names():
            self.req("CreateScene", {"sceneName": name})

    def _ensure_absent(self, input_name: str, timeout: float = 5.0) -> None:
        if input_name not in self._input_names():
            return
        self.req("RemoveInput", {"inputName": input_name})
        # active capture devices (camera/mic/display) tear down asynchronously
        # (~100-200ms measured); wait for the source to actually disappear before
        # callers recreate a same-named input, else CreateInput hits 601.
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if input_name not in self._input_names():
                return
            time.sleep(0.1)
        raise OBSError(f"Input {input_name!r} did not tear down within {timeout}s.")


# --- env-driven connection config + public plain functions ------------------

def _env_host() -> str:
    return os.environ.get("OBS_WS_HOST", "localhost").strip() or "localhost"


def _env_port() -> int:
    return int(os.environ.get("OBS_WS_PORT", "4455"))


def _env_password() -> str | None:
    pw = os.environ.get("OBS_WS_PASSWORD", "").strip()
    return pw or None


def _env_scene() -> str:
    return os.environ.get("OBS_SCENE_NAME", "DopestClipRec").strip() or "DopestClipRec"


def _env_camera_dir() -> str:
    """Where Source Record writes the isolated camera .mkv. Defaults next to the
    projects root (its parent dir / 'obs_camera') so camera takes sit alongside the
    rest of the studio's files. Override with OBS_CAMERA_DIR."""
    env = os.environ.get("OBS_CAMERA_DIR", "").strip()
    if env:
        return env
    return str(config.PROJECTS_ROOT.parent / "obs_camera")


# A module-level long-lived client. The build->record->stop sequence carries
# per-take state (the Source Record settle deadline, the camera-dir snapshot used to
# identify THIS take's camera file) ON THE INSTANCE, so the same client MUST span all
# of setup_scene/start_recording/stop_recording — a fresh connection per call would
# lose _camera_dir and never find the isolated camera file. Reusing one socket also
# avoids hammering OBS with reconnects.
_CLIENT: OBSClient | None = None


def _client() -> OBSClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OBSClient(host=_env_host(), port=_env_port(), password=_env_password())
    return _CLIENT


def disconnect() -> None:
    """Close and drop the shared client (next call reconnects). Mainly for shutdown."""
    global _CLIENT
    if _CLIENT is not None:
        _CLIENT.close()
        _CLIENT = None


def list_devices() -> dict[str, list[dict[str, str]]]:
    """Enumerate available monitors, cameras, and mics. Each device is
    {"name", "device_id"} — the device_id is what setup_scene takes."""
    c = _client()

    def dump(devs: list[Device]) -> list[dict[str, str]]:
        return [{"name": d.name, "device_id": d.device_id} for d in devs]
    return {
        "monitors": dump(c.list_monitors()),
        "cameras": dump(c.list_cameras()),
        "mics": dump(c.list_mics()),
    }


def setup_scene(monitor: str, camera: str, mic: str, mic_track: int = 1) -> dict[str, Any]:
    """Build (idempotently) the recording scene: a monitor capture (the screen file),
    a camera dshow source ENABLED + off-canvas with a Source Record filter writing a
    separate clean camera file, and a mic on `mic_track` of the screen file's audio.
    `monitor`/`camera`/`mic` accept EITHER the friendly name OR the device_id from
    list_devices() (resolved to the exact OBS device_id internally). Reuses unchanged
    inputs so the webcam is never reopened rapidly (DShow 0x800705AA). Raises if the
    camera is not actually delivering frames after the scene is built."""
    return _client().build_scene(
        _env_scene(), monitor, camera, mic, _env_camera_dir(), mic_track=mic_track)


def start_recording() -> dict[str, Any]:
    """Start the OBS recording (and, via the Source Record filter, the isolated camera
    file). Blocks until a freshly built Source Record filter has settled so the camera
    file is never silently dropped by a build->record race. Call setup_scene first."""
    c = _client()
    c.start_recording()
    return {"started": True, "status": c.recording_status()}


def stop_recording() -> dict[str, Any]:
    """Stop the recording and return this take's {"screen", "camera"} file pair."""
    return _client().stop_recording()


def recording_status() -> dict[str, Any]:
    """Current OBS recording status: {"active", "timecode", "bytes"}."""
    return _client().recording_status()
