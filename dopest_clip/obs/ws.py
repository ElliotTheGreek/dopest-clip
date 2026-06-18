"""Minimal, reliable obs-websocket v5 client.

Why not obsws-python: its request path does a blind ``recv()`` with no
requestId correlation, so any non-matching frame (or a stale read) is returned
as if it were the reply — which made deletes appear to succeed while actually
reading the wrong frame. This client matches every reply to the requestId it
sent and skips anything else, which is correct against OBS 32 / websocket 5.7.

The ``websocket-client`` dependency (imported as ``websocket``) is loaded LAZILY
inside ``connect()`` so importing this module — and unit-testing the requestId
correlation with a fake socket — needs no extra installed. Install the recording
extra with: ``pip install dopest-clip[obs]``.
"""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any


class OBSError(RuntimeError):
    pass


def _import_websocket():
    """Lazy-import websocket-client. Kept out of module import so the WS framing /
    requestId-correlation logic can be tested with a fake socket and so the rest of
    the package never requires the recording extra."""
    try:
        import websocket  # provided by websocket-client
    except ImportError as e:  # noqa: BLE001
        raise OBSError(
            "OBS recording control needs the websocket-client library. "
            "Install the recording extra with: pip install dopest-clip[obs]"
        ) from e
    return websocket


def _transport_errors() -> tuple[type[BaseException], ...]:
    """Exception types that mean the SOCKET died (vs an OBS logical error). OSError
    covers WinError 10053 / ConnectionAbortedError; websocket-client's WebSocketException
    covers a closed connection. A failed-request OBSError is a RuntimeError and is NOT in
    here, so it is never mistaken for a transport drop and retried."""
    try:
        import websocket
        return (OSError, websocket.WebSocketException)
    except ImportError:
        return (OSError,)


class WSClient:
    def __init__(self, host: str = "localhost", port: int = 4455,
                 password: str | None = None, timeout: float = 5.0):
        self._host, self._port, self._password = host, port, password
        self._timeout = timeout
        self._ws: Any = None
        self._id = 0

    def connect(self) -> None:
        websocket = _import_websocket()
        try:
            ws = websocket.WebSocket()
            ws.connect(f"ws://{self._host}:{self._port}", timeout=self._timeout)
            hello = json.loads(ws.recv())
        except OBSError:
            raise
        except Exception as e:  # noqa: BLE001
            raise OBSError(
                f"Could not reach OBS websocket at {self._host}:{self._port}. "
                "Is OBS running with Tools -> WebSocket Server Settings enabled?"
            ) from e

        identify: dict[str, Any] = {"rpcVersion": 1, "eventSubscriptions": 0}
        auth = hello["d"].get("authentication")
        if auth:
            # SHA256 challenge-response auth when a password is set: base64 of
            # sha256(password + salt), then base64 of sha256(that secret + challenge).
            if not self._password:
                raise OBSError("OBS websocket requires a password but none was given.")
            secret = base64.b64encode(
                hashlib.sha256((self._password + auth["salt"]).encode()).digest()
            ).decode()
            identify["authentication"] = base64.b64encode(
                hashlib.sha256((secret + auth["challenge"]).encode()).digest()
            ).decode()
        ws.send(json.dumps({"op": 1, "d": identify}))
        ident = json.loads(ws.recv())
        if ident.get("op") != 2:
            raise OBSError(
                "OBS rejected the websocket identify (wrong password?). "
                "Use Tools -> WebSocket Server Settings -> Show Connect Info to confirm."
            )
        self._ws = ws

    @property
    def ws(self) -> Any:
        if self._ws is None:
            self.connect()
        assert self._ws is not None
        return self._ws

    def request(self, req_type: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a request and return its responseData. If the socket has died (e.g. OBS
        was restarted -> WinError 10053), drop it, reconnect, and retry ONCE so a restarted
        OBS heals automatically. A logical failure (result:false -> OBSError) is NOT a
        transport error and is never retried."""
        transport = _transport_errors()
        try:
            return self._do_request(req_type, data)
        except transport:
            self.close()  # drop the dead socket; the `ws` property reconnects lazily
            try:
                return self._do_request(req_type, data)
            except transport as e2:  # noqa: BLE001
                raise OBSError(
                    f"OBS websocket connection lost and reconnect failed: {e2}"
                ) from e2

    def _do_request(self, req_type: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        self._id += 1
        rid = f"r{self._id}"
        payload: dict[str, Any] = {"op": 6, "d": {"requestType": req_type, "requestId": rid}}
        if data:
            payload["d"]["requestData"] = data
        self.ws.send(json.dumps(payload))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("op") != 7:  # not a RequestResponse (e.g. an event) -> skip
                continue
            d = msg["d"]
            if d.get("requestId") != rid:  # response to some other request -> skip
                continue
            status = d["requestStatus"]
            if not status.get("result"):
                raise OBSError(
                    f"{req_type} failed [{status.get('code')}]: {status.get('comment')}"
                )
            return d.get("responseData", {}) or {}

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            finally:
                self._ws = None
