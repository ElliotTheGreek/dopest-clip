"""Localhost JSON-RPC sidecar — the bridge the Electron editor talks to.

Exposes the SAME api.OPERATIONS the MCP server exposes, over a tiny stdlib HTTP server
bound to 127.0.0.1. The Electron main process spawns `python -m dopest_clip --serve` and
the renderer round-trips the project JSON and calls operations by name.

Protocol (JSON-RPC-ish, deliberately minimal):
    POST /rpc   body {"method": "<operation>", "params": {...}}  -> {"ok": true, "result": ...}
                                                                  -> {"ok": false, "error": "..."}
    GET  /ops   -> {"groups": {...}, "operations": [names]}      (discovery)
    GET  /health-> {"ok": true, "name": "dopest-clip", "version": "..."}

Bound to localhost only; no auth (single-user desktop). Not for remote exposure.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__, api

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def _op_catalog() -> dict:
    return {
        "groups": {g: [fn.__name__ for fn in fns] for g, fns in api.GROUPS.items()},
        "operations": sorted(api.OPERATIONS.keys()),
    }


class _Handler(BaseHTTPRequestHandler):
    # quiet logging; the Electron main process captures stdout/stderr if it wants
    def log_message(self, *_args):  # noqa: N802
        pass

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            return self._send(200, {"ok": True, "name": "dopest-clip", "version": __version__})
        if self.path == "/ops":
            return self._send(200, _op_catalog())
        return self._send(404, {"ok": False, "error": f"no route {self.path}"})

    def do_POST(self):  # noqa: N802
        if self.path != "/rpc":
            return self._send(404, {"ok": False, "error": f"no route {self.path}"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError) as e:
            return self._send(400, {"ok": False, "error": f"bad request body: {e}"})

        method = req.get("method")
        params = req.get("params") or {}
        if not isinstance(params, dict):
            return self._send(400, {"ok": False, "error": "params must be an object"})
        fn = api.OPERATIONS.get(method)
        if fn is None:
            return self._send(404, {"ok": False, "error": f"unknown operation {method!r}"})
        try:
            result = fn(**params)
        except TypeError as e:
            return self._send(400, {"ok": False, "error": f"bad params for {method}: {e}"})
        except Exception as e:  # operation-level failure -> structured error, never a 500 crash
            return self._send(200, {"ok": False, "error": f"{type(e).__name__}: {e}"})
        return self._send(200, {"ok": True, "result": result})


def make_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), _Handler)


def serve(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, *, block: bool = True) -> ThreadingHTTPServer:
    """Start the sidecar. block=True runs forever (CLI use); block=False returns the
    server with a daemon thread already serving (test/embedded use)."""
    httpd = make_server(host, port)
    if block:
        print(f"dopest-clip sidecar on http://{host}:{port}  ({len(api.OPERATIONS)} operations)", flush=True)
        httpd.serve_forever()
        return httpd
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd
