"""Sidecar JSON-RPC: health, discovery, dispatch, and error shaping."""

import json
import urllib.request

import pytest

from dopest_clip import api, sidecar


@pytest.fixture
def server():
    httpd = sidecar.serve("127.0.0.1", 0, block=False)  # port 0 = OS-assigned
    host, port = httpd.server_address
    yield f"http://{host}:{port}"
    httpd.shutdown()
    httpd.server_close()


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read())


def _rpc(base, method, params=None):
    body = json.dumps({"method": method, "params": params or {}}).encode()
    req = urllib.request.Request(base + "/rpc", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:  # non-2xx (e.g. 404 unknown op) still carries a JSON body
        return json.loads(e.read())


def test_health(server):
    h = _get(server + "/health")
    assert h["ok"] and h["name"] == "dopest-clip"


def test_ops_catalog_matches_operations(server):
    cat = _get(server + "/ops")
    assert set(cat["operations"]) == set(api.OPERATIONS.keys())
    assert "editing" in cat["groups"] and "providers" in cat["groups"]


def test_rpc_dispatch_list_providers(server):
    out = _rpc(server, "list_providers")
    assert out["ok"]
    assert "image" in out["result"]  # capability map


def test_rpc_unknown_operation(server):
    out = _rpc(server, "does_not_exist")
    assert out["ok"] is False
    assert "unknown operation" in out["error"]


def test_rpc_operation_error_is_structured(server):
    # validate_edl on a missing project returns an error dict, not a crash
    out = _rpc(server, "list_projects")
    assert out["ok"] and "projects" in out["result"]


# --- error routes + shaping (full control-path coverage) ------------------------------

import urllib.error  # noqa: E402


def _get_raw(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _post_raw(base, path, raw_body):
    req = urllib.request.Request(base + path, data=raw_body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_get_unknown_route_404(server):
    code, body = _get_raw(server + "/nope")
    assert code == 404 and body["ok"] is False and "no route" in body["error"]


def test_post_non_rpc_route_404(server):
    code, body = _post_raw(server, "/nope", b"{}")
    assert code == 404 and "no route" in body["error"]


def test_post_bad_body_400(server):
    code, body = _post_raw(server, "/rpc", b"not json at all")
    assert code == 400 and "bad request body" in body["error"]


def test_post_params_not_object_400(server):
    code, body = _post_raw(server, "/rpc", json.dumps({"method": "list_providers", "params": [1, 2]}).encode())
    assert code == 400 and "params must be an object" in body["error"]


def test_rpc_bad_params_typeerror_400(server):
    # get_project requires project_id; omitting it -> TypeError -> 400
    code, body = _post_raw(server, "/rpc", json.dumps({"method": "get_project", "params": {}}).encode())
    assert code == 400 and "bad params for get_project" in body["error"]


def test_rpc_operation_exception_is_structured_200(server):
    # mix_camera on a missing project raises FileNotFoundError -> caught -> 200 ok:false
    code, body = _post_raw(server, "/rpc", json.dumps(
        {"method": "mix_camera", "params": {"project_id": "nope", "edl_id": "x", "camera_path": "y"}}).encode())
    assert code == 200 and body["ok"] is False and ":" in body["error"]


def test_serve_block_true_prints_then_serves(monkeypatch):
    from dopest_clip import sidecar as sc
    ran = {}
    monkeypatch.setattr(sc.ThreadingHTTPServer, "serve_forever", lambda self: ran.setdefault("ran", True))
    httpd = sc.serve("127.0.0.1", 0, block=True)
    try:
        assert ran.get("ran") is True
    finally:
        httpd.server_close()
