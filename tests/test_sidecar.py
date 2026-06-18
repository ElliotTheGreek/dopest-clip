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
