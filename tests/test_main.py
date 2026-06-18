"""CLI entry point (`python -m dopest_clip`): the --serve branch starts the sidecar, the
default branch runs the MCP stdio server. Both delegates are stubbed so nothing blocks."""

from dopest_clip import __main__ as m


def test_main_serve_branch_starts_sidecar(monkeypatch):
    from dopest_clip import sidecar
    calls = {}
    monkeypatch.setattr(sidecar, "serve", lambda host, port: calls.update(host=host, port=port))
    m.main(["--serve", "--host", "1.2.3.4", "--port", "9999"])
    assert calls == {"host": "1.2.3.4", "port": 9999}


def test_main_default_runs_mcp_server(monkeypatch):
    from dopest_clip import server
    ran = {}
    monkeypatch.setattr(server, "run", lambda: ran.setdefault("ran", True))
    m.main([])
    assert ran["ran"] is True
