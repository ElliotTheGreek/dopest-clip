"""MCP server wiring: every operation is a tool, every learn doc is a resource."""

import asyncio

from dopest_clip import api, learn, server


def test_every_operation_is_registered_as_a_tool():
    tools = asyncio.run(server.mcp.list_tools())
    names = {t.name for t in tools}
    assert names == set(api.OPERATIONS.keys())
    # spot-check a few across groups
    for expected in ("create_project", "render", "tts", "image_generate", "list_providers", "setup_scene"):
        assert expected in names


def test_tools_have_descriptions_and_schemas():
    tools = asyncio.run(server.mcp.list_tools())
    by_name = {t.name: t for t in tools}
    t = by_name["render"]
    assert t.description  # pulled from the docstring
    assert "project_id" in (t.inputSchema.get("properties") or {})


def test_learn_resources_registered():
    resources = asyncio.run(server.mcp.list_resources())
    uris = {str(r.uri) for r in resources}
    for key in learn.RESOURCES:
        assert key in uris


def test_resource_bodies_are_distinct():
    # the closure-per-body fix: each resource returns its own text, not the last one
    bodies = set(learn.RESOURCES.values())
    assert len(bodies) == len(learn.RESOURCES)


def test_read_resource_returns_its_body():
    # reading a resource invokes the registered closure (covers the per-body return path)
    res = asyncio.run(server.mcp.read_resource("learn://overview"))
    assert res and "dopest-clip" in str(res)


def test_run_invokes_mcp_run(monkeypatch):
    ran = {}
    monkeypatch.setattr(server.mcp, "run", lambda: ran.setdefault("ran", True))
    server.run()
    assert ran["ran"] is True
