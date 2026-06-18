"""Unified FastMCP server — the agent face of dopest-clip.

Registers every operation in api.OPERATIONS as an MCP tool and every learn:// doc as
an MCP resource. The SAME api.OPERATIONS callables back the Electron sidecar, so the
agent and the human editor drive identical operations.
"""

from mcp.server.fastmcp import FastMCP

from . import api, learn

mcp = FastMCP(
    "dopest-clip",
    instructions=(
        "Open-source, provider-agnostic media studio: record (OBS) -> edit "
        "(transcript-driven clipper) -> process audio -> generate images. READ THE LEARN "
        "RESOURCES FIRST (MCP resources, not tools): start with learn://overview, then the "
        "subsystem you need (learn://editing, learn://recording, learn://reframe, "
        "learn://captions, learn://audio, learn://image, learn://providers, learn://gotchas). "
        "The editing loop is create_project -> transcribe -> READ transcript.txt -> design an "
        "EDL -> validate_edl -> render -> verify_clip. Cloud capabilities route through a "
        "provider registry (list_providers / set_provider) — FlowDot is one option among many, "
        "never required."
    ),
)


# --- learn:// resources ---------------------------------------------------------------
def _register_resources() -> None:
    for uri, body in learn.RESOURCES.items():
        # bind body via default arg so each closure returns its own text
        def _make(_body: str):
            def _res() -> str:
                return _body
            return _res
        mcp.resource(uri)(_make(body))


# --- tools ----------------------------------------------------------------------------
def _register_tools() -> None:
    for name, fn in api.OPERATIONS.items():
        mcp.add_tool(fn, name=name, description=(fn.__doc__ or "").strip() or None)


_register_resources()
_register_tools()


def run() -> None:
    """Run the MCP server over stdio."""
    mcp.run()
