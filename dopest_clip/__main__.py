"""Entry point.

    python -m dopest_clip            -> MCP stdio server (agent face)
    python -m dopest_clip --serve    -> localhost JSON-RPC sidecar (Electron face)
        optional: --host 127.0.0.1 --port 8765
"""

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(prog="dopest-clip", description="dopest-clip media studio")
    parser.add_argument("--serve", action="store_true",
                        help="run the localhost JSON-RPC sidecar (for the Electron editor) instead of MCP stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    if args.serve:
        from . import sidecar
        sidecar.serve(args.host, args.port)
    else:
        from . import server
        server.run()


if __name__ == "__main__":  # pragma: no cover
    main()
