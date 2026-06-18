# dopest-clip desktop editor (Electron)

The human face over the same `dopest_clip` engine the MCP agent drives. Electron main spawns
the Python JSON-RPC sidecar, waits for `/health`, then renders a React editor that round-trips
the project + EDL model over IPC.

## Architecture

```
electron/main.ts      spawns `python -m dopest_clip --serve`, proxies rpc/ops/health over IPC,
                      reads local frame/image files as data URLs, kills sidecar on quit
electron/preload.ts   exposes window.dopest = { rpc, ops, health, readFileAsDataURL }
src/api/sidecar.ts    typed client over window.dopest.rpc; unwraps {ok,result|error}, throws on ok:false
src/edl/edl.ts        PURE TS EDL helpers (add/remove/reorder/move/duplicate/styling/keyframes)
src/edl/types.ts      EDL + validate_edl result types, mirror dopest_clip/edl.py
src/state/useEditor.ts central session state (project, transcript, EDL, validation)
src/components/        Projects · Transcript · Timeline · Inspector · Preview · Providers
```

## Manual launch

From this `desktop/` directory:

```
npm install        # installs Electron + Vite + React (needs network; downloads Electron)
npm run dev        # starts Vite, builds main/preload, launches Electron;
                   # Electron auto-spawns the python sidecar on port 8765
```

Configurable via env:

- `DOPEST_PYTHON` — python executable used to launch the sidecar (default `python`). Point this
  at the project venv, e.g. `E:\FlowdotPlatform\dopest-clip\.venv\Scripts\python.exe`.
- `DOPEST_SIDECAR_PORT` — sidecar port (default `8765`).
- `DOPEST_SIDECAR_EXTERNAL=1` — do NOT spawn; attach to a sidecar you started by hand
  (`python -m dopest_clip --serve --port 8765`), useful for debugging.

The sidecar binds `127.0.0.1` only and has no auth — never expose it.

## Tests

```
npm run typecheck   # tsc for renderer + electron
npm run test        # vitest (pure renderer logic; no Electron, no Python)
```

`src/edl/edl.test.ts` covers the pure EDL helpers; `src/api/sidecar.test.ts` mocks
`window.dopest.rpc` and asserts envelope unwrapping + param passthrough.
