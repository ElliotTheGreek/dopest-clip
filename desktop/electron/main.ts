// Electron main process for the dopest-clip editor.
//
// Responsibilities:
//   1. Spawn the python JSON-RPC sidecar (`python -m dopest_clip --serve --port <port>`).
//   2. Poll GET /health until it is up (or fail loudly after a timeout).
//   3. Create the BrowserWindow and load the renderer.
//   4. Proxy all sidecar HTTP traffic (rpc/ops/health) through IPC so the renderer never
//      makes a raw network call itself (keeps the renderer sandboxed; CORS is moot).
//   5. Read local image/video files from disk as data URLs for the renderer (frame and
//      image results are absolute file paths).
//   6. Kill the sidecar on quit.
//
// The sidecar binds 127.0.0.1 only and has no auth — never expose it. Configurable via
// env: DOPEST_PYTHON (python executable, default "python") and DOPEST_SIDECAR_PORT
// (default 8765). DOPEST_SIDECAR_EXTERNAL=1 skips spawning (attach to an already-running
// sidecar, e.g. one launched by hand for debugging).

import { app, BrowserWindow, ipcMain, protocol, net } from "electron";
import { spawn, ChildProcess } from "node:child_process";
import { readFile } from "node:fs/promises";
import * as path from "node:path";
import { pathToFileURL } from "node:url";
import * as http from "node:http";

const PYTHON = process.env.DOPEST_PYTHON || "python";
const SIDECAR_PORT = Number(process.env.DOPEST_SIDECAR_PORT || "8765");
const SIDECAR_HOST = "127.0.0.1";
const SIDECAR_BASE = `http://${SIDECAR_HOST}:${SIDECAR_PORT}`;
const SIDECAR_EXTERNAL = process.env.DOPEST_SIDECAR_EXTERNAL === "1";

// The python package lives one directory up from desktop/.
const REPO_ROOT = path.resolve(app.getAppPath(), "..");

let sidecar: ChildProcess | null = null;
let win: BrowserWindow | null = null;

// --- tiny HTTP helper against the localhost sidecar -----------------------------------

interface HttpResult {
  status: number;
  body: string;
}

function httpRequest(method: "GET" | "POST", urlPath: string, body?: string): Promise<HttpResult> {
  return new Promise((resolve, reject) => {
    const data = body ?? "";
    const req = http.request(
      {
        host: SIDECAR_HOST,
        port: SIDECAR_PORT,
        path: urlPath,
        method,
        headers: {
          "Content-Type": "application/json",
          "Content-Length": Buffer.byteLength(data),
        },
      },
      (res) => {
        const chunks: Buffer[] = [];
        res.on("data", (c) => chunks.push(c as Buffer));
        res.on("end", () =>
          resolve({ status: res.statusCode || 0, body: Buffer.concat(chunks).toString("utf-8") }),
        );
      },
    );
    req.on("error", reject);
    if (method === "POST") req.write(data);
    req.end();
  });
}

async function waitForHealth(timeoutMs = 30000): Promise<void> {
  const start = Date.now();
  let lastErr: unknown = null;
  while (Date.now() - start < timeoutMs) {
    try {
      const r = await httpRequest("GET", "/health");
      if (r.status === 200) {
        const parsed = JSON.parse(r.body);
        if (parsed.ok) return;
      }
      lastErr = `unexpected /health response: ${r.status} ${r.body}`;
    } catch (e) {
      lastErr = e;
    }
    await new Promise((res) => setTimeout(res, 300));
  }
  throw new Error(`sidecar did not become healthy within ${timeoutMs}ms (last: ${String(lastErr)})`);
}

// --- sidecar lifecycle ----------------------------------------------------------------

function spawnSidecar(): void {
  if (SIDECAR_EXTERNAL) {
    console.log("[dopest] DOPEST_SIDECAR_EXTERNAL=1 — not spawning; attaching to existing sidecar");
    return;
  }
  console.log(`[dopest] spawning sidecar: ${PYTHON} -m dopest_clip --serve --port ${SIDECAR_PORT} (cwd=${REPO_ROOT})`);
  sidecar = spawn(PYTHON, ["-m", "dopest_clip", "--serve", "--host", SIDECAR_HOST, "--port", String(SIDECAR_PORT)], {
    cwd: REPO_ROOT,
    env: process.env,
    stdio: ["ignore", "pipe", "pipe"],
  });
  sidecar.stdout?.on("data", (d) => process.stdout.write(`[sidecar] ${d}`));
  sidecar.stderr?.on("data", (d) => process.stderr.write(`[sidecar] ${d}`));
  sidecar.on("exit", (code, signal) => {
    console.log(`[dopest] sidecar exited code=${code} signal=${signal}`);
    sidecar = null;
  });
  sidecar.on("error", (err) => {
    console.error(`[dopest] failed to spawn sidecar: ${err}`);
  });
}

function killSidecar(): void {
  if (sidecar && !sidecar.killed) {
    console.log("[dopest] killing sidecar");
    sidecar.kill();
    sidecar = null;
  }
}

// --- IPC handlers ---------------------------------------------------------------------

function registerIpc(): void {
  // RPC: returns the parsed {ok, result|error} envelope as-is; renderer client unwraps.
  ipcMain.handle("dopest:rpc", async (_e, method: string, params: unknown) => {
    const r = await httpRequest("POST", "/rpc", JSON.stringify({ method, params: params ?? {} }));
    try {
      return JSON.parse(r.body);
    } catch {
      return { ok: false, error: `non-JSON rpc response (status ${r.status}): ${r.body.slice(0, 500)}` };
    }
  });

  ipcMain.handle("dopest:ops", async () => {
    const r = await httpRequest("GET", "/ops");
    return JSON.parse(r.body);
  });

  ipcMain.handle("dopest:health", async () => {
    const r = await httpRequest("GET", "/health");
    return JSON.parse(r.body);
  });

  // Read a local file (frame/image PNG, etc.) as a data URL for <img>/<video>.
  ipcMain.handle("dopest:readFileAsDataURL", async (_e, filePath: string) => {
    const buf = await readFile(filePath);
    const ext = path.extname(filePath).toLowerCase().replace(".", "");
    const mime =
      ext === "png" ? "image/png" :
      ext === "jpg" || ext === "jpeg" ? "image/jpeg" :
      ext === "gif" ? "image/gif" :
      ext === "webp" ? "image/webp" :
      ext === "svg" ? "image/svg+xml" :
      ext === "mp4" ? "video/mp4" :
      ext === "webm" ? "video/webm" :
      ext === "mov" ? "video/quicktime" :
      ext === "mp3" ? "audio/mpeg" :
      ext === "wav" ? "audio/wav" :
      "application/octet-stream";
    return `data:${mime};base64,${buf.toString("base64")}`;
  });
}

// --- window ---------------------------------------------------------------------------

function createWindow(): void {
  win = new BrowserWindow({
    width: 1440,
    height: 900,
    backgroundColor: "#1a1a1e",
    title: "dopest-clip editor",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  const devUrl = process.env.VITE_DEV_SERVER_URL;
  if (devUrl) {
    void win.loadURL(devUrl);
    win.webContents.openDevTools({ mode: "detach" });
  } else {
    // Load the built renderer over the custom app:// scheme, NOT file://. Chromium gives
    // file:// an opaque ("null") origin, so ES-module scripts won't load from it and the
    // React app never mounts. app:// is a registered standard+secure origin where modules
    // load normally.
    void win.loadURL("app://bundle/index.html");
  }

  win.on("closed", () => {
    win = null;
  });
}

// Register a `dopest-file://` protocol as an alternative to data URLs for large media
// (lets <video> stream rather than loading the whole file). Renderer uses readFileAsDataURL
// for images; this is available for video sources.
function registerFileProtocol(): void {
  protocol.handle("dopest-file", (request) => {
    const url = new URL(request.url);
    // dopest-file://local/<absolute-path-uri-encoded>
    const decoded = decodeURIComponent(url.pathname.replace(/^\//, ""));
    return net.fetch(pathToFileURL(decoded).toString());
  });
}

// Serve the built renderer (dist/) over app://bundle/<path> with explicit MIME types so
// module scripts load with a valid JavaScript content-type.
const RENDERER_DIST = path.join(__dirname, "../dist");

function registerAppProtocol(): void {
  protocol.handle("app", async (request) => {
    const url = new URL(request.url);
    let rel = decodeURIComponent(url.pathname).replace(/^\/+/, "");
    if (!rel) rel = "index.html";
    const filePath = path.join(RENDERER_DIST, rel);
    try {
      const data = await readFile(filePath);
      const ext = path.extname(filePath).toLowerCase();
      const type =
        ext === ".js" || ext === ".mjs" ? "text/javascript" :
        ext === ".css" ? "text/css" :
        ext === ".html" ? "text/html" :
        ext === ".json" ? "application/json" :
        ext === ".svg" ? "image/svg+xml" :
        ext === ".png" ? "image/png" :
        ext === ".woff2" ? "font/woff2" :
        ext === ".woff" ? "font/woff" :
        "application/octet-stream";
      return new Response(data, { headers: { "content-type": type } });
    } catch {
      return new Response("not found", { status: 404 });
    }
  });
}

app.whenReady().then(async () => {
  registerFileProtocol();
  registerAppProtocol();
  registerIpc();
  spawnSidecar();
  try {
    await waitForHealth();
    console.log("[dopest] sidecar healthy");
  } catch (e) {
    console.error(`[dopest] ${e}`);
    // Still create the window; the renderer surfaces a connection error in the UI.
  }
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

// Register the custom scheme as privileged before app ready.
protocol.registerSchemesAsPrivileged([
  { scheme: "dopest-file", privileges: { standard: true, secure: true, supportFetchAPI: true, stream: true, bypassCSP: true } },
  { scheme: "app", privileges: { standard: true, secure: true, supportFetchAPI: true, corsEnabled: true } },
]);

app.on("window-all-closed", () => {
  killSidecar();
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", killSidecar);
app.on("quit", killSidecar);
process.on("exit", killSidecar);
