// Preload: exposes a minimal, safe `window.dopest` bridge to the renderer.
// Everything goes through IPC to the main process; the renderer has no direct node or
// network access.

import { contextBridge, ipcRenderer } from "electron";

export interface RpcEnvelope<T = unknown> {
  ok: boolean;
  result?: T;
  error?: string;
}

export interface OpsCatalog {
  groups: Record<string, string[]>;
  operations: string[];
}

export interface HealthInfo {
  ok: boolean;
  name: string;
  version: string;
}

const api = {
  /** Call a sidecar operation by name. Returns the raw {ok, result|error} envelope. */
  rpc: (method: string, params?: Record<string, unknown>): Promise<RpcEnvelope> =>
    ipcRenderer.invoke("dopest:rpc", method, params ?? {}),
  /** Operation discovery catalog. */
  ops: (): Promise<OpsCatalog> => ipcRenderer.invoke("dopest:ops"),
  /** Sidecar health. */
  health: (): Promise<HealthInfo> => ipcRenderer.invoke("dopest:health"),
  /** Read a local file (PNG frame, image, video) as a data URL. */
  readFileAsDataURL: (filePath: string): Promise<string> =>
    ipcRenderer.invoke("dopest:readFileAsDataURL", filePath),
};

export type DopestBridge = typeof api;

contextBridge.exposeInMainWorld("dopest", api);
