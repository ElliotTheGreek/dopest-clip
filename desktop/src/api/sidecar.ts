// Typed client over the `window.dopest` bridge.
//
// Every method calls one sidecar operation by name and UNWRAPS the {ok, result|error}
// envelope: on `ok:false` it throws an Error with the error string; on `ok:true` it
// returns `result`. Note that several Python ops also return a `{error: "..."}` dict
// *inside* a successful envelope (op-level soft errors) — those are returned as-is so the
// UI can surface them; callers check `.error` where relevant.

import type { Edl } from "../edl/types";
import { toWire } from "../edl/edl";
import type { ValidateResult } from "../edl/types";
import type {
  ListProjectsResult,
  ProjectDetail,
  CreateProjectResult,
  TranscribeResult,
  TranscriptJson,
  RenderResult,
  VerifyResult,
  FrameResult,
  PreviewReframeResult,
  ThumbnailResult,
  CaptionPresetsResult,
  ReframeModesResult,
  ListProvidersResult,
  SetProviderResult,
} from "./types";

// --- the window.dopest bridge typing (provided by preload) -----------------------------

export interface RpcEnvelope<T = unknown> {
  ok: boolean;
  result?: T;
  error?: string;
}

export interface DopestWindow {
  rpc(method: string, params?: Record<string, unknown>): Promise<RpcEnvelope>;
  ops(): Promise<{ groups: Record<string, string[]>; operations: string[] }>;
  health(): Promise<{ ok: boolean; name: string; version: string }>;
  readFileAsDataURL(filePath: string): Promise<string>;
}

declare global {
  interface Window {
    dopest: DopestWindow;
  }
}

function bridge(): DopestWindow {
  if (typeof window === "undefined" || !window.dopest) {
    throw new Error("window.dopest bridge is not available (not running inside Electron preload)");
  }
  return window.dopest;
}

/** Core call: invoke an operation and unwrap, throwing on transport/envelope failure. */
export async function call<T = unknown>(method: string, params?: Record<string, unknown>): Promise<T> {
  const env = (await bridge().rpc(method, params ?? {})) as RpcEnvelope<T>;
  if (!env || env.ok !== true) {
    throw new Error(env?.error || `rpc '${method}' failed`);
  }
  return env.result as T;
}

// --- discovery / health ---------------------------------------------------------------

export const ops = () => bridge().ops();
export const health = () => bridge().health();
export const readFileAsDataURL = (p: string) => bridge().readFileAsDataURL(p);

/** A streamable URL for a local media file, served by the main process `dopest-file://`
 *  protocol. Use for <video>/<audio> src (supports range requests / seeking) instead of
 *  loading the whole file as a data URL. */
export const fileUrl = (p: string) => `dopest-file://local/${encodeURIComponent(p)}`;

// --- editing --------------------------------------------------------------------------

export const listProjects = () => call<ListProjectsResult>("list_projects");
export const getProject = (project_id: string) => call<ProjectDetail>("get_project", { project_id });
export const createProject = (video_path: string, project_id?: string) =>
  call<CreateProjectResult>("create_project", project_id ? { video_path, project_id } : { video_path });

export const transcribe = (
  project_id: string,
  opts?: { model?: string; language?: string; backend?: string },
) => call<TranscribeResult>("transcribe", { project_id, ...(opts ?? {}) });

export const getTranscript = (
  project_id: string,
  opts?: { from_word?: number; to_word?: number; from_time?: number; to_time?: number; fmt?: "text" | "json" },
) => call<TranscriptJson | { project_id: string; count: number; text: string }>("get_transcript", { project_id, ...(opts ?? {}) });

export const getTranscriptJson = (project_id: string) =>
  call<TranscriptJson>("get_transcript", { project_id, fmt: "json" });

export const validateEdl = (project_id: string, edl: Edl) =>
  call<ValidateResult>("validate_edl", { project_id, edl_obj: toWire(edl) });

export const render = (
  project_id: string,
  edlOrId: Edl | string,
  opts?: { aspect?: string; crossfade_ms?: number },
) =>
  call<RenderResult>("render", {
    project_id,
    edl_obj_or_id: typeof edlOrId === "string" ? edlOrId : toWire(edlOrId),
    ...(opts ?? {}),
  });

export const verifyClip = (project_id: string, edl_id: string, backend?: string) =>
  call<VerifyResult>("verify_clip", backend ? { project_id, edl_id, backend } : { project_id, edl_id });

export const grabFrame = (project_id: string, at: number, source = "source", grid = true) =>
  call<FrameResult>("grab_frame", { project_id, at, source, grid });

export const previewReframe = (project_id: string, edl_id: string, at: number, aspect?: string) =>
  call<PreviewReframeResult>("preview_reframe", aspect ? { project_id, edl_id, at, aspect } : { project_id, edl_id, at });

export const extractThumbnail = (
  project_id: string,
  edl_id: string,
  opts?: { at_time?: number; text?: string; aspect?: string; aspect_mode?: string },
) => call<ThumbnailResult>("extract_thumbnail", { project_id, edl_id, ...(opts ?? {}) });

export const mixCamera = (
  project_id: string,
  edl_id: string,
  camera_path: string,
  opts?: { keyframes?: unknown[]; remove_background?: boolean; output_path?: string; rematte?: boolean },
) => call<Record<string, unknown>>("mix_camera", { project_id, edl_id, camera_path, ...(opts ?? {}) });

export const getCutTranscript = (project_id: string, edl_id: string) =>
  call<{ project_id: string; edl_id: string; word_count: number; cut_transcript_txt: string; text: string }>(
    "get_cut_transcript", { project_id, edl_id });

export const makeShort = (
  project_id: string,
  edl_id: string,
  from_word: number,
  to_word: number,
  hook_title: string,
  opts?: { screen_keyframes?: unknown[]; caption_preset?: string; output_path?: string },
) => call<{ output: string; size: number[]; duration: number; hook: string }>(
  "make_short", { project_id, edl_id, from_word, to_word, hook_title, ...(opts ?? {}) });

export const listGraphics = () => call<Record<string, unknown>>("list_graphics");

export const listCaptionPresets = () => call<CaptionPresetsResult>("list_caption_presets");
export const listReframeModes = () => call<ReframeModesResult>("list_reframe_modes");
export const suggestClips = (project_id: string, n = 3, instructions?: string) =>
  call("suggest_clips", instructions ? { project_id, n, instructions } : { project_id, n });

// --- providers ------------------------------------------------------------------------

export const listProviders = () => call<ListProvidersResult>("list_providers");
export const setProvider = (capability: string, provider_name: string) =>
  call<SetProviderResult>("set_provider", { capability, provider_name });
export const validateProvider = (capability: string) =>
  call<Record<string, unknown>>("validate_provider", { capability });
