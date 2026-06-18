// Result shapes for the sidecar operations the editor uses. Mirrors dopest_clip return
// dicts. Optional/loose where the Python side returns open-ended structures.

export interface ProjectSummary {
  project_id: string;
  source?: string;
  duration?: number;
  transcribed: boolean;
}

export interface ListProjectsResult {
  projects: ProjectSummary[];
  count: number;
}

export interface ProjectDetail {
  project_id: string;
  source?: string;
  duration?: number;
  fps?: number;
  width?: number;
  height?: number;
  has_audio?: boolean;
  transcribed: boolean;
  transcript_txt?: string;
  edls?: string[];
  error?: string;
  [k: string]: unknown;
}

export interface CreateProjectResult {
  project_id?: string;
  duration?: number;
  width?: number;
  height?: number;
  fps?: number;
  next?: string;
  error?: string;
  [k: string]: unknown;
}

export interface TranscribeResult {
  project_id?: string;
  language?: string;
  word_count?: number;
  silence_count?: number;
  untimed_tokens_dropped?: number;
  transcript_txt?: string;
  transcript_json?: string;
  next?: string;
  error?: string;
}

export interface TranscriptWord {
  i: number;
  w: string;
  start: number;
  end: number;
}

export interface TranscriptJson {
  project_id: string;
  words: TranscriptWord[];
  count: number;
}

export interface RenderResult {
  project_id?: string;
  edl_id?: string;
  render?: string;
  outputs?: { cut: string; styled?: Record<string, string> };
  segments?: number;
  expected_duration?: number;
  warnings?: string[];
  cleanup?: unknown;
  reframe_mode?: string;
  captions?: string | null;
  next?: string;
  error?: string;
}

export interface VerifyResult {
  match_ratio?: number;
  diffs?: unknown;
  project_id?: string;
  edl_id?: string;
  error?: string;
  [k: string]: unknown;
}

export interface FrameResult {
  frame: string;
  at: number;
  source?: string;
  width: number;
  height: number;
  note?: string;
}

export interface PreviewReframeResult {
  frame: string;
  edl_id: string;
  at: number;
  aspect: string;
  crop_rect: unknown;
  censored?: boolean;
  note?: string;
}

export interface ThumbnailResult {
  project_id: string;
  edl_id: string;
  thumbnail: string;
  aspect: string;
  error?: string;
}

export interface CaptionPresetsResult {
  presets: string[];
  default: string;
}

export interface ReframeModesResult {
  modes: string[];
  aspects: string[];
  see_before_framing?: string;
  timeline?: string;
}

export interface ProviderInfo {
  configured: boolean;
  detail?: string;
  active: boolean;
}

export interface CapabilityProviders {
  active?: string;
  providers: Record<string, ProviderInfo>;
}

export type ListProvidersResult = Record<string, CapabilityProviders>;

export interface SetProviderResult {
  capability: string;
  active: string;
}
