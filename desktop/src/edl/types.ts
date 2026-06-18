// EDL data model — mirrors the dopest_clip Python EDL model exactly.
// See dopest_clip/edl.py and dopest_clip/ops.py. The timeline edits this structure.

export interface EdlSegment {
  from_word: number;
  to_word: number;
  label: string;
}

export interface CleanupConfig {
  enabled?: boolean;
  remove_fillers?: boolean;
  filler_words?: string[];
  max_pause?: number;
}

/** A single keyframe / shot in the reframe timeline (OUTPUT/clip seconds). */
export interface ReframeShot {
  start: number;
  mode: string;
  zoom?: number;
  x?: number;
  y?: number;
  crop?: unknown;
}

export interface ReframeConfig {
  mode?: string;
  aspect?: string;
  transition_s?: number;
  zoom?: number | { factor?: number };
  x?: number;
  y?: number;
  shots?: ReframeShot[];
}

export interface CaptionsConfig {
  enabled?: boolean;
  preset?: string;
  font?: string;
  position?: string;
}

export interface TitleCardConfig {
  text?: string;
  hold_s?: number;
}

export interface Edl {
  edl_id: string;
  title?: string;
  segments: EdlSegment[];
  cleanup?: CleanupConfig;
  reframe?: ReframeConfig;
  captions?: CaptionsConfig;
  title_card?: TitleCardConfig;
  loudnorm?: boolean;
  export_aspects?: string[];
  censor?: unknown[];
}

// --- validate_edl result shape (from resolve_edl) -------------------------------------

export interface ResolvedSegment {
  label: string;
  from_word: number;
  to_word: number;
  start: number;
  end: number;
  dur: number;
  start_clean: boolean;
  end_clean: boolean;
  start_gap: number;
  end_gap: number;
  text: string;
}

export interface CleanupReport {
  orig_segments: number;
  removed_fillers: { i: number; w: string }[];
  split_pauses: { after_i: number; before_i: number; gap: number }[];
  result_segments: number;
}

export interface ValidateResult {
  edl_id: string | null;
  title?: string | null;
  segments: ResolvedSegment[];
  total_duration: number;
  reconstructed_text: string;
  warnings: string[];
  cleanup?: CleanupReport;
}
