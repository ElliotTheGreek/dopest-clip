// Pure, side-effect-free EDL manipulation helpers.
//
// These never touch the network or DOM — they take an EDL object and return a NEW EDL
// object (immutably), so the renderer can treat EDL state functionally and these can be
// unit-tested with no Electron/Python. The shape matches dopest_clip/edl.py exactly:
// segments are an ORDERED list of {from_word, to_word, label} and reordering /
// reuse / non-contiguity are all first-class.

import type {
  Edl,
  EdlSegment,
  ReframeConfig,
  ReframeShot,
  CaptionsConfig,
  TitleCardConfig,
  CleanupConfig,
} from "./types";

let _counter = 0;

/** Stable-ish unique id helper for client-side defaults (edl ids, etc.). */
export function makeEdlId(prefix = "edl"): string {
  _counter += 1;
  return `${prefix}-${Date.now().toString(36)}-${_counter}`;
}

/** A fresh empty EDL. */
export function emptyEdl(edl_id?: string, title?: string): Edl {
  return {
    edl_id: edl_id ?? makeEdlId(),
    title: title ?? "Untitled clip",
    segments: [],
  };
}

function clampIndex(n: number): number {
  return Math.max(0, Math.floor(n));
}

/** Append a segment. from/to are normalized so from_word <= to_word. */
export function addSegment(edl: Edl, fromWord: number, toWord: number, label = "seg"): Edl {
  let a = clampIndex(fromWord);
  let b = clampIndex(toWord);
  if (a > b) [a, b] = [b, a];
  const seg: EdlSegment = { from_word: a, to_word: b, label };
  return { ...edl, segments: [...edl.segments, seg] };
}

/** Insert a segment at a specific position in the order. */
export function insertSegment(edl: Edl, index: number, segment: EdlSegment): Edl {
  const segs = [...edl.segments];
  const i = Math.max(0, Math.min(index, segs.length));
  segs.splice(i, 0, { ...segment });
  return { ...edl, segments: segs };
}

/** Remove the segment at `index`. No-op if out of range. */
export function removeSegment(edl: Edl, index: number): Edl {
  if (index < 0 || index >= edl.segments.length) return edl;
  const segs = edl.segments.filter((_, i) => i !== index);
  return { ...edl, segments: segs };
}

/** Update one field-set of the segment at `index`, normalizing the word range. */
export function updateSegment(edl: Edl, index: number, patch: Partial<EdlSegment>): Edl {
  if (index < 0 || index >= edl.segments.length) return edl;
  const segs = edl.segments.map((s, i) => {
    if (i !== index) return s;
    const merged = { ...s, ...patch };
    let a = clampIndex(merged.from_word);
    let b = clampIndex(merged.to_word);
    if (a > b) [a, b] = [b, a];
    return { ...merged, from_word: a, to_word: b };
  });
  return { ...edl, segments: segs };
}

/** Move the segment at `from` to position `to` (reorder; non-contiguous order is fine). */
export function moveSegment(edl: Edl, from: number, to: number): Edl {
  const segs = [...edl.segments];
  if (from < 0 || from >= segs.length) return edl;
  const clampedTo = Math.max(0, Math.min(to, segs.length - 1));
  if (from === clampedTo) return edl;
  const [item] = segs.splice(from, 1);
  segs.splice(clampedTo, 0, item);
  return { ...edl, segments: segs };
}

/** Reorder by an explicit permutation of current indices. Throws if not a permutation. */
export function reorder(edl: Edl, order: number[]): Edl {
  const n = edl.segments.length;
  if (order.length !== n) {
    throw new Error(`reorder: expected ${n} indices, got ${order.length}`);
  }
  const seen = new Set<number>();
  for (const i of order) {
    if (i < 0 || i >= n || seen.has(i)) {
      throw new Error(`reorder: invalid or duplicate index ${i}`);
    }
    seen.add(i);
  }
  return { ...edl, segments: order.map((i) => edl.segments[i]) };
}

/** Duplicate the segment at `index` (reuse is first-class), inserting the copy after it. */
export function duplicateSegment(edl: Edl, index: number): Edl {
  if (index < 0 || index >= edl.segments.length) return edl;
  const copy = { ...edl.segments[index] };
  return insertSegment(edl, index + 1, copy);
}

// --- EDL-level styling ----------------------------------------------------------------

/** Shallow-merge styling fields onto the EDL (reframe/captions/title_card/cleanup/etc.). */
export interface StylingPatch {
  title?: string;
  reframe?: ReframeConfig | undefined;
  captions?: CaptionsConfig | undefined;
  title_card?: TitleCardConfig | undefined;
  cleanup?: CleanupConfig | undefined;
  loudnorm?: boolean;
  export_aspects?: string[] | undefined;
}

export function applyStyling(edl: Edl, patch: StylingPatch): Edl {
  const next: Edl = { ...edl };
  if ("title" in patch && patch.title !== undefined) next.title = patch.title;
  if ("loudnorm" in patch && patch.loudnorm !== undefined) next.loudnorm = patch.loudnorm;
  if ("reframe" in patch) next.reframe = patch.reframe;
  if ("captions" in patch) next.captions = patch.captions;
  if ("title_card" in patch) next.title_card = patch.title_card;
  if ("cleanup" in patch) next.cleanup = patch.cleanup;
  if ("export_aspects" in patch) next.export_aspects = patch.export_aspects;
  return next;
}

// --- reframe keyframe / shot list (basic list editor backing) -------------------------

export function setReframeMode(edl: Edl, mode: string, aspect?: string): Edl {
  const reframe: ReframeConfig = { ...(edl.reframe ?? {}), mode };
  if (aspect !== undefined) reframe.aspect = aspect;
  return { ...edl, reframe };
}

export function addReframeShot(edl: Edl, shot: ReframeShot): Edl {
  const reframe: ReframeConfig = { ...(edl.reframe ?? {}) };
  const shots = [...(reframe.shots ?? []), { ...shot }];
  // keep shots sorted by start time so the timeline reads in order
  shots.sort((a, b) => a.start - b.start);
  reframe.shots = shots;
  return { ...edl, reframe };
}

export function updateReframeShot(edl: Edl, index: number, patch: Partial<ReframeShot>): Edl {
  if (!edl.reframe?.shots || index < 0 || index >= edl.reframe.shots.length) return edl;
  const shots = edl.reframe.shots.map((s, i) => (i === index ? { ...s, ...patch } : s));
  shots.sort((a, b) => a.start - b.start);
  return { ...edl, reframe: { ...edl.reframe, shots } };
}

export function removeReframeShot(edl: Edl, index: number): Edl {
  if (!edl.reframe?.shots || index < 0 || index >= edl.reframe.shots.length) return edl;
  const shots = edl.reframe.shots.filter((_, i) => i !== index);
  return { ...edl, reframe: { ...edl.reframe, shots } };
}

/** Strip the client-only/ephemeral nothing — return a clean EDL object for the sidecar.
 *  (Currently the model is identical; this is the single normalization point so the
 *  renderer always sends exactly what Python expects.) */
export function toWire(edl: Edl): Edl {
  const wire: Edl = {
    edl_id: edl.edl_id,
    segments: edl.segments.map((s) => ({
      from_word: s.from_word,
      to_word: s.to_word,
      label: s.label,
    })),
  };
  if (edl.title !== undefined) wire.title = edl.title;
  if (edl.cleanup) wire.cleanup = edl.cleanup;
  if (edl.reframe) wire.reframe = edl.reframe;
  if (edl.captions) wire.captions = edl.captions;
  if (edl.title_card) wire.title_card = edl.title_card;
  if (edl.loudnorm !== undefined) wire.loudnorm = edl.loudnorm;
  if (edl.export_aspects) wire.export_aspects = edl.export_aspects;
  if (edl.censor) wire.censor = edl.censor;
  return wire;
}
