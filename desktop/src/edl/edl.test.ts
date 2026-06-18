import { describe, it, expect } from "vitest";
import {
  emptyEdl,
  addSegment,
  insertSegment,
  removeSegment,
  updateSegment,
  moveSegment,
  reorder,
  duplicateSegment,
  applyStyling,
  setReframeMode,
  addReframeShot,
  updateReframeShot,
  removeReframeShot,
  toWire,
} from "./edl";
import type { Edl } from "./types";

function seg(from: number, to: number, label = "seg") {
  return { from_word: from, to_word: to, label };
}

describe("emptyEdl", () => {
  it("creates an edl with an id and no segments", () => {
    const e = emptyEdl("clip-1", "My Clip");
    expect(e.edl_id).toBe("clip-1");
    expect(e.title).toBe("My Clip");
    expect(e.segments).toEqual([]);
  });
  it("generates an id when none given", () => {
    const e = emptyEdl();
    expect(e.edl_id).toMatch(/^edl-/);
  });
});

describe("addSegment", () => {
  it("appends a segment", () => {
    const e = addSegment(emptyEdl("c"), 0, 5, "hook");
    expect(e.segments).toEqual([seg(0, 5, "hook")]);
  });
  it("normalizes reversed ranges so from<=to", () => {
    const e = addSegment(emptyEdl("c"), 9, 3, "x");
    expect(e.segments[0]).toEqual(seg(3, 9, "x"));
  });
  it("floors and clamps negative indices", () => {
    const e = addSegment(emptyEdl("c"), -4, 2.9, "x");
    expect(e.segments[0]).toEqual(seg(0, 2, "x"));
  });
  it("does not mutate the input", () => {
    const base = emptyEdl("c");
    addSegment(base, 0, 1);
    expect(base.segments).toEqual([]);
  });
});

describe("insertSegment", () => {
  it("inserts at the given index", () => {
    let e = addSegment(addSegment(emptyEdl("c"), 0, 1, "a"), 10, 11, "c");
    e = insertSegment(e, 1, seg(5, 6, "b"));
    expect(e.segments.map((s) => s.label)).toEqual(["a", "b", "c"]);
  });
  it("clamps out-of-range index to the end", () => {
    let e = addSegment(emptyEdl("c"), 0, 1, "a");
    e = insertSegment(e, 99, seg(5, 6, "b"));
    expect(e.segments.map((s) => s.label)).toEqual(["a", "b"]);
  });
});

describe("removeSegment", () => {
  it("removes the segment at index", () => {
    let e = addSegment(addSegment(addSegment(emptyEdl("c"), 0, 1, "a"), 2, 3, "b"), 4, 5, "c");
    e = removeSegment(e, 1);
    expect(e.segments.map((s) => s.label)).toEqual(["a", "c"]);
  });
  it("is a no-op for out-of-range", () => {
    const e = addSegment(emptyEdl("c"), 0, 1, "a");
    expect(removeSegment(e, 7).segments).toHaveLength(1);
    expect(removeSegment(e, -1).segments).toHaveLength(1);
  });
});

describe("updateSegment", () => {
  it("patches label and normalizes the word range", () => {
    let e = addSegment(emptyEdl("c"), 0, 5, "a");
    e = updateSegment(e, 0, { label: "renamed", from_word: 20, to_word: 10 });
    expect(e.segments[0]).toEqual(seg(10, 20, "renamed"));
  });
  it("is a no-op for out-of-range index", () => {
    const e = addSegment(emptyEdl("c"), 0, 5, "a");
    expect(updateSegment(e, 5, { label: "x" })).toBe(e);
  });
});

describe("moveSegment / reorder (non-contiguous reordering)", () => {
  function abc(): Edl {
    return { edl_id: "c", segments: [seg(0, 1, "a"), seg(2, 3, "b"), seg(4, 5, "c")] };
  }
  it("moves a segment forward", () => {
    const e = moveSegment(abc(), 0, 2);
    expect(e.segments.map((s) => s.label)).toEqual(["b", "c", "a"]);
  });
  it("moves a segment backward", () => {
    const e = moveSegment(abc(), 2, 0);
    expect(e.segments.map((s) => s.label)).toEqual(["c", "a", "b"]);
  });
  it("clamps the destination", () => {
    const e = moveSegment(abc(), 0, 99);
    expect(e.segments.map((s) => s.label)).toEqual(["b", "c", "a"]);
  });
  it("no-ops when from===to", () => {
    const base = abc();
    expect(moveSegment(base, 1, 1)).toBe(base);
  });
  it("reorder applies an explicit permutation", () => {
    const e = reorder(abc(), [2, 0, 1]);
    expect(e.segments.map((s) => s.label)).toEqual(["c", "a", "b"]);
  });
  it("reorder rejects wrong-length order", () => {
    expect(() => reorder(abc(), [0, 1])).toThrow();
  });
  it("reorder rejects duplicate indices", () => {
    expect(() => reorder(abc(), [0, 0, 1])).toThrow();
  });
});

describe("duplicateSegment (reuse is first-class)", () => {
  it("inserts a copy right after the original", () => {
    let e = addSegment(addSegment(emptyEdl("c"), 0, 1, "a"), 2, 3, "b");
    e = duplicateSegment(e, 0);
    expect(e.segments.map((s) => s.label)).toEqual(["a", "a", "b"]);
    expect(e.segments[0]).not.toBe(e.segments[1]); // distinct objects
  });
});

describe("applyStyling", () => {
  it("merges styling fields", () => {
    let e = emptyEdl("c");
    e = applyStyling(e, {
      title: "T",
      loudnorm: true,
      captions: { enabled: true, preset: "karaoke-bold" },
      reframe: { mode: "track", aspect: "9:16" },
    });
    expect(e.title).toBe("T");
    expect(e.loudnorm).toBe(true);
    expect(e.captions).toEqual({ enabled: true, preset: "karaoke-bold" });
    expect(e.reframe).toEqual({ mode: "track", aspect: "9:16" });
  });
  it("can clear a field by passing undefined", () => {
    let e = applyStyling(emptyEdl("c"), { captions: { enabled: true } });
    e = applyStyling(e, { captions: undefined });
    expect(e.captions).toBeUndefined();
  });
});

describe("reframe shots (keyframe list)", () => {
  it("setReframeMode sets mode and aspect", () => {
    const e = setReframeMode(emptyEdl("c"), "track", "9:16");
    expect(e.reframe).toEqual({ mode: "track", aspect: "9:16" });
  });
  it("addReframeShot keeps shots sorted by start", () => {
    let e = emptyEdl("c");
    e = addReframeShot(e, { start: 5, mode: "track" });
    e = addReframeShot(e, { start: 0, mode: "full" });
    expect(e.reframe?.shots?.map((s) => s.start)).toEqual([0, 5]);
  });
  it("updateReframeShot patches and re-sorts", () => {
    let e = addReframeShot(addReframeShot(emptyEdl("c"), { start: 0, mode: "full" }), { start: 5, mode: "track" });
    e = updateReframeShot(e, 0, { start: 9 });
    expect(e.reframe?.shots?.map((s) => s.start)).toEqual([5, 9]);
  });
  it("removeReframeShot removes by index", () => {
    let e = addReframeShot(addReframeShot(emptyEdl("c"), { start: 0, mode: "full" }), { start: 5, mode: "track" });
    e = removeReframeShot(e, 0);
    expect(e.reframe?.shots).toHaveLength(1);
    expect(e.reframe?.shots?.[0].start).toBe(5);
  });
});

describe("toWire", () => {
  it("emits only the model fields the sidecar expects", () => {
    let e = emptyEdl("c", "Title");
    e = addSegment(e, 0, 5, "hook");
    e = applyStyling(e, { loudnorm: true });
    const w = toWire(e);
    expect(w).toEqual({
      edl_id: "c",
      title: "Title",
      segments: [seg(0, 5, "hook")],
      loudnorm: true,
    });
  });
  it("omits empty optional fields", () => {
    const w = toWire(emptyEdl("c", "T"));
    expect(Object.keys(w).sort()).toEqual(["edl_id", "segments", "title"]);
  });
});
