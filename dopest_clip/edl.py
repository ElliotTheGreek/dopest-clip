"""EDL model + resolution.

An EDL (edit-decision list) is the design: an ordered list of segments, each a
word-index range into the transcript. Segments may appear in any order and be reused
— non-contiguous reordering is first-class. resolve_edl() turns word ranges into
silence-snapped source times and reconstructs the exact dialog the design produces
(the cheap, no-render iteration loop). This is the editor's data model: the Electron
timeline renders and mutates exactly this structure.
"""

import re

from . import config

# Conservative defaults — only sounds/phrases that are almost always filler.
# "like" is intentionally excluded (too often a real verb/preposition).
DEFAULT_FILLERS = ["um", "uh", "er", "ah", "hmm", "umm", "uhh", "mm", "you know", "i mean"]


def _norm(w: str) -> str:
    return re.sub(r"[^\w']", "", w.lower())


def _mk_segment(label: str, run: list[int]) -> dict:
    return {"from_word": run[0], "to_word": run[-1], "label": label}


def apply_cleanup(edl: dict, transcript: dict) -> tuple[dict, dict | None]:
    """Tighten an EDL: drop filler-word spans and split segments at internal pauses
    longer than max_pause (the pause is excluded because the split becomes a
    word-boundary cut). Returns (new_edl, report) or (edl, None) if no cleanup set."""
    cfg = edl.get("cleanup") or {}
    if not cfg or cfg.get("enabled") is False:
        return edl, None

    remove_fillers = cfg.get("remove_fillers", True)
    fillers = [f.lower() for f in cfg.get("filler_words", DEFAULT_FILLERS)]
    max_pause = float(cfg.get("max_pause", 1.0))
    unigrams = {f for f in fillers if " " not in f}
    bigrams = {f for f in fillers if f.count(" ") == 1}

    by_index = {w["i"]: w for w in transcript["words"]}
    new_segments: list[dict] = []
    report = {
        "orig_segments": len(edl.get("segments", [])),
        "removed_fillers": [],
        "split_pauses": [],
    }

    for seg in edl.get("segments", []):
        fw, tw = int(seg["from_word"]), int(seg["to_word"])
        label = seg.get("label", "seg")
        idxs = [i for i in range(fw, tw + 1) if i in by_index]

        drop: set[int] = set()
        if remove_fillers:
            for i in idxs:
                if _norm(by_index[i]["w"]) in unigrams:
                    drop.add(i)
            for a, b in zip(idxs, idxs[1:]):
                if f"{_norm(by_index[a]['w'])} {_norm(by_index[b]['w'])}" in bigrams:
                    drop.add(a)
                    drop.add(b)
        for i in sorted(drop):
            report["removed_fillers"].append({"i": i, "w": by_index[i]["w"]})

        run: list[int] = []
        prev: int | None = None
        for i in idxs:
            if i in drop:
                if run:
                    new_segments.append(_mk_segment(label, run))
                    run = []
                prev = None
                continue
            if prev is not None:
                gap = by_index[i]["start"] - by_index[prev]["end"]
                if gap > max_pause:
                    report["split_pauses"].append({"after_i": prev, "before_i": i, "gap": round(gap, 3)})
                    if run:
                        new_segments.append(_mk_segment(label, run))
                        run = []
            run.append(i)
            prev = i
        if run:
            new_segments.append(_mk_segment(label, run))

    new_edl = dict(edl)
    new_edl["segments"] = new_segments
    report["result_segments"] = len(new_segments)
    return new_edl, report


def remap_to_output_timeline(resolved: dict, transcript: dict) -> list[dict]:
    """Per-word timing on the OUTPUT (cut) timeline: each word shifted by the sum of
    prior segment durations. Drives captions and emphasis. Reframe never changes
    timing, so this holds regardless of reframe."""
    by_index = {w["i"]: w for w in transcript["words"]}
    out: list[dict] = []
    offset = 0.0
    for seg in resolved["segments"]:
        sstart, dur = seg["start"], seg["dur"]
        for i in range(seg["from_word"], seg["to_word"] + 1):
            w = by_index.get(i)
            if not w:
                continue
            o_start = offset + max(0.0, w["start"] - sstart)
            o_end = offset + min(dur, w["end"] - sstart)
            if o_end <= o_start:
                o_end = o_start + 0.04
            out.append({"w": w["w"], "start": round(o_start, 3), "end": round(o_end, 3)})
        offset += dur
    return out


def _build_lookups(transcript: dict):
    words = transcript["words"]
    by_index = {w["i"]: w for w in words}
    sil_ending = {round(s["end"], 2): s for s in transcript.get("silences", [])}
    sil_starting = {round(s["start"], 2): s for s in transcript.get("silences", [])}
    return words, by_index, sil_ending, sil_starting


def _snap_start(word, sil_ending):
    """Cut point for the start of a segment: just inside the silence before `word`."""
    gap = sil_ending.get(round(word["start"], 2))
    if gap is None:
        return word["start"], False, 0.0  # no clean boundary -> hard cut at word onset
    margin = min(config.SNAP_MARGIN, gap["dur"] / 2)
    return round(word["start"] - margin, 3), True, gap["dur"]


def _snap_end(word, sil_starting):
    """Cut point for the end of a segment: just inside the silence after `word`."""
    gap = sil_starting.get(round(word["end"], 2))
    if gap is None:
        return word["end"], False, 0.0
    margin = min(config.SNAP_MARGIN, gap["dur"] / 2)
    return round(word["end"] + margin, 3), True, gap["dur"]


def resolve_edl(edl: dict, transcript: dict) -> dict:
    words, by_index, sil_ending, sil_starting = _build_lookups(transcript)
    if not words:
        raise ValueError("transcript has no words")
    max_index = words[-1]["i"]

    segments_out = []
    warnings = []
    total = 0.0
    full_text_parts = []

    for n, seg in enumerate(edl.get("segments", [])):
        fw = int(seg["from_word"])
        tw = int(seg["to_word"])
        label = seg.get("label", f"seg{n}")
        if fw not in by_index or tw not in by_index:
            raise ValueError(f"segment {n} '{label}': word index out of range (0..{max_index})")
        if fw > tw:
            raise ValueError(f"segment {n} '{label}': from_word ({fw}) > to_word ({tw})")

        start, start_clean, start_gap = _snap_start(by_index[fw], sil_ending)
        end, end_clean, end_gap = _snap_end(by_index[tw], sil_starting)
        start = max(0.0, start)
        dur = round(end - start, 3)
        total += dur

        text = " ".join(by_index[i]["w"] for i in range(fw, tw + 1)).strip()
        full_text_parts.append(text)

        if not start_clean:
            warnings.append(
                f"segment {n} '{label}': start at word #{fw} has no silence before it "
                f"(hard cut at {start:.3f}s) — pick an earlier boundary word for a clean join"
            )
        if not end_clean:
            warnings.append(
                f"segment {n} '{label}': end at word #{tw} has no silence after it "
                f"(hard cut at {end:.3f}s) — pick a later boundary word for a clean join"
            )

        segments_out.append({
            "label": label,
            "from_word": fw,
            "to_word": tw,
            "start": start,
            "end": end,
            "dur": dur,
            "start_clean": start_clean,
            "end_clean": end_clean,
            "start_gap": round(start_gap, 3),
            "end_gap": round(end_gap, 3),
            "text": text,
        })

    return {
        "edl_id": edl.get("edl_id"),
        "title": edl.get("title"),
        "segments": segments_out,
        "total_duration": round(total, 3),
        "reconstructed_text": "\n\n".join(full_text_parts),
        "warnings": warnings,
    }


def segment_times(resolved: dict) -> list[tuple[float, float]]:
    return [(s["start"], s["end"]) for s in resolved["segments"]]
