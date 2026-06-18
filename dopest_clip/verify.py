"""QA: diff the intended (designed) transcript against a re-STT of the rendered clip.

Pure logic — no ffmpeg, no STT. ops.verify_clip re-transcribes the render (via the STT
backend), joins the words into text, and calls diff_transcripts() to score boundary drift
(dropped/duplicated/altered words) against the EDL's intended dialog.
"""

import difflib
import re


def _normalize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return text.split()


def diff_transcripts(intended_text: str, actual_text: str) -> dict:
    """Return {match_ratio, intended_word_count, actual_word_count, diff_count, diffs,
    actual_transcript}. diffs lists each non-equal opcode span (replace|delete|insert)."""
    a = _normalize(intended_text)
    b = _normalize(actual_text)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    ratio = sm.ratio()

    diffs = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        diffs.append({
            "op": tag,  # replace | delete | insert
            "intended": " ".join(a[i1:i2]),
            "actual": " ".join(b[j1:j2]),
            "intended_word_range": [i1, i2],
        })

    return {
        "match_ratio": round(ratio, 4),
        "intended_word_count": len(a),
        "actual_word_count": len(b),
        "diff_count": len(diffs),
        "diffs": diffs,
        "actual_transcript": actual_text,
    }
