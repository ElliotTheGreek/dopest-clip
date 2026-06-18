"""EDL resolution + cleanup + output-timeline remap — pure logic, no ffmpeg."""

import pytest

from dopest_clip import edl


def test_resolve_contiguous(synthetic_transcript):
    e = {"edl_id": "t", "segments": [{"from_word": 0, "to_word": 2, "label": "a"}]}
    r = edl.resolve_edl(e, synthetic_transcript)
    seg = r["segments"][0]
    assert seg["from_word"] == 0 and seg["to_word"] == 2
    assert seg["start_clean"] and seg["end_clean"]
    assert seg["start"] >= 0.0
    assert r["reconstructed_text"] == "hello world this"
    assert r["warnings"] == []


def test_resolve_non_contiguous_reorder(synthetic_transcript):
    # Reorder + reuse: tail first, then head. Non-contiguous is first-class.
    e = {"segments": [
        {"from_word": 4, "to_word": 5, "label": "tail"},
        {"from_word": 0, "to_word": 1, "label": "head"},
    ]}
    r = edl.resolve_edl(e, synthetic_transcript)
    assert [s["label"] for s in r["segments"]] == ["tail", "head"]
    assert r["reconstructed_text"] == "a test\n\nhello world"


def test_resolve_out_of_range_raises(synthetic_transcript):
    with pytest.raises(ValueError):
        edl.resolve_edl({"segments": [{"from_word": 0, "to_word": 99}]}, synthetic_transcript)


def test_resolve_inverted_range_raises(synthetic_transcript):
    with pytest.raises(ValueError):
        edl.resolve_edl({"segments": [{"from_word": 3, "to_word": 1}]}, synthetic_transcript)


def test_segment_times(synthetic_transcript):
    e = {"segments": [{"from_word": 0, "to_word": 0}, {"from_word": 2, "to_word": 2}]}
    r = edl.resolve_edl(e, synthetic_transcript)
    times = edl.segment_times(r)
    assert len(times) == 2
    assert all(end > start for start, end in times)


def test_cleanup_removes_fillers():
    transcript = {
        "words": [
            {"i": 0, "w": "so", "start": 0.0, "end": 0.3},
            {"i": 1, "w": "um", "start": 0.4, "end": 0.6},
            {"i": 2, "w": "yeah", "start": 0.7, "end": 1.0},
        ],
        "silences": [],
    }
    e = {"segments": [{"from_word": 0, "to_word": 2, "label": "s"}], "cleanup": {"remove_fillers": True}}
    new_edl, report = edl.apply_cleanup(e, transcript)
    dropped = {d["w"] for d in report["removed_fillers"]}
    assert "um" in dropped
    # "um" split the run, so we should have two segments: [so] and [yeah]
    assert report["result_segments"] == 2


def test_cleanup_disabled_is_noop(synthetic_transcript):
    e = {"segments": [{"from_word": 0, "to_word": 2}]}
    new_edl, report = edl.apply_cleanup(e, synthetic_transcript)
    assert report is None
    assert new_edl is e


def test_remap_to_output_timeline(synthetic_transcript):
    e = {"segments": [{"from_word": 0, "to_word": 1, "label": "a"}]}
    r = edl.resolve_edl(e, synthetic_transcript)
    words = edl.remap_to_output_timeline(r, synthetic_transcript)
    assert [w["w"] for w in words] == ["hello", "world"]
    # output timeline starts at 0
    assert words[0]["start"] >= 0.0
    assert words[1]["start"] >= words[0]["start"]
