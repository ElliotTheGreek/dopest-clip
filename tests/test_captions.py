"""Caption ASS generation — pure string building, no ffmpeg, no heavy deps."""

import pytest

from dopest_clip import captions


@pytest.fixture
def out_words():
    """Output-timeline word list (what edl.remap_to_output_timeline produces)."""
    words = ["hello", "world", "this", "is", "a", "test."]
    out = []
    t = 0.0
    for w in words:
        out.append({"w": w, "start": round(t, 3), "end": round(t + 0.4, 3)})
        t += 0.5
    return out


def test_presets_exist_and_listed():
    presets = captions.list_presets()
    assert set(presets) == {"karaoke-bold", "lower-third", "minimal-top"}
    for name in presets:
        assert name in captions.PRESETS


def test_fonts_dir_points_at_bundled_anton():
    # FONTS_DIR is re-exported from config; the bundled Anton font lives there.
    assert (captions.FONTS_DIR / "Anton-Regular.ttf").exists()


def test_build_ass_header_well_formed(out_words):
    ass = captions.build_ass(out_words, 1080, 1920, preset="karaoke-bold")
    assert "[Script Info]" in ass
    assert "ScriptType: v4.00+" in ass
    assert "PlayResX: 1080" in ass
    assert "PlayResY: 1920" in ass
    assert "[V4+ Styles]" in ass
    assert "Style: Main," in ass
    assert "Style: Title," in ass
    assert "[Events]" in ass
    assert "Format: Layer, Start, End, Style," in ass


def test_build_ass_emits_dialogue_lines(out_words):
    ass = captions.build_ass(out_words, 1080, 1920, preset="lower-third")
    dialogues = [ln for ln in ass.splitlines() if ln.startswith("Dialogue:")]
    assert dialogues, "expected at least one Dialogue line"
    # lower-third is not per-word, so each line is a single phrase event
    assert all(ln.startswith("Dialogue: 0,") for ln in dialogues)


def test_karaoke_is_per_word_with_highlight(out_words):
    ass = captions.build_ass(out_words, 1080, 1920, preset="karaoke-bold")
    dialogues = [ln for ln in ass.splitlines() if ln.startswith("Dialogue:")]
    # per-word: one event per word; uppercase; highlight override tags present
    assert len(dialogues) >= len(out_words)
    assert "\\fscx114" in ass  # highlight scale tag
    assert "HELLO" in ass  # uppercase applied


def test_minimal_top_is_phrase_lines(out_words):
    ass = captions.build_ass(out_words, 1080, 1920, preset="minimal-top")
    dialogues = [ln for ln in ass.splitlines() if ln.startswith("Dialogue:")]
    # not per-word -> fewer events than words; original case preserved
    assert len(dialogues) < len(out_words)
    assert "hello world" in ass


def test_title_card_emits_title_event(out_words):
    ass = captions.build_ass(out_words, 1080, 1920, preset="karaoke-bold",
                             title="My Hook", title_hold=2.5)
    title_lines = [ln for ln in ass.splitlines()
                   if ln.startswith("Dialogue:") and ",Title," in ln]
    assert len(title_lines) == 1
    assert "MY HOOK" in title_lines[0]  # titles are uppercased


def test_position_override_changes_alignment(out_words):
    top = captions.build_ass(out_words, 1080, 1920, preset="lower-third", position="top")
    main_style = [ln for ln in top.splitlines() if ln.startswith("Style: Main,")][0]
    # alignment is the field right before MarginL(60),MarginR(60),MarginV
    assert ",8,60,60," in main_style


def test_unknown_preset_raises(out_words):
    with pytest.raises(ValueError):
        captions.build_ass(out_words, 1080, 1920, preset="does-not-exist")
