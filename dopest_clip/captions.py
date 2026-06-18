"""Caption generation: word-timed ASS (Advanced SubStation Alpha) burned by libass.

Input is the output-timeline word list from edl.remap_to_output_timeline(). Presets:
- karaoke-bold : big centered Anton, word-by-word pop highlight (TikTok/Hormozi look)
- lower-third  : clean pop-on phrase lines near the bottom
- minimal-top  : small clean lines near the top, out of the subject's way
A title_card can be overlaid (bold headline held for the opening seconds).

The bundled Anton font ships at config.FONTS_DIR / "Anton-Regular.ttf"; FONTS_DIR is
re-exported here so the renderer can pass it as ffmpeg's fontsdir. Pure string building —
no ffmpeg, no heavy deps.
"""

from . import config

FONTS_DIR = config.FONTS_DIR

# ASS colours are &HAABBGGRR (alpha, blue, green, red); AA=00 is opaque.
_WHITE = "&H00FFFFFF&"
_BLACK = "&H00000000&"
_YELLOW = "&H0000FFFF&"  # R255 G255 B0

PRESETS = {
    "karaoke-bold": {
        "font": "Anton", "size_ratio": 0.072, "bold": 1, "uppercase": True,
        "alignment": 2, "margin_v_ratio": 0.20, "outline": 6, "shadow": 2,
        "per_word": True, "highlight": _YELLOW, "highlight_scale": 114,
        "words_per_line": 4,
    },
    "lower-third": {
        "font": "Arial", "size_ratio": 0.045, "bold": 1, "uppercase": False,
        "alignment": 2, "margin_v_ratio": 0.08, "outline": 3, "shadow": 1,
        "per_word": False, "highlight": _YELLOW, "highlight_scale": 100,
        "words_per_line": 7,
    },
    "minimal-top": {
        "font": "Arial", "size_ratio": 0.040, "bold": 0, "uppercase": False,
        "alignment": 8, "margin_v_ratio": 0.06, "outline": 2, "shadow": 0,
        "per_word": False, "highlight": _YELLOW, "highlight_scale": 100,
        "words_per_line": 7,
    },
}

_ALIGN_OVERRIDE = {"lower-center": 2, "center": 5, "top": 8, "bottom": 2}


def _ass_time(t: float) -> str:
    if t < 0:
        t = 0.0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def _group_lines(words: list[dict], per_line: int) -> list[list[dict]]:
    lines, cur = [], []
    for w in words:
        cur.append(w)
        ends_sentence = w["w"].strip().endswith((".", "?", "!"))
        if len(cur) >= per_line or ends_sentence:
            lines.append(cur)
            cur = []
    if cur:
        lines.append(cur)
    return lines


def _disp(text: str, upper: bool) -> str:
    return text.upper() if upper else text


def build_ass(
    words: list[dict],
    video_w: int,
    video_h: int,
    preset: str = "karaoke-bold",
    font: str | None = None,
    position: str | None = None,
    title: str | None = None,
    title_hold: float = 3.0,
    margin_v: int | None = None,
) -> str:
    if preset not in PRESETS:
        raise ValueError(f"unknown caption preset '{preset}' (have: {', '.join(PRESETS)})")
    p = PRESETS[preset]
    fontname = font or p["font"]
    size = max(12, int(video_h * p["size_ratio"]))
    align = _ALIGN_OVERRIDE.get(position or "", p["alignment"])
    # explicit margin_v wins (lets a custom layout place captions in a precise band,
    # e.g. the top zone of a screen-top / face-bottom vertical stack)
    margin_v = int(video_h * p["margin_v_ratio"]) if margin_v is None else int(margin_v)
    upper = p["uppercase"]

    title_size = max(16, int(video_h * 0.055))
    styles = [
        f"Style: Main,{fontname},{size},{_WHITE},{_YELLOW},{_BLACK},{_BLACK},"
        f"{p['bold']},0,0,0,100,100,0,0,1,{p['outline']},{p['shadow']},{align},60,60,{margin_v},1",
        f"Style: Title,Anton,{title_size},{_WHITE},{_YELLOW},{_BLACK},{_BLACK},"
        f"1,0,0,0,100,100,0,0,1,5,2,8,60,60,{int(video_h*0.06)},1",
    ]

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {video_w}\nPlayResY: {video_h}\n"
        "WrapStyle: 2\nScaledBorderAndShadow: yes\nYCbCr Matrix: TV.601\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        + "\n".join(styles) + "\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    events: list[str] = []
    if title:
        events.append(
            f"Dialogue: 0,{_ass_time(0)},{_ass_time(title_hold)},Title,,0,0,0,,"
            f"{_disp(title, True)}"
        )

    lines = _group_lines(words, p["words_per_line"])
    hl, hl_scale = p["highlight"], p["highlight_scale"]
    for line in lines:
        line_start = line[0]["start"]
        line_end = line[-1]["end"]
        if not p["per_word"]:
            text = _disp(" ".join(w["w"] for w in line), upper)
            events.append(
                f"Dialogue: 0,{_ass_time(line_start)},{_ass_time(line_end)},Main,,0,0,0,,{text}"
            )
            continue
        # per-word: one event per word holding the whole line, active word emphasized
        for k, w in enumerate(line):
            start = w["start"]
            end = line[k + 1]["start"] if k + 1 < len(line) else w["end"]
            if end <= start:
                end = start + 0.05
            parts = []
            for j, wj in enumerate(line):
                token = _disp(wj["w"], upper)
                if j == k:
                    parts.append(f"{{\\c{hl}\\fscx{hl_scale}\\fscy{hl_scale}}}{token}{{\\c{_WHITE}\\fscx100\\fscy100}}")
                else:
                    parts.append(token)
            text = " ".join(parts)
            events.append(
                f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Main,,0,0,0,,{text}"
            )

    return header + "\n".join(events) + "\n"


def list_presets() -> list[str]:
    return list(PRESETS.keys())
