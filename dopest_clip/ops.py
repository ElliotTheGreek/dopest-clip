"""Orchestration layer: plain functions that tie the editing core together.

This is the editing half of dopest-clip, ported from the short-form-editor server but as
ordinary Python (the MCP tool layer wraps these later, in Phase 7). Every function returns
a JSON-able dict. The loop the agent runs is:

    create_project -> transcribe -> (Read transcript.txt) -> design EDL ->
    validate_edl -> render -> verify_clip

The agent does the creative reasoning (reading the transcript, designing EDLs); these
functions give it accurate STT, a cheap text-space validation loop, a renderer, and an
STT-based QA gate.

Heavy deps stay lazy: reframe (cv2/ultralytics) and the STT backends only import their
ML/cloud deps when their work actually runs, so importing this module is cheap.
"""

from pathlib import Path

from . import captions, config, edl, media, project, reframe, verify
from .stt import get_backend


# --- transcript.txt rendering ---------------------------------------------------------

def _fmt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h:
        return f"{h:02d}:{m:02d}:{s:06.3f}"
    return f"{m:02d}:{s:06.3f}"


def _write_transcript_txt(transcript: dict, path: Path) -> None:
    """Readable transcript with line-start timestamps and word indices, line-broken on
    longer silences so the agent can locate word ranges to cut on."""
    words = transcript["words"]
    silence_starts = {round(s["start"], 2) for s in transcript.get("silences", []) if s["dur"] >= 0.6}
    lines = []
    cur: list[str] = []
    line_start_word = None
    for w in words:
        if not cur:
            line_start_word = w
        cur.append(w["w"])
        long_pause_after = round(w["end"], 2) in silence_starts
        if long_pause_after or len(cur) >= 22:
            lines.append(f"[{_fmt_ts(line_start_word['start'])}] (#{line_start_word['i']}) " + " ".join(cur))
            cur = []
    if cur and line_start_word is not None:
        lines.append(f"[{_fmt_ts(line_start_word['start'])}] (#{line_start_word['i']}) " + " ".join(cur))
    header = (
        f"# Transcript for project '{transcript['project_id']}'\n"
        f"# source: {transcript['source']}\n"
        f"# duration: {transcript['duration']}s | words: {len(words)} | language: {transcript.get('language')}\n"
        f"# Format: [timestamp] (#word_index) text.  Design EDLs by referencing word indices.\n\n"
    )
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")


# --- project lifecycle ----------------------------------------------------------------

def create_project(video_path: str, project_id: str | None = None) -> dict:
    """Register a source video: probe metadata and extract 16kHz mono audio. Does not
    transcribe. Returns the project_id to use for every later call."""
    src = Path(video_path)
    if not src.exists():
        return {"error": f"video not found: {video_path}"}
    info = media.probe(str(src))
    if not info["has_audio"]:
        return {"error": "source has no audio stream; this tool is audio/dialog-driven"}

    pid = project_id or project.new_project_id(video_path)
    project.ensure_project(pid)
    media.extract_audio(str(src), project.audio_path(pid))

    meta = {"project_id": pid, "source": str(src), **info}
    project.write_meta(pid, meta)
    return {"project_id": pid, **info, "next": "call transcribe(project_id)"}


def transcribe(
    project_id: str,
    model: str | None = None,
    language: str | None = None,
    backend: str | None = None,
) -> dict:
    """Transcribe the project audio with word-level timestamps + a silence map. Writes
    transcript.json and transcript.txt. Returns stats only — then READ the transcript.txt
    file to design clips. backend defaults to config.STT_BACKEND."""
    if not project.project_exists(project_id):
        return {"error": f"unknown project '{project_id}'"}
    meta = project.read_meta(project_id)
    stt = get_backend(backend)
    result = stt.transcribe(project.audio_path(project_id), model=model, language=language)

    transcript = {
        "project_id": project_id,
        "source": meta["source"],
        "duration": meta["duration"],
        "fps": meta["fps"],
        "width": meta["width"],
        "height": meta["height"],
        "language": result["language"],
        "words": result["words"],
        "silences": result["silences"],
    }
    project.write_json(project.transcript_json_path(project_id), transcript)
    _write_transcript_txt(transcript, project.transcript_txt_path(project_id))

    return {
        "project_id": project_id,
        "language": result["language"],
        "word_count": len(result["words"]),
        "silence_count": len(result["silences"]),
        "untimed_tokens_dropped": result.get("untimed_tokens_dropped", 0),
        "transcript_txt": str(project.transcript_txt_path(project_id)),
        "transcript_json": str(project.transcript_json_path(project_id)),
        "next": "Read the transcript_txt file, then design EDLs and call validate_edl",
    }


def get_transcript(
    project_id: str,
    from_word: int | None = None,
    to_word: int | None = None,
    from_time: float | None = None,
    to_time: float | None = None,
    fmt: str = "text",
) -> dict:
    """Return a slice of the transcript by word-index or time range (optional helper; you
    can also just Read the transcript.txt file). fmt = 'text' or 'json'."""
    if not project.transcript_json_path(project_id).exists():
        return {"error": f"no transcript for '{project_id}' — call transcribe first"}
    t = project.read_transcript(project_id)
    words = t["words"]

    def keep(w):
        if from_word is not None and w["i"] < from_word:
            return False
        if to_word is not None and w["i"] > to_word:
            return False
        if from_time is not None and w["end"] < from_time:
            return False
        if to_time is not None and w["start"] > to_time:
            return False
        return True

    sel = [w for w in words if keep(w)]
    if fmt == "json":
        return {"project_id": project_id, "words": sel, "count": len(sel)}
    return {
        "project_id": project_id,
        "count": len(sel),
        "text": " ".join(w["w"] for w in sel),
        "first_word": sel[0]["i"] if sel else None,
        "last_word": sel[-1]["i"] if sel else None,
    }


def list_projects() -> dict:
    """List all projects in the workspace."""
    out = []
    for pid in project.list_project_ids():
        meta = project.read_meta(pid)
        out.append({
            "project_id": pid,
            "source": meta.get("source"),
            "duration": meta.get("duration"),
            "transcribed": project.transcript_json_path(pid).exists(),
        })
    return {"projects": out, "count": len(out)}


def get_project(project_id: str) -> dict:
    """Inspect one project's state: metadata, whether transcribed, and saved EDLs."""
    if not project.project_exists(project_id):
        return {"error": f"unknown project '{project_id}'"}
    meta = project.read_meta(project_id)
    return {
        **meta,
        "transcribed": project.transcript_json_path(project_id).exists(),
        "transcript_txt": str(project.transcript_txt_path(project_id)),
        "edls": project.list_edl_ids(project_id),
    }


# --- the cheap validation loop --------------------------------------------------------

def validate_edl(project_id: str, edl_obj: dict) -> dict:
    """Resolve an EDL (no rendering): apply cleanup, snap cuts to silence, check bounds,
    SAVE the EDL, and return the reconstructed dialog in designed order plus per-join
    warnings + a cleanup report. This is the cheap loop — iterate here until the story
    reads right, then render. edl_obj = {edl_id, title, segments:[{from_word,to_word,
    label}], cleanup?}."""
    if not project.transcript_json_path(project_id).exists():
        return {"error": f"no transcript for '{project_id}' — call transcribe first"}
    t = project.read_transcript(project_id)
    try:
        cleaned, cleanup_report = edl.apply_cleanup(edl_obj, t)
        result = edl.resolve_edl(cleaned, t)
    except (ValueError, KeyError) as e:
        return {"error": str(e)}

    saved_id = project.save_edl(project_id, edl_obj)
    if cleanup_report is not None:
        result["cleanup"] = cleanup_report
    result["edl_id"] = saved_id
    return result


# --- render ---------------------------------------------------------------------------

def _resolve_aspects(edl_obj: dict, aspect_param: str) -> list[str]:
    if edl_obj.get("export_aspects"):
        return list(edl_obj["export_aspects"])
    rf = edl_obj.get("reframe") or {}
    if rf.get("aspect"):
        return [rf["aspect"]]
    return [aspect_param]


def render(
    project_id: str,
    edl_obj_or_id,
    aspect: str | None = None,
    crossfade_ms: int | None = None,
) -> dict:
    """Render an EDL through the full pipeline: cleanup -> silence-snapped cut -> optional
    reframe (portrait, subject-tracked) -> optional burned captions + title + loudnorm ->
    multi-aspect fan-out.

    `edl_obj_or_id` is either the EDL dict (saved then rendered) or a saved edl_id string.

    Styling is read from the EDL (all optional, backward compatible): `cleanup`,
    `reframe`={mode/shots, aspect, zoom}, `captions`={enabled, preset, font, position},
    `title_card`={text, hold_s}, `loudnorm`, `export_aspects`, and `censor`=[{x,y,w,h,
    style?:'blur'|'pixelate'|'box', start?, end?}] (SOURCE-pixel redactions burned in before
    any crop). With no styling fields it behaves like a plain source-aspect cut. The clean
    cut is ALWAYS written to renders/<edl_id>.mp4 (stable audio for verify_clip); styled
    deliverables go to renders/<edl_id>__<aspect>.mp4."""
    if not project.transcript_json_path(project_id).exists():
        return {"error": f"no transcript for '{project_id}' — call transcribe first"}
    meta = project.read_meta(project_id)
    t = project.read_transcript(project_id)
    aspect = aspect or "source"

    if isinstance(edl_obj_or_id, str):
        p = project.edl_path(project_id, edl_obj_or_id)
        if not p.exists():
            return {"error": f"no saved EDL '{edl_obj_or_id}'"}
        edl_obj = project.read_json(p)
        eid = edl_obj.get("edl_id") or edl_obj_or_id
    elif isinstance(edl_obj_or_id, dict):
        edl_obj = edl_obj_or_id
        eid = edl_obj.get("edl_id")
        if not eid:
            return {"error": "edl_obj must include an 'edl_id'"}
    else:
        return {"error": "provide an edl_obj dict or a saved edl_id string"}

    edl_obj = dict(edl_obj)
    edl_obj["edl_id"] = eid

    try:
        cleaned, cleanup_report = edl.apply_cleanup(edl_obj, t)
        resolved = edl.resolve_edl(cleaned, t)
    except (ValueError, KeyError) as e:
        return {"error": str(e)}

    eid = project.save_edl(project_id, edl_obj)

    # 1) the clean cut (source aspect) — always written; stable audio for verify_clip
    cut = project.render_path(project_id, eid)
    cf = config.CROSSFADE_MS if crossfade_ms is None else crossfade_ms
    media.render(meta["source"], edl.segment_times(resolved), cut,
                 project.filtergraph_path(project_id, eid), crossfade_ms=cf)

    src_w, src_h = meta["width"], meta["height"]
    cap_cfg = edl_obj.get("captions") or {}
    captions_on = bool(cap_cfg.get("enabled"))
    title_cfg = edl_obj.get("title_card") or {}
    loudnorm = bool(edl_obj.get("loudnorm"))
    rf = edl_obj.get("reframe") or {}
    rf_mode = rf.get("mode", "none")
    censor = edl_obj.get("censor") or None
    aspects = _resolve_aspects(edl_obj, aspect)

    styling_on = captions_on or loudnorm or bool(title_cfg) or rf_mode not in ("none", None) \
        or bool(rf.get("shots")) or bool(censor) or aspects != ["source"]

    out_info = {"cut": str(cut)}
    if not styling_on:
        return {
            "project_id": project_id, "edl_id": eid, "render": str(cut),
            "outputs": out_info, "segments": len(resolved["segments"]),
            "expected_duration": resolved["total_duration"], "warnings": resolved["warnings"],
            "cleanup": cleanup_report,
            "next": f"call verify_clip(project_id='{project_id}', edl_id='{eid}')",
        }

    words_out = edl.remap_to_output_timeline(resolved, t)
    styled = {}
    for asp in aspects:
        tw, th = media.aspect_dims(asp, src_w, src_h)
        ass_file = None
        if captions_on or title_cfg.get("text"):
            ass_str = captions.build_ass(
                words_out, tw, th,
                preset=cap_cfg.get("preset", "karaoke-bold"),
                font=cap_cfg.get("font"),
                position=cap_cfg.get("position"),
                title=title_cfg.get("text"),
                title_hold=float(title_cfg.get("hold_s", 3.0)),
            )
            ass_file = project.ass_path(project_id, eid, asp)
            ass_file.write_text(ass_str, encoding="utf-8")

        out = project.styled_render_path(project_id, eid, asp)
        shots = rf.get("shots")
        # censor must be burned into source frames before any crop -> force the frame compositor
        uses_compositor = bool(shots) or bool(censor) or rf_mode in ("track", "zoom", "full", "focus", "center")
        if uses_compositor:
            zoomv = rf.get("zoom")
            default_zoom = zoomv.get("factor", 1.0) if isinstance(zoomv, dict) else (zoomv or 1.0)
            if shots:
                shot_list = shots
            elif rf_mode and rf_mode != "none":
                shot_list = [{"start": 0.0, "mode": rf_mode, "x": rf.get("x"),
                              "y": rf.get("y"), "zoom": rf.get("zoom"), "crop": rf.get("crop")}]
            else:  # censor-only (no reframe intent): show the whole frame, just redacted
                shot_list = [{"start": 0.0, "mode": "full"}]
            transition_s = float(rf.get("transition_s", config.REFRAME_TRANSITION_S))
            rects, fps, _sw, _sh = reframe.build_plan(cut, tw, th, shot_list, transition_s, default_zoom)
            reframe.apply_reframe(cut, rects, fps, tw, th, out, ass_file, loudnorm, captions.FONTS_DIR, censor=censor)
        else:
            mode = "pad" if rf_mode == "pad" else "crop"
            media.style_encode(cut, out, tw, th, src_w, src_h, aspect_mode=mode,
                               ass_path=ass_file, loudnorm=loudnorm, fonts_dir=captions.FONTS_DIR)
        styled[asp] = str(out)

    out_info["styled"] = styled
    return {
        "project_id": project_id, "edl_id": eid,
        "render": styled[aspects[0]], "outputs": out_info,
        "segments": len(resolved["segments"]), "expected_duration": resolved["total_duration"],
        "warnings": resolved["warnings"], "cleanup": cleanup_report,
        "reframe_mode": rf_mode, "captions": cap_cfg.get("preset") if captions_on else None,
        "next": f"call verify_clip(project_id='{project_id}', edl_id='{eid}')",
    }


# --- caption burn over a finished composite -------------------------------------------

def burn_captions(project_id: str, edl_id: str, video_path: str = "",
                  position: str = "bottom", top_window: list | None = None,
                  montage_at: float | None = None, montage_dur: float = 0.0,
                  preset: str = "karaoke-bold", output_path: str = "") -> dict:
    """Burn word-highlight captions over a finished composite (the camera-mixed master). Words
    come from the EDL's OUTPUT timeline (same source the styled render uses). `position` =
    'bottom'|'top'; `top_window` = [start, end] cut-seconds that flips captions to the TOP for
    that span (e.g. a dictation demo), back to `position` outside it; `montage_at`/`montage_dur`
    shift every word after an inserted pause montage so captions stay synced. Default video is
    the project's <edl>_mixed.mp4; writes <edl>_captioned.mp4."""
    if not project.transcript_json_path(project_id).exists():
        return {"error": f"no transcript for '{project_id}' — call transcribe first"}
    edl_p = project.edl_path(project_id, edl_id)
    if not edl_p.exists():
        return {"error": f"no saved EDL '{edl_id}'"}
    t = project.read_transcript(project_id)
    cleaned, _ = edl.apply_cleanup(project.read_json(edl_p), t)
    resolved = edl.resolve_edl(cleaned, t)
    words = edl.remap_to_output_timeline(resolved, t)
    tw = [float(top_window[0]), float(top_window[1])] if top_window else None
    if montage_at is not None and montage_dur:
        cut, off = float(montage_at), float(montage_dur)
        for w in words:
            if w["start"] >= cut:
                w["start"] += off
                w["end"] += off
        if tw and tw[0] >= cut:
            tw = [tw[0] + off, tw[1] + off]
    meta = project.read_meta(project_id)
    slug = project.slugify(edl_id)
    vid = Path(video_path) if video_path else project.render_path(project_id, edl_id).with_name(f"{slug}_mixed.mp4")
    if not vid.exists():
        return {"error": f"video not found: {vid} — run mix_camera first"}
    ass = captions.build_ass(words, meta["width"], meta["height"], preset=preset,
                             position=position, top_window=(tw[0], tw[1]) if tw else None)
    ass_path = vid.with_name(f"{slug}_captions.ass")
    ass_path.write_text(ass, encoding="utf-8")
    out = Path(output_path) if output_path else vid.with_name(f"{slug}_captioned.mp4")
    sub = f"subtitles=filename='{media.escape_filter_path(ass_path)}'"
    sub += f":fontsdir='{media.escape_filter_path(captions.FONTS_DIR)}'"
    media.run_ff([config.FFMPEG, "-y", "-loglevel", "error", "-i", str(vid), "-vf", sub,
                  "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                  "-c:a", "copy", str(out)])
    return {"project_id": project_id, "edl_id": edl_id, "output": str(out),
            "captioned_from": str(vid), "preset": preset, "position": position,
            "top_window": tw, "word_count": len(words)}


# --- QA -------------------------------------------------------------------------------

def verify_clip(project_id: str, edl_id: str, backend: str | None = None) -> dict:
    """QA a rendered clip: re-STT it and diff against the EDL's intended dialog. Returns a
    match ratio and any boundary drift (dropped/duplicated/altered words). The re-STT uses
    the same backend contract as transcribe — words are joined into text for the diff."""
    if not project.transcript_json_path(project_id).exists():
        return {"error": f"no transcript for '{project_id}'"}
    edl_p = project.edl_path(project_id, edl_id)
    render_p = project.render_path(project_id, edl_id)
    if not edl_p.exists():
        return {"error": f"no saved EDL '{edl_id}'"}
    if not render_p.exists():
        return {"error": f"no render for '{edl_id}' — call render first"}

    t = project.read_transcript(project_id)
    cleaned, _ = edl.apply_cleanup(project.read_json(edl_p), t)
    resolved = edl.resolve_edl(cleaned, t)
    intended = resolved["reconstructed_text"]

    stt = get_backend(backend)
    re_stt = stt.transcribe(render_p)
    actual = " ".join(w["w"] for w in re_stt["words"])
    result = verify.diff_transcripts(intended, actual)
    result.update({"project_id": project_id, "edl_id": edl_id})
    project.write_json(project.verify_path(project_id, edl_id), result)
    return result


# --- frames / previews ----------------------------------------------------------------

def extract_thumbnail(
    project_id: str,
    edl_id: str,
    at_time: float = 0.5,
    text: str | None = None,
    aspect: str = "9:16",
    aspect_mode: str = "crop",
) -> dict:
    """Grab a cover frame from a rendered clip's cut at `at_time` seconds, cropped to `aspect`,
    with optional burned title text. Writes renders/<edl_id>__<aspect>.jpg."""
    cut = project.render_path(project_id, edl_id)
    if not cut.exists():
        return {"error": f"no render for '{edl_id}' — call render first"}
    meta = project.read_meta(project_id)
    tw, th = media.aspect_dims(aspect, meta["width"], meta["height"])
    out = project.thumb_path(project_id, edl_id, aspect)
    font = config.FONTS_DIR / "Anton-Regular.ttf"
    media.extract_thumbnail(cut, out, at_time, tw, th, meta["width"], meta["height"],
                            aspect_mode=aspect_mode, text=text,
                            font_file=font if font.exists() else None)
    return {"project_id": project_id, "edl_id": edl_id, "thumbnail": str(out), "aspect": aspect}


def grab_frame(project_id: str, at: float, source: str = "source", grid: bool = True) -> dict:
    """Extract a still frame at `at` seconds so you can SEE the footage and choose framing.
    source='source' grabs the original video (use these pixel coords for focus shots); or pass
    an edl_id to grab that clip's cut at output time. grid overlays labeled pixel gridlines."""
    if not project.project_exists(project_id):
        return {"error": f"unknown project '{project_id}'"}
    meta = project.read_meta(project_id)
    if source == "source":
        video = Path(meta["source"])
        tag = f"source_{at:.2f}"
    else:
        video = project.render_path(project_id, source)
        if not video.exists():
            return {"error": f"no render for '{source}' — render it first, or use source='source'"}
        tag = f"{source}_{at:.2f}"
    out = project.frame_path(project_id, tag + ("_grid" if grid else ""))
    w, h = media.grab_frame(video, at, out, grid=grid)
    return {"frame": str(out), "at": at, "source": source, "width": w, "height": h,
            "note": "Read this PNG. x grows right, y grows down; coords are in these pixels."}


def preview_reframe(project_id: str, edl_id: str, at: float, aspect: str | None = None) -> dict:
    """Composite ONE output frame for an EDL's reframe plan at `at` seconds, so you can see what
    the viewer sees and tune framing before a full render. Requires the clip's cut to exist
    (render it once). Reads reframe from the saved EDL. Then Read the returned PNG."""
    cut = project.render_path(project_id, edl_id)
    if not cut.exists():
        return {"error": f"no render for '{edl_id}' — call render first"}
    edl_p = project.edl_path(project_id, edl_id)
    if not edl_p.exists():
        return {"error": f"no saved EDL '{edl_id}'"}
    meta = project.read_meta(project_id)
    saved = project.read_json(edl_p)
    rf = saved.get("reframe") or {"mode": "full"}
    censor = saved.get("censor") or None
    asp = aspect or rf.get("aspect") or "9:16"
    tw, th = media.aspect_dims(asp, meta["width"], meta["height"])
    out = project.frame_path(project_id, f"{edl_id}_preview_{at:.2f}_{asp.replace(':', 'x')}")
    rect = reframe.preview_frame(cut, rf, at, tw, th, out, censor=censor)
    return {"frame": str(out), "edl_id": edl_id, "at": at, "aspect": asp, "crop_rect": rect,
            "censored": bool(censor),
            "note": "Read this PNG to see the composited output (with any redactions) at this moment."}


# --- catalogs -------------------------------------------------------------------------

def list_caption_presets() -> dict:
    """List available caption style presets."""
    return {"presets": captions.list_presets(), "default": "karaoke-bold"}


def list_reframe_modes() -> dict:
    """List reframe shot modes and export aspects."""
    return {
        "modes": reframe.REFRAME_MODES,
        "see_before_framing": "Use grab_frame(at, grid=True) to view the footage and read pixel "
            "coords, then author focus shots; preview_reframe(edl_id, at) shows the composited "
            "output for a moment before a full render.",
        "aspects": list(media.ASPECT_DIMS.keys()),
        "timeline": (
            "Set reframe={aspect, transition_s, shots:[{start, mode, zoom?}, ...]} to switch "
            "between wide ('full') and zoomed-in ('track') across the clip; shots use OUTPUT "
            "(clip) seconds and transitions lerp the crop for a smooth zoom. A single "
            "reframe={mode,...} with no shots applies that mode to the whole clip."
        ),
    }


# --- clip suggestion (DESIGN BRIEF ONLY) ----------------------------------------------

_RUBRIC = (
    "Score each clip 0-10 on: hook (does it grab in the first 2s and tie to the theme), "
    "flow (logical build + payoff), value (utility/emotion/shareability). "
    "Favour self-contained segments that start and end on full sentences. "
    "Non-contiguous reordering is allowed and encouraged (open on the hook, cut back to "
    "setup, build, land the payoff). Each clip ~15-60s. Reference word INDICES only."
)

_SCHEMA_HINT = (
    '{"clips":[{"edl_id":"slug","title":"...","rationale":"why this works",'
    '"scores":{"hook":0,"flow":0,"value":0},'
    '"segments":[{"from_word":0,"to_word":0,"label":"hook"}]}]}'
)


def _indexed_text(transcript: dict, max_words: int = 6000) -> str:
    words = transcript["words"][:max_words]
    chunks, line = [], []
    line_start = 0
    for w in words:
        if not line:
            line_start = w["i"]
        line.append(w["w"])
        if len(line) >= 18:
            chunks.append(f"[#{line_start}] " + " ".join(line))
            line = []
    if line:
        chunks.append(f"[#{line_start}] " + " ".join(line))
    note = "" if len(transcript["words"]) <= max_words else f"\n(NOTE: truncated to first {max_words} words)"
    return "\n".join(chunks) + note


def suggest_clips(project_id: str, n: int = 3, instructions: str | None = None) -> dict:
    """Return a DESIGN BRIEF for proposing N candidate clip designs from the transcript.

    DESIGN-BRIEF ONLY for now: this returns the indexed transcript + the EDL schema + a
    scoring rubric so the calling agent can design the clips itself. It does NOT call any
    LLM/provider — the LLM-backed scoring path is wired in a later phase. Validate any EDL
    you design with validate_edl before rendering."""
    if not project.transcript_json_path(project_id).exists():
        return {"error": f"no transcript for '{project_id}' — call transcribe first"}
    t = project.read_transcript(project_id)
    return {
        "mode": "design_brief",
        "note": "Design the clips yourself from this brief, then validate_edl each. "
                "(LLM-backed scoring is wired in a later phase.)",
        "project_id": project_id,
        "n": n,
        "instructions": instructions or "",
        "rubric": _RUBRIC,
        "edl_schema": _SCHEMA_HINT,
        "indexed_transcript": _indexed_text(t),
    }
