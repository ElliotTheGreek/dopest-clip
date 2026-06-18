"""Shared operation surface.

Every capability dopest-clip exposes is a clean-signature function here. The MCP server
(server.py) registers each as a tool; the Electron sidecar (sidecar.py) dispatches to
the same callables by name. ONE definition, two faces — the agent and the human editor
drive identical operations over the same project model.

Heavy/optional subsystems (obs, reframe, the cloud providers) import their deps lazily,
so importing this module is cheap and the editing core works without them.
"""

from . import ops
from .audio import asr as _asr
from .audio import dsp as _dsp
from .audio import qa as _qa
from .audio import sfx as _sfx
from .audio import tts as _tts
from .image import gen as _img_gen
from .image import ops as _img_ops
from .providers import registry


# --- editing (ops) --------------------------------------------------------------------
create_project = ops.create_project
transcribe = ops.transcribe
get_transcript = ops.get_transcript
list_projects = ops.list_projects
get_project = ops.get_project
validate_edl = ops.validate_edl
render = ops.render
verify_clip = ops.verify_clip
extract_thumbnail = ops.extract_thumbnail
grab_frame = ops.grab_frame
preview_reframe = ops.preview_reframe
list_caption_presets = ops.list_caption_presets
list_reframe_modes = ops.list_reframe_modes
suggest_clips = ops.suggest_clips


# --- audio: local DSP -----------------------------------------------------------------
def audio_normalize(src: str, out: str | None = None, project_id: str | None = None, name: str = "normalized") -> dict:
    """Loudness-normalize audio to broadcast target (loudnorm I=-14:TP=-1.5:LRA=11)."""
    return _dsp.normalize(src, out, project_id=project_id, name=name)


def audio_denoise(src: str, out: str | None = None, method: str = "afftdn", project_id: str | None = None, name: str = "denoised") -> dict:
    """Reduce background noise via ffmpeg afftdn (default) or arnndn."""
    return _dsp.denoise(src, out, method=method, project_id=project_id, name=name)


def audio_trim_silence(src: str, out: str | None = None, threshold_db: float = -50.0, min_silence_s: float = 0.5, project_id: str | None = None, name: str = "trimmed") -> dict:
    """Remove leading/trailing/interior silence below threshold_db via silenceremove."""
    return _dsp.trim_silence(src, out, threshold_db=threshold_db, min_silence_s=min_silence_s, project_id=project_id, name=name)


def audio_gain(src: str, out: str | None = None, db: float = 0.0, project_id: str | None = None, name: str = "gain") -> dict:
    """Apply a fixed gain (dB) to audio."""
    return _dsp.gain(src, out, db=db, project_id=project_id, name=name)


def audio_fade(src: str, out: str | None = None, fade_in_s: float = 0.0, fade_out_s: float = 0.0, project_id: str | None = None, name: str = "faded") -> dict:
    """Apply an audio fade-in and/or fade-out (seconds)."""
    return _dsp.fade(src, out, fade_in_s=fade_in_s, fade_out_s=fade_out_s, project_id=project_id, name=name)


def audio_mix(srcs: list[str], out: str | None = None, weights: list[float] | None = None, project_id: str | None = None, name: str = "mixed") -> dict:
    """Mix multiple audio files into one (amix), optional per-input weights."""
    return _dsp.mix(srcs, out, weights=weights, project_id=project_id, name=name)


def audio_convert(src: str, out: str | None = None, fmt: str | None = None, sample_rate: int | None = None, channels: int | None = None, project_id: str | None = None, name: str = "converted") -> dict:
    """Convert audio format / sample rate / channel count."""
    return _dsp.convert(src, out, fmt=fmt, sample_rate=sample_rate, channels=channels, project_id=project_id, name=name)


# --- audio: cloud (provider-routed) ---------------------------------------------------
def tts(text: str, project_id: str | None = None, out: str | None = None, voice: str | None = None, fmt: str = "mp3") -> dict:
    """Synthesize speech from text via the active 'tts' provider. See learn://providers."""
    return _tts.synthesize(text, project_id=project_id, out=out, voice=voice, fmt=fmt)


def asr(src: str, engine: str = "auto") -> dict:
    """Transcribe an audio file. engine='auto'|'registry'|'local' (registry STT provider vs local whisperx)."""
    return _asr.transcribe_audio(src, engine=engine)


def sfx(prompt: str, project_id: str | None = None, out: str | None = None, duration: float | None = None) -> dict:
    """Generate a sound effect from a text description via the active 'sfx' provider."""
    return _sfx.sound_effect(prompt, project_id=project_id, out=out, duration=duration)


def audio_qa(audio_path: str, prompt: str | None = None, model: str | None = None) -> dict:
    """Assess audio quality with the active 'audio_qa' provider (e.g. OpenAI gpt-4o-audio)."""
    return _qa.quality_check(audio_path, prompt=prompt, model=model)


# --- image: provider-routed -----------------------------------------------------------
def image_generate(prompt: str, model: str, project_id: str | None = None, out: str | None = None) -> dict:
    """Generate an image from text via the active 'image' provider (you pass the model id)."""
    return _img_gen.generate(prompt, model, project_id=project_id, out=out)


def image_edit(image_path: str, instruction: str, model: str, project_id: str | None = None, out: str | None = None) -> dict:
    """Edit one image per an instruction via the active 'image' provider."""
    return _img_gen.edit(image_path, instruction, model, project_id=project_id, out=out)


def image_compose(image_paths: list[str], instruction: str, model: str, project_id: str | None = None, out: str | None = None) -> dict:
    """Compose multiple images per an instruction via the active 'image' provider."""
    return _img_gen.compose(image_paths, instruction, model, project_id=project_id, out=out)


def image_analyze(image_path: str, instruction: str, model: str) -> dict:
    """Vision: analyze/critique an image via the active 'image' provider. Returns text."""
    return _img_gen.analyze(image_path, instruction, model)


# --- image: local ops -----------------------------------------------------------------
def image_crop(src: str, out: str, x: int, y: int, w: int, h: int) -> dict:
    """Crop an image to (x, y, w, h)."""
    return _img_ops.crop(src, out, x, y, w, h)


def image_resize(src: str, out: str, width: int | None = None, height: int | None = None, keep_aspect: bool = True) -> dict:
    """Resize an image (optionally keeping aspect)."""
    return _img_ops.resize(src, out, width=width, height=height, keep_aspect=keep_aspect)


def image_pad(src: str, out: str, left: int = 0, top: int = 0, right: int = 0, bottom: int = 0, color: str = "#00000000") -> dict:
    """Pad an image with a colored/transparent border."""
    return _img_ops.pad(src, out, left=left, top=top, right=right, bottom=bottom, color=color)


def image_square_canvas(src: str, out: str, size: int | None = None, bg: str = "#00000000") -> dict:
    """Center an image on a square canvas."""
    return _img_ops.square_canvas(src, out, size=size, bg=bg)


def image_invert(src: str, out: str) -> dict:
    """Invert image colors."""
    return _img_ops.invert_colors(src, out)


def image_remove_background(src: str, out: str) -> dict:
    """Remove an image background (needs the [matting] extra / rembg)."""
    return _img_ops.remove_background(src, out)


def image_svg_to_png(src: str, out: str, width: int | None = None, height: int | None = None) -> dict:
    """Rasterize an SVG to PNG (needs the [graphics] extra / resvg)."""
    return _img_ops.svg_to_png(src, out, width=width, height=height)


def image_info(src: str) -> dict:
    """Return {width, height, mode, format} for an image."""
    return _img_ops.get_image_info(src)


def image_icon_set(src: str, out_dir: str, sizes: list[int] | None = None) -> dict:
    """Generate a square icon set at the given sizes."""
    return _img_ops.generate_icon_set(src, out_dir, sizes=sizes)


# --- providers ------------------------------------------------------------------------
def list_providers() -> dict:
    """List every capability, its providers, which is active, and whether each is configured."""
    return registry.list_providers()


def set_provider(capability: str, provider_name: str) -> dict:
    """Select the active provider for a capability (persists if providers.toml is writable)."""
    registry.set_provider(capability, provider_name)
    return {"capability": capability, "active": provider_name}


def validate_provider(capability: str) -> dict:
    """Report configuration status for the providers of a capability."""
    return {capability: registry.list_providers().get(capability, {})}


# --- recording (OBS, optional) --------------------------------------------------------
def list_devices() -> dict:
    """List OBS-visible monitors, cameras, and mics (needs a running OBS + [obs] extra)."""
    from .obs import client
    return client.list_devices()


def setup_scene(monitor: str, camera: str, mic: str, mic_track: int = 1) -> dict:
    """Build the track-separated OBS recording scene (idempotent, self-verifying)."""
    from .obs import client
    return client.setup_scene(monitor, camera, mic, mic_track=mic_track)


def start_recording() -> dict:
    """Start the OBS recording (screen + isolated camera together)."""
    from .obs import client
    return client.start_recording()


def stop_recording() -> dict:
    """Stop the OBS recording; returns the screen + camera file paths."""
    from .obs import client
    return client.stop_recording()


def recording_status() -> dict:
    """Current OBS recording state."""
    from .obs import client
    return client.recording_status()


def compose_camera(screen_path: str, camera_path: str, keyframes: list[dict], output_path: str,
                   remove_background: bool = False, max_duration: float | None = None,
                   overlays: list[dict] | None = None, blurs: list[dict] | None = None) -> dict:
    """Composite an isolated camera over a screen recording with an animated keyframe
    timeline (+ optional background removal, graphic overlays, and blur/focus effects)."""
    from .obs import compositor
    return compositor.compose(screen_path, camera_path, keyframes, output_path,
                              remove_background=remove_background, max_duration=max_duration,
                              overlays=overlays, blurs=blurs)


def mix_camera(project_id: str, edl_id: str, camera_path: str, keyframes: list[dict] | None = None,
               remove_background: bool = True, output_path: str = "", rematte: bool = False,
               overlays: list[dict] | None = None, blurs: list[dict] | None = None,
               screen_keyframes: list[dict] | None = None, bg_visible_until: float | None = None) -> dict:
    """Mix the camera into a project's cut screen (GPU matte+NVENC, cut-synced) with the full
    effect stack — all optional, all timed in CUT-timeline seconds (see get_cut_transcript):
    `keyframes` animate the camera (presets fullscreen/center/pip/top-left/.../pos+scale, for
    make-me-big / corner / slide); `overlays` are animated graphics (arrow/ring/box/label,
    inline `svg`, or a transparent `image` PNG — for arrows, a ring on a face, a kitten in
    hand) with keyframes + t_in/t_out/fade; `blurs` are animated screen blur/focus regions
    (`invert:true` = blur everything BUT the shape = "blur all but my face"); `screen_keyframes`
    crop+zoom the screen over time (zoom into a button); `bg_visible_until` keeps the FULL camera
    background visible until that second, then drops to the cutout. Any effect can FOLLOW a moving
    target: add `track:{target, source}` inside an overlay/blur spec, or on a screen_keyframe/keyframe
    (target = 'cursor' | 'face' | a COCO class | {template_at, region}; source 'screen'|'camera').
    The static keyframes still set HOW BIG the effect is; the track sets WHERE it sits. Use
    preview_track first to confirm the target locks on."""
    from .obs import camera_mix
    return camera_mix.mix(project_id, edl_id, camera_path, keyframes=keyframes,
                          remove_background=remove_background, output_path=output_path,
                          rematte=rematte, overlays=overlays, blurs=blurs,
                          screen_keyframes=screen_keyframes, bg_visible_until=bg_visible_until)


def get_cut_transcript(project_id: str, edl_id: str) -> dict:
    """Derive + return the CUT-timeline transcript (final spoken words after cleanup/cut,
    re-indexed 0..N with cut-timeline timestamps). Design shorts against THESE indices
    (from_word/to_word for make_short). Writes <render>.cut_transcript.{json,txt}."""
    from .obs import camera_mix
    cj, ct, n = camera_mix.write_cut_transcript(project_id, edl_id)
    text = ""
    try:
        with open(ct, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        pass
    return {"project_id": project_id, "edl_id": edl_id, "word_count": n,
            "cut_transcript_json": cj, "cut_transcript_txt": ct, "text": text}


def make_short(project_id: str, edl_id: str, from_word: int, to_word: int, hook_title: str,
               screen_keyframes: list[dict] | None = None, caption_preset: str = "karaoke-bold",
               output_path: str = "", overlays: list[dict] | None = None) -> dict:
    """Render a 9:16 SHORT-FORM clip with the signature stacked layout: hook + karaoke
    captions in the TOP band, the screen (with optional per-frame zoom from
    `screen_keyframes`) in the MIDDLE, and the background-removed person BIG at the BOTTOM
    over a blurred backdrop. `overlays` adds animated graphics (arrows/labels/etc.) on top.
    Requires render(edl_id) (cut screen) and mix_camera(edl_id, remove_background=True) (GPU
    matte) to have run first. `from_word`/`to_word` index the CUT transcript (see
    get_cut_transcript). GPU/NVENC."""
    from .obs import camera_mix
    return camera_mix.short_clip(project_id, edl_id, from_word, to_word, hook_title,
                                 screen_keyframes=screen_keyframes, caption_preset=caption_preset,
                                 output_path=output_path, overlays=overlays)


def preview_track(project_id: str, target: str | dict, edl_id: str = "", source: str = "screen",
                  video_path: str = "", at: float | None = None) -> dict:
    """Confirm a tracking target locks on BEFORE a full render. Runs the detector over the
    source video and returns a downsampled per-frame track plus a preview frame (at `at`
    seconds, default midpoint) with the tracked point drawn. Source: 'screen' = the cut
    render <project>/renders/<edl>.mp4, 'camera' = the cut camera <project>/camera/<edl>_cut.mp4,
    or pass an explicit `video_path`. `target` is 'cursor' | 'face' | a COCO class name
    (person/cup/laptop/phone/...) | {'template_at': seconds, 'region': [x,y,w,h]} to follow a
    UI element. The same `track:{target, source}` field then rides inside the overlays / blurs /
    screen_keyframes / keyframes of mix_camera / make_short to make that effect follow the target."""
    from .obs import tracking
    if not video_path:
        from . import project
        pdir = project.require_project(project_id)
        slug = project.slugify(edl_id)
        video_path = str((pdir / "camera" / f"{slug}_cut.mp4") if source == "camera"
                         else (pdir / "renders" / f"{slug}.mp4"))
    return tracking.preview(video_path, target, at=at)


def list_graphics() -> dict:
    """The built-in overlay graphic kinds and their params, for use in compose_camera
    `overlays`. You can also pass a custom inline `svg` or a pre-rendered transparent
    `image` (e.g. a generated lightbulb/clip-art PNG). Each overlay animates over
    keyframes [{t, pos:[nx,ny], scale, ease}] with t_in/t_out, fade, and opacity."""
    return {
        "kinds": {
            "arrow": {"params": ["direction", "color", "stroke"],
                      "anchor": "tip (lands on pos)",
                      "direction": ["up", "down", "left", "right",
                                    "up-left", "up-right", "down-left", "down-right"]},
            "ring": {"params": ["color", "stroke"], "anchor": "center"},
            "box": {"params": ["color", "stroke", "aspect", "radius"], "anchor": "center"},
            "label": {"params": ["text", "color", "text_color", "font_size"], "anchor": "center"},
        },
        "custom": {"svg": "inline SVG string", "image": "path to a transparent PNG"},
        "animation": {"keyframes": "[{t, pos:[nx,ny], scale, ease}]",
                      "extras": ["t_in", "t_out", "fade", "opacity"]},
        "used_by": "compose_camera(overlays=[...])",
    }


# --- async render jobs (long renders run in the background; poll instead of blocking) ----
_RENDER_OPS = {"mix_camera", "make_short", "render", "compose_camera", "verify_clip"}


def start_render(operation: str, params: dict | None = None) -> dict:
    """Start a long render in the BACKGROUND and return a job_id immediately, so a
    multi-minute render goes through MCP without a synchronous tool-call timeout. `operation`
    is one of mix_camera / make_short / render / compose_camera / verify_clip; `params` is that
    op's keyword args. Poll render_status(job_id) until status == 'done' (result holds the op's
    return) or 'error'. Use this for any real-length render; the matte is cached so re-runs are
    quick."""
    from . import jobs
    if operation not in _RENDER_OPS:
        return {"error": f"start_render runs only {sorted(_RENDER_OPS)}, got {operation!r}"}
    fn = OPERATIONS.get(operation)
    if fn is None:
        return {"error": f"unknown operation {operation!r}"}
    jid = jobs.start(operation, fn, **(params or {}))
    return {"job_id": jid, "operation": operation, "status": "running",
            "next": f"poll render_status(job_id='{jid}')"}


def render_status(job_id: str) -> dict:
    """Poll a background render started with start_render: {status: running|done|error,
    elapsed_s, result (the op's return when done), error (when failed)}."""
    from . import jobs
    return jobs.status(job_id)


def list_render_jobs() -> dict:
    """List all background render jobs this session with their status + elapsed time."""
    from . import jobs
    return jobs.list_jobs()


# --- the operation registry (name -> callable), grouped for discovery -----------------
GROUPS: dict[str, list] = {
    "editing": [create_project, transcribe, get_transcript, list_projects, get_project,
                validate_edl, render, verify_clip, extract_thumbnail, grab_frame,
                preview_reframe, list_caption_presets, list_reframe_modes, suggest_clips],
    "audio": [audio_normalize, audio_denoise, audio_trim_silence, audio_gain, audio_fade,
              audio_mix, audio_convert, tts, asr, sfx, audio_qa],
    "image": [image_generate, image_edit, image_compose, image_analyze, image_crop,
              image_resize, image_pad, image_square_canvas, image_invert,
              image_remove_background, image_svg_to_png, image_info, image_icon_set],
    "providers": [list_providers, set_provider, validate_provider],
    "recording": [list_devices, setup_scene, start_recording, stop_recording,
                  recording_status, compose_camera, mix_camera, get_cut_transcript,
                  make_short, list_graphics, preview_track],
    "jobs": [start_render, render_status, list_render_jobs],
}

OPERATIONS: dict[str, object] = {fn.__name__: fn for fns in GROUPS.values() for fn in fns}
