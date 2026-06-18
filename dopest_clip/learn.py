"""Baked-in MCP learn:// resources.

These are docs the agent fetches as MCP resources (not tools) to learn the studio
before driving it. RESOURCES maps each URI to its markdown body; server.py registers
one @mcp.resource per key.
"""

RESOURCES: dict[str, str] = {}

RESOURCES["learn://overview"] = """\
# dopest-clip — overview

dopest-clip is a self-contained, provider-agnostic media studio: record (OBS) ->
edit (transcript-driven clipper) -> process audio -> generate/edit images. It is
drivable by an MCP agent (these tools) and by a human in an Electron editor over the
SAME project model.

## The core loop (editing)
    create_project(video) -> transcribe(project) -> READ the transcript.txt file ->
    design an EDL -> validate_edl (cheap, no render) -> render -> verify_clip (QA)

The agent does the creative reasoning (reading the transcript, designing the edit).
The tools are accurate primitives: STT, a text-space validation loop, a renderer, a
subject-tracked portrait reframe, burned captions, and an STT-based QA gate.

## The subsystems (read the matching resource before using each)
- learn://recording  — OBS track-separated capture (optional; editing needs no OBS)
- learn://editing     — projects, EDLs, render pipeline
- learn://cutting     — silence-aware cuts + reading warnings
- learn://reframe      — portrait subject tracking + shot timeline
- learn://captions     — burned caption presets + title cards
- learn://audio        — local ffmpeg DSP + cloud TTS/ASR/SFX/QA
- learn://image        — provider image gen/edit/compose/analyze + local ops
- learn://providers     — the provider registry (FlowDot is one option among many)
- learn://gotchas       — environment + runtime lessons

## Provider-agnostic by design
Every cloud capability (LLM, STT, TTS, SFX, audio-QA, image) routes through a
registry where you pick the provider. FlowDot is one option; so are OpenAI,
OpenRouter, Fish, ElevenLabs, Gemini. Nothing is locked to FlowDot. See
learn://providers.
"""

RESOURCES["learn://recording"] = """\
# Recording (OBS) — OPTIONAL

dopest-clip records with OBS via obs-websocket v5. This half is optional: editing
works on any input video without OBS. Recording requires OBS 28+, obs-websocket
enabled, and the **Source Record** OBS plugin installed.

## Track-separated capture (the value)
The scene records the SCREEN (monitor_capture, with mic on a track) to the main OBS
output, and the CAMERA to a SEPARATE isolated file via the Source Record filter. The
camera input is enabled but moved OFF-CANVAS so it streams frames for Source Record
without appearing in the screen file. Post-recording you composite the camera over
the screen with full control (position, scale, keyframes, background removal).

## Tools
- list_devices() — monitors / cameras / mics
- setup_scene(monitor, camera, mic, mic_track=1) — idempotent; self-verifies a test record
- start_recording() / stop_recording() — screen + camera start/stop together
- recording_status()
- compose_camera(...) — animate the camera over the FULL screen master (+ clip-art
  overlays + blur); list_graphics() shows overlay kinds (arrow/ring/box/label) — you can
  also pass a custom inline svg or a transparent PNG (e.g. a generated lightbulb).
- mix_camera(edl_id, camera_path, keyframes=…, remove_background=True, overlays=…, blurs=…,
  screen_keyframes=…, bg_visible_until=…) — the UNIFIED GPU compose-over-cut. One GPU pass
  (RVM+NVENC; CPU rembg fallback) over the CUT screen: animated camera rect (presets
  fullscreen/center/pip/top-left/… = make-me-big / corner / slide) + `overlays` (arrow/ring/
  box/label, inline `svg`, or transparent `image` PNG) + `blurs` (animated blur; `invert:true`
  = blur everything BUT the shape = "blur all but my face") + `screen_keyframes` (crop+zoom
  the screen, e.g. into a button) + `bg_visible_until` (keep the FULL camera background visible
  until that second, then drop to a cutout). All times are CUT-timeline seconds.
- get_cut_transcript(edl_id) — the cut-timeline word indices/times to drive shorts AND to time
  every mix_camera effect (overlays/zooms/bg-drop land on the right spoken moments after cuts).
- make_short(edl_id, from_word, to_word, hook_title, screen_keyframes=…, overlays=…) — render a
  9:16 SHORT: hook + karaoke captions TOP, screen (optional per-frame zoom) MIDDLE, background-
  removed person BIG at the BOTTOM, with optional graphic overlays on top. Needs render() +
  mix_camera(remove_background=True) first (reuses the cached matte). GPU/NVENC.

## Long renders: drive them async over MCP (never block the tool call)
A real-length matte+composite takes minutes — too long for one synchronous tool call. So
run any render through the job surface: start_render(operation, params) returns a job_id
immediately (operation = "mix_camera"|"make_short"|"render"|"compose_camera"|"verify_clip",
params = that op's kwargs); poll render_status(job_id) until status == "done" (its `result`
holds the op's return) or "error". list_render_jobs() shows all jobs. The matte is cached, so
re-running a mix_camera composite (tweaking overlays/positions) is quick.

## Talking-head short, end to end
record → create_project(SCREEN) → transcribe → design+render an EDL → mix_camera(edl_id,
camera_path, remove_background=True) (builds the GPU matte) → get_cut_transcript(edl_id) →
make_short(edl_id, from_word, to_word, hook_title). Matte the SHORT, not the long form, as
a first proof (see learn://gotchas for matte speed + the compose-CPU vs mix-GPU split).

## Clip-art / graphic overlays
Overlays work in mix_camera (over the CUT, GPU), make_short (over the 9:16 short), AND
compose_camera (over the FULL uncut master, CPU). Each overlay is {"image": "transparent.png"}
OR {"svg": "<inline>"} OR a built-in kind (see list_graphics), plus: anchor [ax,ay] (the point on the graphic
placed at pos), keyframes [{t, pos:[nx,ny], scale, ease}] (t is LOCAL to t_in), and
t_in/t_out/fade/opacity. Time a pop to a spoken word: read that word's start/end from
transcript.json and set t_in just before it (a scale 0.05→0.14→0.11 keyframe = a pop).
Place it over the head by putting pos just ABOVE the camera rect — e.g. camera bottom-right
PIP at scale 0.3 → head ≈ [0.83, 0.71], so the bulb sits at ≈ [0.83, 0.59]. ALWAYS grab a
frame at the pop time to check placement and nudge. (Transparent PNG recipe: learn://image.)

## Dynamic tracking — make any effect FOLLOW a moving target
Every effect above sits at static keyframes. To make one RIDE a moving target instead, add a
`track` field; the keyframes still set HOW BIG the effect is, the track sets WHERE it sits:
- `track: {target, source}` rides INSIDE an overlay or blur spec, or on a screen_keyframe /
  camera keyframe (e.g. `screen_keyframes=[{t:0, zoom:2, track:{target:"cursor"}}]`).
- target = `"cursor"` (the OS pointer, matched on the screen), `"face"` (the camera face),
  a COCO class (`"person"`,`"cup"`,`"laptop"`,`"cell phone"`,…), or `{"template_at": seconds,
  "region": [x,y,w,h]}` to lock onto a UI element (a button) and follow it as the page scrolls.
- source = `"screen"` (default; tracks the cut screen) or `"camera"` (tracks the cut camera —
  use for `"face"`). A screen-zoom that rides the cursor, an arrow glued to a button, a ring
  that tracks the face: same `track` field, different target.
- `offset: [ox, oy]` places the effect RELATIVE to the target, in units of the tracked box size
  (so it scales with the target). This is how the caller says WHERE around the target: a halo/
  lightbulb ABOVE the head = `track:{target:"face", source:"camera", offset:[0,-0.9]}` on an
  overlay anchored bottom-centre; a ring AROUND the head = `offset:[0,0]` with the ring scaled to
  the head; a badge to the side = `offset:[0.9,0]`. A camera-source overlay maps through the
  per-frame composited camera rect, so it lands on the cutout whether the camera is PIP,
  fullscreen, or animated.
- preview_track(project_id, edl_id, target, source=…) FIRST — it runs the detector and returns
  a downsampled track + a frame with the tracked point drawn, so you confirm the lock before a
  full render. The track is cached per (video, target) under <project>/camera/.
- Cursor tracking needs assets/cursor.png (a crop of the OS pointer); if absent it raises a
  clear error telling you to grab+crop one. Detectors run on cv2 (cursor/face/template) or GPU
  YOLO (COCO classes); the rest of the GPU compose pass is unchanged.

## Connection (env)
OBS_WS_HOST (localhost), OBS_WS_PORT (4455), OBS_WS_PASSWORD, OBS_SCENE_NAME, OBS_CAMERA_DIR.

See learn://gotchas for the Source Record CBR / settle-time / device-race lessons.
"""

RESOURCES["learn://editing"] = """\
# Editing — projects, EDLs, render

## Project
create_project(video_path) probes the file, extracts 16kHz mono audio, and makes a
project dir. transcribe(project_id) writes transcript.json + a readable transcript.txt
with [timestamp] (#word_index) lines — READ that file to find word ranges to cut on.

## EDL (edit decision list) — the data model
An EDL is the edit design:
    {"edl_id": "hook-v1", "title": "...",
     "segments": [{"from_word": 880, "to_word": 905, "label": "hook"},
                  {"from_word": 0,   "to_word": 120, "label": "setup"}],
     "cleanup": {"remove_fillers": true, "max_pause": 1.0},
     "reframe": {"mode": "track", "aspect": "9:16"},
     "captions": {"enabled": true, "preset": "karaoke-bold"},
     "title_card": {"text": "...", "hold_s": 3},
     "loudnorm": true, "export_aspects": ["9:16","1:1"]}
Word indices are stable. Segments are reorderable and reusable — non-contiguous
reordering (open on the hook, cut back to setup) is first-class.

## Loop
validate_edl(project_id, edl) resolves the EDL with NO render: snaps cuts to silence,
checks bounds, returns the reconstructed dialog + warnings + cleanup report. Iterate
here cheaply. Then render(project_id, edl_or_id) produces the clean cut
renders/<edl_id>.mp4 (always) plus styled per-aspect deliverables (if styling set).
verify_clip(project_id, edl_id) re-STTs the render and diffs vs intended text.
"""

RESOURCES["learn://cutting"] = """\
# Cutting — silence-aware boundaries

Cuts snap to silence so joins don't clip words. A gap >= MIN_SILENCE (0.15s) counts
as a clean boundary; the cut lands SNAP_MARGIN (0.08s) inside it. If a segment's start
or end word has no silence beside it, validate_edl/render return a WARNING and make a
hard cut at the word onset/offset — pick an earlier/later boundary word for a clean
join. A per-join audio fade (CROSSFADE_MS, 15ms) kills clicks. cleanup={remove_fillers,
max_pause} drops filler spans and splits segments at long internal pauses.
"""

RESOURCES["learn://reframe"] = """\
# Reframe — portrait, subject-tracked

reframe on an EDL converts landscape to portrait/square. Modes:
- track: YOLO11 follows the person, One-Euro-smoothed crop
- zoom: track with a tighter crop
- focus: frame explicit source pixels (use grab_frame(grid=True) to read coords)
- full: whole frame centered with blurred bands top/bottom
- center / pad / none
Aspects: 9:16, 4:5, 1:1, 16:9, source. A shot timeline switches modes across the clip:
    reframe={aspect, transition_s, shots:[{start, mode, zoom?}, ...]}  (OUTPUT seconds)
Use grab_frame to SEE the footage and preview_reframe to SEE the composited output
before a full render. Needs the [reframe] extra (ultralytics + opencv) + a GPU helps.

## Tracking effects (not just the portrait crop)
The same subject-following that drives the portrait crop is also exposed to the COMPOSE
effects so any overlay / blur / screen-zoom / camera rect can FOLLOW a target (the cursor,
the face, a COCO object, or a UI element) instead of static keyframes. See learn://recording
"Dynamic tracking": add `track:{target, source}` to an effect spec and confirm with
preview_track before rendering.
"""

RESOURCES["learn://captions"] = """\
# Captions — burned ASS

captions={enabled:true, preset, font?, position?} burns word-timed captions. Presets:
- karaoke-bold (Anton, per-word pop highlight, bottom)
- lower-third (Arial, phrase-based, bottom)
- minimal-top (Arial, phrase-based, top)
title_card={text, hold_s} burns an opening title. Captions are driven by the OUTPUT
(cut) timeline, so reordering is reflected correctly. The bundled Anton font ships in
the package assets.
"""

RESOURCES["learn://audio"] = """\
# Audio — local DSP + cloud

## Local ffmpeg DSP (no provider needed)
normalize (loudnorm I=-14:TP=-1.5:LRA=11), denoise (afftdn/arnndn), trim_silence
(silenceremove), gain, fade, mix, convert. All run through the shared ffmpeg runner.

## Cloud (via the provider registry — see learn://providers)
- tts: synthesize speech (Fish / FlowDot)
- asr: transcribe audio (registry Fish ASR, or local whisperx via dopest_clip.stt)
- sfx: generate a sound effect (ElevenLabs)
- audio_qa: rate audio quality (OpenAI gpt-4o-audio)
Outputs persist into the project's audio/ folder.
"""

RESOURCES["learn://image"] = """\
# Image — generate/edit + local ops

## Provider-routed (registry capability "image", default Gemini BYOK)
- generate(prompt, model) — text -> image
- edit(image, instruction, model) — one image + instruction -> image
- compose(images, instruction, model) — combine images
- analyze(image, instruction, model) — vision / critique -> text
You pass the model id, so it stays current as vendors ship new models.

## Local ops (Pillow; some need extras)
crop, pad, resize, square_canvas, recolor, invert_colors, get_image_info,
generate_icon_set; remove_background ([matting] / rembg), svg_to_png ([graphics] /
resvg), png_to_svg (needs vtracer; raises clearly if absent — never faked).

## Transparent clip-art for video overlays (the "lightbulb" recipe)
compose_camera overlays take a transparent PNG, an inline SVG, or a built-in kind
(list_graphics). Simplest + provider-free: pass {"svg": "<inline svg>"} — resvg
rasterizes it with alpha, no key and no extra step. For a generated raster instead:
image_generate(prompt, model) the art, then image_remove_background (rembg) to cut it to
an RGBA cutout, and use it as an {"image": path} overlay.
"""

RESOURCES["learn://providers"] = """\
# Providers — the registry (not locked to FlowDot)

Every cloud capability routes through a registry: capability -> active provider.
Capabilities: llm, stt, tts, sfx, audio_qa, image.

## Tools
- list_providers() — per capability: available providers, which is active, and whether
  each is `configured` (key present). Never raises, never hits the network.
- set_provider(capability, provider_name) — choose the active provider (persists to
  providers.toml if writable). Selecting an unconfigured provider is allowed; it only
  errors when actually used.
- validate_provider(capability) — configuration status for that capability's providers.

## Resolution order for the active provider
in-memory selection -> providers.toml [active] -> env DOPEST_PROVIDER_<CAP> -> code
default -> first registered. A missing key turns into a LOUD error at use time, never a
silent fallback to a different vendor.

## Keys (env)
OPENAI_API_KEY, ELEVENLABS_API_KEY, FISH_AUDIO_API_KEY, GEMINI_API_KEY,
OPENROUTER_API_KEY, and FLOWDOT_* / FLOWDOT_BASE_URL. FlowDot is one aggregator option
(llm/image/audio); OpenRouter is an LLM aggregator only.
"""

RESOURCES["learn://gotchas"] = """\
# Gotchas — environment + runtime lessons

- ffmpeg/ffprobe are spawned with stdin=DEVNULL inside the MCP stdio server; otherwise
  ffmpeg inherits the agent<->server pipe and hangs the server. (media.run_ff handles it.)
- An empty OPENAI_BASE_URL (or any base_url) env var must be treated as UNSET, not as a
  valid empty URL. The registry's env_str + the STT backend coerce empty->None.
- OBS Source Record: rate_control MUST be uppercase "CBR" (lowercase silently fails
  NVENC); record_mode=3; a fresh filter needs ~6s to settle before recording; capture
  devices tear down asynchronously (poll until absent before recreate); reopening a USB
  camera too fast triggers DShow 0x800705AA, so the scene reuses unchanged inputs.
- Heavy deps are optional extras: [stt] torch/whisperx, [reframe] ultralytics/opencv,
  [matting] rembg+torch, [graphics] resvg, [obs] websocket-client. The package imports
  and the editing core run without them; each errors clearly only when its work runs.
- GPU paths (whisperx large-v3, RVM matting, YOLO) want CUDA; CPU works but is slower.
  torch must be a CUDA build — a `+cpu` wheel reports cuda_available False and matting
  falls back to slow CPU. Install the matching CUDA wheel from download.pytorch.org/whl.
- TWO camera-matte paths: mix_camera (the CUT path) uses GPU RVM + NVENC; compose_camera
  (the FULL-master path, and the ONLY one with graphic overlays) mattes on CPU with rembg
  per frame — SLOW (minutes for ~15s). For a fast cutout use mix_camera; reach for
  compose_camera+remove_background only for short windows or when you need overlays.
- GPU RVM matte runs ~16 fps at 1080p on this pipeline (model forward ~30 fps; decode +
  CPU->GPU + dual nvenc encode roughly halve it). So an 87s long-form matte is ~2.7 min, a
  ~9s short ~15s. PROVE / iterate on a SHORT clip first — never launch the long-form matte
  as a first test. matte fgr/pha are CACHED per (project, edl) under <project>/camera/;
  pass rematte=True only after the cut changes.
- Cancelling/interrupting an MCP tool does NOT stop in-flight work in the server: a matte
  keeps running (runaway) and holds the GPU. If a render seems stuck or weirdly slow, check
  `nvidia-smi` (a hidden compute app at high util = a runaway) and stop the `dopest_clip`
  python processes before retrying.
- Matte intermediates encode via h264_nvenc (libx264 fallback). Do NOT set
  torch.backends.cudnn.benchmark for the matte: the last partial chunk is a new tensor
  shape, so cudnn re-autotunes per size and it gets SLOWER on short clips.
"""
