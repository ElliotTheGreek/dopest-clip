# dopest-clip — Concrete Build Plan

> **BUILD STATUS (all 8 phases implemented + tested; 51 MCP ops).** Python core complete
> and self-contained (config, project, edl, media, stt, providers, audio, image, obs, ops,
> api, server, sidecar, learn). `python -m dopest_clip` (stdio) and `python -m dopest_clip
> --serve` (localhost JSON-RPC sidecar) both work. Electron editor in `desktop/` builds
> clean. Tests: **200 passed / 6 skipped** (Python) + **51 passed** (vitest).
>
> **PARITY FIXES 2026-06-18** (a live OBS capture test exposed that dopest-clip had regressed
> from the old obs-mcp+short-form combo — no cutout, no person in the short, no clip art, no
> camera animation, "empty" editor). Done + unit-tested, **live verification PENDING a session
> restart**: (1) **GPU enabled** — venv torch was `2.12.1+cpu` on a box with an RTX 3080 Ti;
> swapped to `+cu130` (torch 2.12.1 / torchvision 0.27.1 / torchaudio 2.11.0), CUDA now live.
> (2) **mix_camera** now GPU RVM+NVENC matte when CUDA present, CPU rembg fallback otherwise
> (was GPU-only and crashed). (3) **make_short / vertical_clip / write_cut_transcript ported**
> — the signature 9:16 featured layout (captions top / screen-zoom middle / background-removed
> person big at bottom). (4) **3 ops exposed**: `make_short`, `get_cut_transcript`,
> `list_graphics` (clip-art overlays already work via `compose_camera`). (5) **Editor** got an
> in-window `<video>` player (streams over `dopest-file://`) + camera-mix + make-short controls,
> so it drives the same pipeline and you can watch the result.
>
> **QA'd live via electron-qa MCP** — app launches, mounts, reaches the sidecar (health
> in header), Providers tab renders live `list_providers` for all 6 capabilities,
> `create_project` round-trips (ffmpeg probe + audio extract) and lists the project, and
> provider/transcribe errors surface gracefully in the status bar (no crash). **Three
> real bugs found + fixed during QA:** (1) Electron main built as ESM → `__dirname` crash;
> fixed with CommonJS electron output (dropped `type:module`; electron tsconfig
> `module:CommonJS`). (2) renderer wouldn't load over `file://` (opaque-origin ES modules
> blocked) → added an `app://` privileged protocol serving `dist/`. (3) Inspector crashed
> on `modes.map` because `list_reframe_modes().modes` is a dict, not an array → added an
> `asNames()` coercion. Also added a fatal-error catcher in `main.tsx` so a boot throw
> shows in-window instead of a blank screen.
>
## REMAINING WORK — full parity + the live demo (2026-06-18)

A real recorded demo (`projects/demo1`, a 4.5-min take with ~10 spoken effect requests) is the
acceptance test. GOAL: **100% parity with the old obs-mcp + short-form-editor combo — every
effect re-implemented properly and tested.** No corner-cutting.

**Where we actually stand (evidence-based audit of obs-mcp vs dopest_clip):** dopest-clip already
holds the old combo's full FEATURE set — `compose_camera` does graphic overlays (arrow/ring/box/
label/inline-svg/image-PNG) + animated blur including **focus-region via `invert`** (this IS
"blur everything but my face") + camera keyframe moves (big/corner/slide presets) + CPU rembg
matte; `make_short`/`vertical_clip` does GPU screen-zoom + karaoke captions + person-bottom 9:16;
`mix_camera` does GPU RVM+NVENC matte over the cut; EDL cut/transcript/reframe/verify all present.
The old `obs-mcp` `compose_camera` was ALSO CPU-matte with no screen-zoom, so those were never
parity gaps. Every effect the demo asks for (lightbulb, kitten place+remove, arrows, circle-on-
face=ring, blur-bg, blur-all-but-face=invert blur, big/corner/slide, captioned cutout-bottom short)
is expressible with TODAY's tools.

**What's genuinely unfinished (the real work):** making the full effect stack run **fast (GPU) over
the CUT, in one pass** — neither old nor new does this (old combo had GPU-matte in `mix_camera` OR
overlays in `compose_camera`, never both; CPU `compose_camera` on a 4-min clip is far too slow).

### Tasks
1. **Unified GPU compose-over-cut** (centerpiece). Extend the GPU composite path
   (`obs/camera_mix.py` `composite_gpu`, reusing the cached RVM fgr/pha) so a single render over
   the CUT screen + cut-synced camera applies, on the GPU with NVENC: camera keyframes (have it) +
   **graphic overlays** + **animated blur incl. focus-region** + **screen zoom** (`timeline.
   sample_screen`, port from `vertical_clip`). Expose via `mix_camera` gaining `overlays`, `blurs`,
   `screen_keyframes` params (or a new `compose_cut` op). This is what makes the demo feasible.
2. **Mid-clip background toggle** — background VISIBLE then removed at a cue ("enter dopest clip").
   Add a time-gated matte to the GPU compose (opaque camera before t_cue, RVM alpha after), or a
   clean two-phase render + concat. New (old combo lacked it).
3. **Effect timing on the CUT timeline** — effect `t` values key off the cut-timeline word times
   (`get_cut_transcript` / `edl.remap_to_output_timeline`), so overlays/zooms land on the right
   spoken moments after cuts remove sections.
4. **make_short overlays** — add an `overlays` param to `vertical_clip` (arrows/labels in shorts).
5. **Asset prep helpers** — generate transparent PNGs (kitten, lightbulb) via the image provider +
   `image_remove_background`; locate hand/face position from `grab_frame` to place them.
6. **CPU fallback stays correct** — `compose_camera` (CPU rembg) remains for no-GPU boxes; same
   effect API, just slower. Parity does not depend on a GPU.

### Acceptance / verification (against `projects/demo1`)
- Long-form (16:9): clean cut (remove the 4 "cut"-marked fumbles) + GPU cutout from "enter dopest
  clip" + lightbulb at "then I had an idea" + kitten place/remove + circle-on-face + blur-all-but-
  face + make-me-big + corner/slide + arrows on pointed items — rendered fast, verified by frames.
- Short-form (9:16): a compelling segment via `make_short` (captions top / screen-zoom mid / cutout
  bottom) with arrows.
- Both added near the top of the README. Unit tests for every new GPU-compose path.
- Screen-zoom-into-the-record-button in LONG-form is delivered by Task 1's screen-zoom-over-cut.

### STATUS — DONE (2026-06-18, verified on demo1)
Unified GPU compose-over-cut shipped: `composite_gpu` now does screen-zoom + animated blur/focus +
graphic overlays + mid-clip background toggle in one NVENC pass; `mix_camera`/`make_short` expose
`overlays`/`blurs`/`screen_keyframes`/`bg_visible_until`. Long renders run async over MCP via
`start_render`/`render_status` (no synchronous timeout). MCP control surface (api/server/sidecar/
jobs/__main__) at **100% coverage**. demo1 cut to landscape (16:9, all effects) + portrait (9:16);
every requested feature frame-verified.

---

## NEXT LEG — dynamic tracking (follow faces, the mouse, called-out objects) (2026-06-18)

**Where we stand:** subject/face tracking ALREADY exists at parity — `reframe.py` `track` mode
(YOLO11 `_load_model`/`_detect_center` + `_OneEuro` smoothing) follows the person for a portrait
crop, via `render(reframe={mode:"track", aspect:...})`. That matches the old combo.

**The extension (new, not in the old combo):** make ANY effect *follow a moving target* instead of
sitting at static keyframes — so a screen-zoom rides the mouse, an arrow stays glued to a button as
the page scrolls, the face-ring auto-tracks the face, etc. The missing piece is a per-frame
**tracking position source** feeding the effect position.

### Tasks
1. **Tracking module** (`dopest_clip/obs/tracking.py`): `compute_track(video, target, fps) ->
   [{t, x, y, w, h}]` (normalized), smoothed with `reframe._OneEuro` (reuse). Targets:
   - `"cursor"` — detect the OS cursor baked into the screen capture (template-match a shipped
     cursor sprite asset + small-fast-moving fallback). Source = the screen video.
   - `{template_at: t, region: [x,y,w,h]}` — seed an OpenCV tracker (CSRT) from that region and
     follow it across frames (handles scroll/move). Source = screen or camera.
   - `"face"` — face detector (OpenCV DNN/haar or YOLO-face) on the CAMERA frames.
   - `"<yolo-class>"` (person/cup/laptop/…) — reuse `reframe._load_model` (YOLO11) on the camera.
2. **`track` field on effect specs** (overlay / blur / screen_keyframes / camera keyframes):
   `track: {target, source:"screen"|"camera"}`. When present, the compositor resolves the per-frame
   position from a PRE-COMPUTED track (run detection once before the GPU frame loop, index by frame)
   instead of `timeline.sample_*` static keyframes. Wire into `camera_mix.composite_gpu` +
   `vertical_clip` (+ CPU `compositor.compose`).
3. **Surface**: the `track` field rides inside the existing `overlays`/`blurs`/`screen_keyframes`
   dicts — no new required params. Add a `preview_track(project_id, video, target)` op (returns the
   track + a sample frame) so the agent can sanity-check a target before a full render. Document in
   `learn://reframe` + `learn://recording`.
4. **Tests**: keep MCP-control coverage at 100%; unit-test the track-vs-static position selection,
   `_OneEuro` smoothing of a synthetic track, and spec plumbing (detectors themselves marked
   needs_gpu/needs_cv and exercised live). 
5. **Verify on demo1**: a screen-zoom that follows the cursor through the record→render-cut moment;
   an arrow glued to the "record" button; the face-ring tracking the face.

### Reuse (do not rewrite)
`reframe._OneEuro`, `reframe._load_model` + `reframe._detect_center` (YOLO), `timeline.sample_*`
(static fallback), `grab_frame` (seed templates), `_prep_overlays`/`_paste_overlays_gpu` (compose).

## BACKGROUND REPLACEMENT (cutout over an AI/any still image, per time window) (2026-06-18)
Since we already hold the cutout matte, the recorded room can be swapped for a different
background per time window — provider-agnostic: the op only composites; the background image is
made by whatever image tool the user has (for us: Gemini to remove the person + modify the office).
- NEW op `replace_background(project_id, edl_id, segments, output_path, max_duration)` in
  camera_mix.py. `segments`=[{start,end,background}] cut-timeline secs; background = "camera" (real
  room passthrough, no cutout) or a still-image path (cover-fit, cutout composited over it). Reuses
  cached cut_cam + fgr/pha (needs mix_camera(remove_background=True) first); audio from cut screen;
  GPU compose + NVENC (CPU fallback). Helpers `_cover_to` (cover-fit+centre-crop) + `_segment_at`
  (pure, last-covering-segment-wins). Exposed in api (56 ops), GROUP recording, added to _RENDER_OPS
  (async). Documented in mix_camera + learn://recording. 290 py pass, control 100%.
- Background generation (external, our method): grab a camera frame (ffmpeg) → local image-tools
  `gemini_edit_image` (Nano Banana gemini-2.5-flash-image; FlowDot Gemini HTTP toolkit also exists
  but needs base64 INLINE which is impractical to hand-feed a 1080p frame from the agent — works
  from a recipe/hosted-image). Made 3 demo1 office variants: bg_window.png / bg_shelves.png /
  bg_evening.png (person removed, same room + window / bookshelves / night string-lights).
- STATUS: code-complete + asset-ready, NOT yet rendered (needs restart). NEXT: start_render(
  'replace_background', segments=[0-8 camera, 8-20 window, 20-32 shelves, 32-46 evening],
  max_duration=46) → verify frames per segment.

### STATUS — IMPLEMENTED (2026-06-18, code-complete; live verify pending a server restart)
1. **`dopest_clip/obs/tracking.py`** — DONE. `compute_track(video, target, every=, cache_path=)`
   → dense per-frame normalized `[{t,x,y,w,h}]`, smoothed via `reframe._smooth_centers` (reuses
   `_OneEuro`). Detectors: `"cursor"` (cv2 matchTemplate vs `assets/cursor.png`), `"face"` (haar
   cascade), COCO class (`reframe._load_model` + new `reframe._detect_box`, sizes the box),
   `{template_at, region}` (per-frame matchTemplate of the seed crop — robust + always available;
   CSRT noted as a future option). `sample_track`, `apply_track_to_rect` (pure recentre, keeps
   size), `resolve_track` (shared GPU/CPU resolver, cached under `<project>/camera/`), `preview`.
2. **`track` field** — DONE. Rides inside an overlay/blur spec OR on a screen_keyframe/camera
   keyframe (lifted by `camera_mix._split_track`, stripped before normalize). Wired through
   `camera_mix.composite_gpu` + `vertical_clip` (screen-zoom, blur, overlays, camera rect) AND the
   CPU `compositor.compose`/`_build_overlay`. Static keyframes set SIZE; the track sets POSITION.
3. **Surface** — DONE. `preview_track(project_id, target, edl_id=, source=, video_path=, at=)`
   registered (55 ops now). Documented in `learn://recording` (Dynamic tracking) + `learn://reframe`.
4. **Tests** — DONE. MCP control surface still **100%** (api 140/140, server/sidecar/jobs/__main__).
   `tests/test_tracking.py` covers the pure logic + cursor/face/template/yolo detector wiring at
   96% (the 7 misses are defensive weak-match guards + the real-face-found branch, exercised live).
   283 passed / 6 skipped.
5. **Verify on demo1** — DONE (2026-06-18, live via MCP after restart). `assets/cursor.png` created
   (cropped the OS arrow from a demo1 grab_frame at src 251,565,22x24). preview_track proved ALL THREE
   detectors lock, each with a drawn marker frame: `face` (camera, 6005 pts, marker on face), `cursor`
   (screen, 6003 pts, marker on the pointer), `{template_at,region}` (screen UI box, follows it as the
   page scrolls). Then rendered the landscape via start_render('mix_camera', screen_keyframes=[{zoom:1.5,
   track:{target:'cursor'}}]) → demo_track_cursor.mp4 (200s, rvm-gpu/NVENC, matte_cached). Frames at
   t=30/80/160 confirm the 1.5x zoom rides the pointer (panned left → centered → on the button).
   CAMERA-SOURCE OVERLAY MAPPING — WIRED (2026-06-18, code-complete; needs restart to render):
   apply_track_to_rect gained `src_rect` (map camera-normalized coords through the per-frame
   composited camera rect, so a face-ring/bulb lands on the cutout at ANY camera placement —
   PIP/fullscreen/animated) + `offset:[ox,oy]` in tracked-box-size units (the prescriptive
   "where around the target": halo/bulb ABOVE head = offset[0,-0.9] anchored bottom-centre, ring
   AROUND head = offset[0,0] scaled to head, badge to the side = offset[0.9,0]). Wired into
   composite_gpu (passes the per-frame cam rect into _paste_overlays_gpu) + vertical_clip (person
   rect) + CPU compositor (offset; camera src_rect GPU-path). Documented in mix_camera + learn.
   286 py pass, control still 100%. VERIFIED (2026-06-18, live via MCP): rendered demo1
   demo_track_bulb.mp4 — camera as a 0.55 bottom-right PIP + a lightbulb overlay (captest1/
   lightbulb.png, anchor bottom-centre, track {target:'face', source:'camera', offset:[0,-0.85]},
   t_in 10/t_out 60). Frames t=14/48/56 confirm the bulb sits ABOVE the head and rides it on the
   cutout in the corner as the head leans/moves (not stuck at screen-centre) — the camera-rect
   mapping + offset both work. 200s, rvm-gpu/NVENC, matte_cached.

> Earlier v2 polish (deferred, not parity): native file-picker, graphical keyframe-curve editor UI,
> audio/image editor panels, FlowDot provider endpoint specifics.

## Context

`dopest-clip` is a **new, standalone, fully self-contained** open-source media studio
at `E:\FlowdotPlatform\dopest-clip\`. It records (via OBS), edits (transcript-driven
clipper), processes audio, and generates/edits images — drivable by **both** an MCP
agent and a human in an **Electron** editor.

It is OpusClip-shaped but local-first, AI-drivable, and **provider-agnostic**: FlowDot
is one provider option among many, never a lock-in.

**Hard constraint (the reason this is a fresh build):** dopest-clip has **zero runtime
references to the scratch projects**. It does not import, depend on, or call
`obs-mcp`, `short-form-editor-mcp`, `audio-mcp-servers`, or the gemini-image toolkit.
Those four are **reference only** — read to learn from and reimplement, then left
untouched. Everything dopest-clip needs lives inside its own tree.

The four reference projects (READ-ONLY sources of proven logic + hard-won gotchas):
- `E:\FlowdotPlatform\obs-mcp` — Python/FastMCP: OBS track-separated recording (custom
  WS v5 client, Source Record plugin, off-canvas camera), moviepy compositing, RVM/rembg
  matting, keyframe timeline, SVG overlays, animated blur/focus.
- `E:\FlowdotPlatform\short-form-editor-mcp` — Python/FastMCP: EDL data model
  (non-contiguous, silence-snapped), WhisperX + OpenAI STT backends, YOLO11 reframe,
  ASS captions, ffmpeg render, re-STT verify loop, file-based workspace.
- `E:\audio-mcp-servers` — 3 separate TS/Node MCPs: OpenAI `audio_qa`, ElevenLabs SFX,
  Fish `tts/voices/transcribe/account/health` (HTTP clients + LRU disk cache).
- FlowDot `gemini-image` toolkit + local `image-tools` MCP — image gen/edit/compose/
  analyze (BYOK Google) + local image ops.

### Resolved decisions (locked)
1. **One self-contained Python core**; reimplement/copy from references, import nothing.
2. **MCP core + provider registry FIRST**; Electron editor AFTER the core is stable.
3. **v1 providers:** local WhisperX (STT) · OpenAI (Whisper STT + gpt-4o audio-QA) ·
   Fish (TTS/ASR/voice-clone) · ElevenLabs (SFX) · Gemini (image gen/edit/compose/
   analyze) · **FlowDot** (LLM + image + audio aggregator) · **OpenRouter** (LLM only).
4. **Local audio = focused ffmpeg DSP set:** loudnorm, denoise (afftdn/arnndn), trim
   silence, gain, fade, mix, format convert. (No local ML audio in v1.)

## Architecture — one core, two faces

```
                 ┌──────────────────────────────────────┐
                 │            dopest_clip  (CORE)          │
                 │  project/EDL · render/compose · OBS ·  │
                 │  audio DSP · providers · image         │
                 └───────────────┬────────────────────────┘
        ┌────────────────────────┼─────────────────────────┐
   ┌────▼─────┐            ┌──────▼──────┐            ┌──────▼──────┐
   │ MCP server│            │ Electron UI │            │ OBS control │
   │ (FastMCP) │            │  (sidecar)  │            │ (optional)  │
   └───────────┘            └─────────────┘            └─────────────┘
```

- **Core is Python** — forced by the ML/media stack (moviepy, torch, whisperx, YOLO,
  rembg, resvg, ffmpeg); not reimplementable in Node.
- **MCP server** = thin FastMCP adapter exposing core ops as tools + `learn://` docs.
- **Electron** = JS/TS app; main process launches the core as a **localhost
  HTTP/JSON-RPC sidecar**; renderer edits the **same project JSON** the agent edits and
  calls the **same** core ops (one engine, no second implementation).
- **OBS control** = optional recording subsystem; editing works on any input video
  with no OBS installed.

---

## Phase 0 — Project scaffold

Create the self-contained tree and packaging. No reference imports anywhere.

```
dopest-clip/
  pyproject.toml          # package "dopest-clip"; extras: [obs] [stt] [matting] [reframe] [all]
  README.md  LICENSE      # open-source (MIT or Apache-2.0 — decide at write time)
  providers.example.toml  # provider config template
  dopest_clip/
    __init__.py
    config.py             # env + providers.toml loader (no hardcoding, no fallbacks)
    __main__.py           # `python -m dopest_clip` → MCP stdio server
```

`pyproject.toml` extras keep heavy deps optional: `[stt]` = whisperx/torch/openai,
`[matting]` = rembg/onnxruntime + torch, `[reframe]` = ultralytics/opencv, `[obs]` =
obsws-python; base install = ffmpeg-driven editing only.

## Phase 1 — Editing core (project model, EDL, media, render, verify)

Reimplement the proven editing engine as dopest-clip's own code.

- `dopest_clip/project.py` — file-based project store (`projects/<id>/{meta,audio,
  transcript,edls,renders,verify,frames}`). Reference: short-form-editor `workspace.py`.
- `dopest_clip/edl.py` — EDL model + silence-snapped resolution (stable word indices,
  non-contiguous reorderable segments, cleanup + cleanup-warnings). Reference:
  short-form-editor `edl.py` — its model is the design and is reimplemented in spirit
  (this is the editor's data model the Electron UI will render).
- `dopest_clip/media.py` — ffmpeg/ffprobe wrappers: probe, extract_audio, build
  filtergraph, render (concat + crossfade), aspect filter, grab_frame, thumbnail.
  **Carry gotcha:** every ffmpeg/ffprobe spawn uses `stdin=subprocess.DEVNULL` (else it
  hangs the MCP stdio server). Reference: short-form-editor `media.py`.
- `dopest_clip/reframe.py` — YOLO11 subject tracking + One-Euro smoothing + shot
  timeline (track/zoom/focus/full/center/pad/none); aspects 9:16/4:5/1:1/16:9; censor.
  Reference: short-form-editor `reframe.py`. Ships `yolo11n.pt`.
- `dopest_clip/captions.py` — burned ASS captions, 3 presets, word-timed, title card.
  Reference: short-form-editor `captions.py`. Ships `Anton-Regular.ttf`.
- `dopest_clip/verify.py` — re-STT a render + difflib diff vs intended text (QA loop).

Core functions (used by both MCP tools and Electron): `create_project`, `get_project`,
`list_projects`, `get_transcript`, `validate_edl`, `render`, `verify_clip`,
`extract_thumbnail`, `grab_frame`, `preview_reframe`.

## Phase 2 — STT subsystem

`dopest_clip/stt/` with a backend protocol + two backends:
- `whisperx_backend.py` — local WhisperX large-v3 + forced alignment + silence map
  (GPU default, int8 CPU fallback). Reference: short-form-editor `stt/whisperx_backend.py`.
- `openai_backend.py` — OpenAI Whisper-1, 25 MB chunking at silence boundaries.
  **Carry gotcha:** empty `OPENAI_BASE_URL` must be coerced to `None` (scheme-less URL
  bug). Reference: short-form-editor `stt/openai_backend.py`.
- `__init__.py` — `get_backend()` factory + `compute_silences()`.

## Phase 3 — Provider registry (the new spine)

`dopest_clip/providers/` — the provider-management layer that none of the references has.

- `registry.py` — maps `capability → provider → {credential_env, model, base_url}`.
  Capabilities: `llm`, `stt`, `tts`, `sfx`, `audio_qa`, `image`. A `Provider` protocol
  per capability; concrete providers self-register; per-provider `validate()`/`health()`
  (reference: fish-mcp health pattern). Active provider per capability is config-driven.
- Provider modules (reimplemented as Python HTTP clients from the TS references):
  - `openai.py` — STT (Whisper) + audio-QA (gpt-4o-audio). Ref: `audio-qa-mcp-server`.
  - `fish.py` — TTS / ASR / voice CRUD / account / health + LRU disk cache.
    Ref: `fish-mcp-server` (`FishAudioClient`, `AudioCache`).
  - `elevenlabs.py` — sound-generation v2. Ref: `elevenlabs-mcp-server`.
  - `gemini.py` — image generate/edit/compose/analyze (BYOK Google Generative Language
    API). Ref: gemini-image toolkit + local `image-tools` gemini ops.
  - `flowdot.py` — aggregator for `llm` + `image` + audio (FlowDot provider API).
  - `openrouter.py` — `llm` only.
- Config: env vars (`OPENAI_API_KEY`, `ELEVENLABS_API_KEY`, `FISH_AUDIO_API_KEY`,
  `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, `FLOWDOT_*`) + `providers.toml` overrides.
  No hardcoded keys, no silent fallbacks — an unconfigured capability fails loudly.

## Phase 4 — Audio subsystem (local DSP + cloud)

`dopest_clip/audio/`:
- `dsp.py` — ffmpeg DSP core ops + MCP tools: `normalize` (loudnorm
  I=-14:TP=-1.5:LRA=11), `denoise` (afftdn/arnndn), `trim_silence` (silenceremove),
  `gain`, `fade` (afade in/out), `mix` (amix), `convert`. Reuses `media.py`'s runner.
- `tts.py` / `asr.py` / `sfx.py` / `qa.py` — thin core ops that route through the
  provider registry (Fish/ElevenLabs/OpenAI/etc.), persisting outputs into the project.

## Phase 5 — Image subsystem

`dopest_clip/image/`:
- `gen.py` — generate/edit/compose/analyze routed through the registry (Gemini/FlowDot).
- `ops.py` — local ffmpeg/Pillow ops: crop, pad, resize, square_canvas,
  remove_background, recolor, invert, png↔svg (resvg), get_info. Ref: `image-tools` MCP.
  Used for thumbnails, overlays, and title-card art feeding the compositor.

## Phase 6 — OBS recording subsystem (optional)

`dopest_clip/obs/` — reimplemented from obs-mcp (its hard-won correctness is the value):
- `ws.py` — custom obs-websocket v5 client with strict requestId correlation
  (avoids the obsws-python blind-recv race). Ref: obs-mcp `ws.py`.
- `client.py` — scene build (monitor_capture + off-canvas dshow camera + wasapi mic),
  Source Record filter, start/stop, status. **Carry gotchas:** `rate_control="CBR"`
  (uppercase), `record_mode=3`, 6s filter settle, async device-teardown poll, idempotent
  scene reuse (DShow `0x800705AA`). Ref: obs-mcp `obs_client.py`.
- `compositor.py` — moviepy camera-over-screen compose with keyframes/overlays/blur.
- `camera_mix.py` — GPU RVM matting + cut-synced mix into a rendered cut.
- `timeline.py` — keyframe interpolation + easing (shared by camera/overlays/blur).
- `graphics.py` — SVG overlay builders (arrow/ring/box/label) via resvg.
- `blur.py` — animated blur/focus (redact/focus, feather, dim).

Tools: `list_devices`, `setup_scene`, `start_recording`, `stop_recording`,
`recording_status`, `compose_camera`, `mix_camera`, `grab_screen_frame`, `list_graphics`.

## Phase 7 — Unified MCP server + learn docs (v1 agent-face milestone)

- `dopest_clip/server.py` — FastMCP server registering every core op as a tool, grouped:
  recording, editing, audio (DSP + tts/asr/sfx/qa), image, providers
  (`list_providers`, `set_provider`, `validate_provider`).
- `dopest_clip/learn.py` — `learn://` resources: `overview`, `recording`, `editing`,
  `cutting`, `reframe`, `captions`, `audio`, `image`, `providers`, `gotchas`. Written
  fresh for dopest-clip (references: studio_learn + short-form-editor learn for content).
- **Milestone:** `python -m dopest_clip` is a complete, agent-drivable studio.

## Phase 8 — Electron editor (the largest new build)

`dopest-clip/desktop/` (Electron + React + TS):
- Main process launches `python -m dopest_clip --serve` as a **localhost HTTP/JSON-RPC
  sidecar** (a thin HTTP wrapper over the same core ops the MCP server exposes).
- Renderer panels: **timeline** (EDL segments, drag/reorder — non-contiguous is
  first-class), **layers** (screen / camera / overlays / blur), **inspector** (rect
  transform + keyframe curves over `timeline.py`'s model), **transcript**
  (click-to-cut), **preview** (uses `grab_frame`/`preview_reframe`/`extract_thumbnail`/
  `grab_screen_frame`), **provider settings** (registry validate/set).
- The UI only edits the project JSON and calls core ops; render/verify call the same
  `render`/`verify_clip`. No second engine.

---

## Known constraints carried into the build
- ffmpeg/ffprobe spawn with `stdin=DEVNULL` inside the MCP stdio server (else hang).
- OBS path needs OBS 28+ + obs-websocket + the **Source Record** plugin — documented as
  optional; editing works with no OBS.
- Source Record: `rate_control="CBR"` (uppercase), `record_mode=3`, 6s filter settle,
  device-teardown poll, idempotent scene reuse.
- Empty `OPENAI_BASE_URL`/`base_url` → coerce to `None`.
- GPU paths (WhisperX large-v3, RVM, YOLO) need CUDA; provide CPU fallbacks via extras.
- Per project rules: no hardcoding, no silent fallbacks — unconfigured providers and
  missing deps fail loudly with clear errors.

## Verification (end-to-end, per phase)
- **Editing core (P1–P2):** `create_project` → `transcribe` → `validate_edl` →
  `render` → `verify_clip` round-trips on a sample clip (match_ratio ≥ 0.95); confirm
  clean cut + a 9:16 styled output with captions render.
- **Providers (P3):** `list_providers` + `validate_provider` report health per
  configured provider; an unconfigured provider errors clearly (no silent skip).
- **Audio (P4):** normalize/denoise/trim a sample wav (verify via ffprobe); a TTS call
  through Fish and an SFX call through ElevenLabs persist into the project.
- **Image (P5):** Gemini generate + a local crop/resize/remove-background op produce
  files; analyze returns text.
- **OBS (P6, OBS running):** `list_devices` → `setup_scene` → `start/stop_recording`
  yields a separate screen MP4 + camera MKV; `compose_camera` mixes them.
- **MCP server (P7):** `python -m dopest_clip` starts; tools + `learn://` resources list.
- **Electron (P8):** launch app, load sample project, scrub timeline, reorder a segment,
  edit a camera keyframe, render — output matches the agent-rendered file.

## Out of scope for v1
- Local ML audio (diarization, neural enhance) — `arnndn` is the ceiling.
- Publishing/upload integrations.
- Multi-user / cloud sync (file-based project store only).
