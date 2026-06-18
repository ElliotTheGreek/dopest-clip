"""Runtime configuration, read from environment with documented defaults.

Read once at import. The MCP server is registered with the relevant env vars; the
Electron sidecar inherits the same environment. Provider credentials are read by
the provider registry (see dopest_clip.providers), not here — this module owns the
media/editing knobs and the shared filesystem layout.

Project rule: no hardcoded secrets, no silent fallbacks. A missing required value
surfaces as a loud error at the point of use, not a quiet default that hides it.
"""

import os
from pathlib import Path

# --- Speech-to-text ---
# Default backend for transcribe/verify: "whisperx" (local, GPU) or "openai" (cloud).
STT_BACKEND = os.environ.get("STT_BACKEND", "whisperx").strip().lower()
WHISPERX_MODEL = os.environ.get("WHISPERX_MODEL", "large-v3").strip()
DEVICE = os.environ.get("DEVICE", "cuda").strip()                      # "cuda" or "cpu"
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()

# --- Reframe (portrait subject tracking) ---
REFRAME_MODEL = os.environ.get("REFRAME_MODEL", "yolo11n.pt").strip()
REFRAME_CONF = float(os.environ.get("REFRAME_CONF", "0.4"))
REFRAME_SAMPLE_EVERY = int(os.environ.get("REFRAME_SAMPLE_EVERY", "2"))
REFRAME_MINCUTOFF = float(os.environ.get("REFRAME_MINCUTOFF", "0.22"))
REFRAME_BETA = float(os.environ.get("REFRAME_BETA", "0.018"))
REFRAME_DEADZONE = float(os.environ.get("REFRAME_DEADZONE", "0.06"))   # fraction of frame width
REFRAME_MAX_PAN = float(os.environ.get("REFRAME_MAX_PAN", "0.5"))      # fraction of frame width / second
REFRAME_TRANSITION_S = float(os.environ.get("REFRAME_TRANSITION_S", "0.5"))
REFRAME_BAND_BLUR = int(os.environ.get("REFRAME_BAND_BLUR", "39"))     # odd kernel for band blur

# --- Cutting / snapping ---
MIN_SILENCE = float(os.environ.get("MIN_SILENCE", "0.15"))   # min gap (s) to count as a clean boundary
SNAP_MARGIN = float(os.environ.get("SNAP_MARGIN", "0.08"))   # how far into the silence to place the cut
CROSSFADE_MS = int(os.environ.get("CROSSFADE_MS", "15"))     # per-join audio fade (ms); 0 disables

# --- Filesystem layout ---
# All projects live under PROJECTS_ROOT/<project_id>/{meta,audio,transcript,edls,renders,verify,frames}.
_DEFAULT_PROJECTS = Path(__file__).resolve().parent.parent / "projects"
PROJECTS_ROOT = Path(os.environ.get("DOPEST_PROJECTS_ROOT", str(_DEFAULT_PROJECTS)))

# Bundled assets shipped inside the package (fonts, yolo weights live alongside).
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"

# Optional providers.toml that overrides per-capability provider selection + models.
# Read by dopest_clip.providers.registry; default location is next to PROJECTS_ROOT's parent.
PROVIDERS_TOML = Path(os.environ.get("DOPEST_PROVIDERS_TOML", str(Path(__file__).resolve().parent.parent / "providers.toml")))

# --- External tools ---
FFMPEG = os.environ.get("FFMPEG_BIN", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE_BIN", "ffprobe")
