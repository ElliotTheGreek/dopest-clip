"""Per-project store: directory layout, id generation, JSON/meta read+write.

Layout (PROJECTS_ROOT/<project_id>/):
    meta.json          probe metadata + source path
    audio.wav          16kHz mono PCM (STT input)
    transcript.json    words[] + silences[]
    transcript.txt     human-readable dialog
    edls/<edl>.json    saved edit-decision lists
    renders/...        clean cut + styled outputs + ass + thumbnails
    verify/<edl>.json  re-STT QA results
    frames/...         grabbed stills / reframe previews

This module is shared by the MCP server and the Electron sidecar; both read/write
the SAME project tree.
"""

import json
import re
import uuid
from pathlib import Path

from . import config


def slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", str(name)).strip("-").lower()
    return s or "project"


def new_project_id(video_path: str) -> str:
    stem = Path(video_path).stem
    return f"{slugify(stem)}-{uuid.uuid4().hex[:8]}"


def project_dir(project_id: str) -> Path:
    return config.PROJECTS_ROOT / project_id


def ensure_project(project_id: str) -> Path:
    root = project_dir(project_id)
    for sub in ("", "edls", "renders", "verify", "frames"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def project_exists(project_id: str) -> bool:
    return meta_path(project_id).exists()


def require_project(project_id: str) -> Path:
    if not project_exists(project_id):
        raise FileNotFoundError(f"project '{project_id}' does not exist (no meta.json)")
    return project_dir(project_id)


def list_project_ids() -> list[str]:
    root = config.PROJECTS_ROOT
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "meta.json").exists())


# --- path helpers ---
def meta_path(project_id: str) -> Path:
    return project_dir(project_id) / "meta.json"


def audio_path(project_id: str) -> Path:
    return project_dir(project_id) / "audio.wav"


def transcript_json_path(project_id: str) -> Path:
    return project_dir(project_id) / "transcript.json"


def transcript_txt_path(project_id: str) -> Path:
    return project_dir(project_id) / "transcript.txt"


def edl_path(project_id: str, edl_id: str) -> Path:
    return project_dir(project_id) / "edls" / f"{slugify(edl_id)}.json"


def render_path(project_id: str, edl_id: str) -> Path:
    return project_dir(project_id) / "renders" / f"{slugify(edl_id)}.mp4"


def filtergraph_path(project_id: str, edl_id: str) -> Path:
    return project_dir(project_id) / "renders" / f"{slugify(edl_id)}.filtergraph.txt"


def _aspect_tag(aspect: str) -> str:
    return aspect.replace(":", "x")


def styled_render_path(project_id: str, edl_id: str, aspect: str) -> Path:
    return project_dir(project_id) / "renders" / f"{slugify(edl_id)}__{_aspect_tag(aspect)}.mp4"


def ass_path(project_id: str, edl_id: str, aspect: str) -> Path:
    return project_dir(project_id) / "renders" / f"{slugify(edl_id)}__{_aspect_tag(aspect)}.ass"


def thumb_path(project_id: str, edl_id: str, aspect: str) -> Path:
    return project_dir(project_id) / "renders" / f"{slugify(edl_id)}__{_aspect_tag(aspect)}.jpg"


def frame_path(project_id: str, name: str) -> Path:
    return project_dir(project_id) / "frames" / f"{slugify(name)}.png"


def verify_path(project_id: str, edl_id: str) -> Path:
    return project_dir(project_id) / "verify" / f"{slugify(edl_id)}.json"


def audio_out_path(project_id: str, name: str, ext: str = "wav") -> Path:
    """Generic output slot for audio subsystem results (tts/sfx/dsp)."""
    d = project_dir(project_id) / "audio"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{slugify(name)}.{ext.lstrip('.')}"


def image_out_path(project_id: str, name: str, ext: str = "png") -> Path:
    """Generic output slot for image subsystem results (gen/ops)."""
    d = project_dir(project_id) / "images"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{slugify(name)}.{ext.lstrip('.')}"


# --- json io ---
def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# --- typed meta / transcript / edl accessors ---
def read_meta(project_id: str) -> dict:
    return read_json(meta_path(project_id))


def write_meta(project_id: str, meta: dict) -> None:
    write_json(meta_path(project_id), meta)


def read_transcript(project_id: str) -> dict:
    p = transcript_json_path(project_id)
    if not p.exists():
        raise FileNotFoundError(f"project '{project_id}' has no transcript — run transcribe first")
    return read_json(p)


def read_edl(project_id: str, edl_id: str) -> dict:
    p = edl_path(project_id, edl_id)
    if not p.exists():
        raise FileNotFoundError(f"edl '{edl_id}' not found in project '{project_id}'")
    return read_json(p)


def save_edl(project_id: str, edl: dict) -> str:
    edl_id = slugify(edl.get("edl_id") or edl.get("title") or "edl")
    edl = dict(edl)
    edl["edl_id"] = edl_id
    write_json(edl_path(project_id, edl_id), edl)
    return edl_id


def list_edl_ids(project_id: str) -> list[str]:
    d = project_dir(project_id) / "edls"
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.json"))
