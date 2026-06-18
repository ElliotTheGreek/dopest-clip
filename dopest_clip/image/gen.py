"""Provider-routed image generation/edit/compose/analyze.

These functions are the agent-facing wrappers around the registry's "image" capability
(default provider: gemini, BYOK). The active provider is resolved via
`registry.get("image")`; resolution + a missing-key error are handled by the registry —
we never fall back silently.

The image-output methods (generate/edit/compose) return raw bytes; we persist them to
disk and return paths. analyze returns text only. Output destination is:
  * the explicit `out` path if given, else
  * `project.image_out_path(project_id, <name>, <ext>)` when `project_id` is given, else
  * a ValueError (we will not invent a location).

The model id is always passed by the caller so it stays current. The Gemini provider
*requires* a model; we surface a clear error if it is missing rather than letting a
None reach the wire.
"""

from __future__ import annotations

from pathlib import Path

from .. import project
from ..providers import registry

# Image bytes -> file extension, sniffed from magic bytes so the persisted file gets a
# truthful suffix regardless of what the provider returned.
_DEFAULT_EXT = "png"


def _require_model(model: str) -> str:
    if not model or not str(model).strip():
        raise ValueError(
            "an image model id is required (the caller passes `model` so it stays "
            "current, e.g. 'gemini-2.5-flash-image'); none was given"
        )
    return str(model).strip()


def _sniff_ext(image_bytes: bytes) -> str:
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if image_bytes[:3] == b"\xff\xd8\xff":
        return "jpg"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "webp"
    if image_bytes[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    return _DEFAULT_EXT


def _resolve_out(out, project_id, name: str, image_bytes: bytes) -> Path:
    if out is not None:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    if project_id is not None:
        return project.image_out_path(project_id, name, _sniff_ext(image_bytes))
    raise ValueError(
        "no output location: pass either `out=<path>` or `project_id=<id>` so the "
        "result can be persisted (no default location is invented)"
    )


def _read_bytes(image_path) -> bytes:
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"input image not found: {p}")
    return p.read_bytes()


def _persist(image_bytes: bytes, out: Path) -> None:
    if not image_bytes:
        raise RuntimeError("provider returned empty image bytes")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(image_bytes)


def generate(prompt: str, model: str, project_id=None, out=None, **opts) -> dict:
    """Text -> image. Persists the result and returns {out, provider, model}."""
    model = _require_model(model)
    provider = registry.get("image")
    image_bytes = provider.generate_image(prompt, model=model, **opts)
    dest = _resolve_out(out, project_id, "generated", image_bytes)
    _persist(image_bytes, dest)
    return {"out": str(dest), "provider": provider.name, "model": model}


def edit(image_path, instruction: str, model: str, project_id=None, out=None, **opts) -> dict:
    """Image + instruction -> image. Persists the result and returns {out, provider, model}."""
    model = _require_model(model)
    src_bytes = _read_bytes(image_path)
    provider = registry.get("image")
    image_bytes = provider.edit_image(src_bytes, instruction, model=model, **opts)
    dest = _resolve_out(out, project_id, "edited", image_bytes)
    _persist(image_bytes, dest)
    return {"out": str(dest), "provider": provider.name, "model": model}


def compose(image_paths, instruction: str, model: str, project_id=None, out=None, **opts) -> dict:
    """Two or more images + instruction -> image. Returns {out, provider, model}."""
    model = _require_model(model)
    if not image_paths:
        raise ValueError("compose requires at least one input image path")
    image_bytes_list = [_read_bytes(p) for p in image_paths]
    provider = registry.get("image")
    image_bytes = provider.compose_images(image_bytes_list, instruction, model=model, **opts)
    dest = _resolve_out(out, project_id, "composed", image_bytes)
    _persist(image_bytes, dest)
    return {"out": str(dest), "provider": provider.name, "model": model}


def analyze(image_path, instruction: str, model: str, **opts) -> dict:
    """Image + instruction -> {"text": str}. No file is written."""
    model = _require_model(model)
    src_bytes = _read_bytes(image_path)
    provider = registry.get("image")
    result = provider.analyze_image(src_bytes, instruction, model=model, **opts)
    # Provider contract returns {"text": str}; pass it through, guarding the shape.
    if not isinstance(result, dict) or "text" not in result:
        raise RuntimeError(f"image provider analyze returned unexpected shape: {result!r}")
    return {"text": result["text"]}
