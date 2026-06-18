"""Gemini image provider (BYOK Google Generative Language API).

Capabilities: image — generate (text->image), edit (image+instruction->image),
compose (2+ images+instruction->image), analyze (image+instruction->text).

Endpoint (v1beta REST):
    POST https://generativelanguage.googleapis.com/v1beta/models/<model>:generateContent?key=<KEY>
    body { contents: [{ parts: [...] }], generationConfig: { responseModalities: [...] } }

For image *output* (generate/edit/compose) we request responseModalities ["IMAGE"]
(plus "TEXT" so models that insist on emitting text don't error) and pull the first
inlineData part's base64 back out as bytes. For analyze we request ["TEXT"] and read the
text parts. Input images are sent as inlineData parts (base64 + mimeType).

The model id is always supplied by the caller so it stays current (the contract passes
`model` explicitly to every method). Key: GEMINI_API_KEY, read at call time. Uses
`requests`.
"""

from __future__ import annotations

import base64

import requests

from .registry import Provider, env_str

_BASE = "https://generativelanguage.googleapis.com/v1beta"


def _detect_mime(image_bytes: bytes) -> str:
    """Best-effort image mime sniff from magic bytes; defaults to png.

    Google accepts png/jpeg/webp/gif/heic. We only need to label the inlineData part
    correctly; if unknown we fall back to png (the most common generator output).
    """
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if image_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    if image_bytes[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return "image/png"


def _inline_part(image_bytes: bytes) -> dict:
    return {
        "inlineData": {
            "mimeType": _detect_mime(image_bytes),
            "data": base64.b64encode(image_bytes).decode("ascii"),
        }
    }


class GeminiProvider(Provider):
    name = "gemini"
    capabilities = ("image",)

    def validate(self) -> dict:
        if env_str("GEMINI_API_KEY"):
            return {"ok": True, "detail": "GEMINI_API_KEY present"}
        return {"ok": False, "detail": "GEMINI_API_KEY not set"}

    def _key(self) -> str:
        key = env_str("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        return key

    def _generate(self, model: str, parts: list, want_image: bool) -> dict:
        modalities = ["IMAGE", "TEXT"] if want_image else ["TEXT"]
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {"responseModalities": modalities},
        }
        resp = requests.post(
            f"{_BASE}/models/{model}:generateContent",
            params={"key": self._key()},
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=300,
        )
        if not resp.ok:
            raise RuntimeError(f"Gemini {resp.status_code}: {resp.text}")
        return resp.json()

    @staticmethod
    def _extract_image(data: dict) -> bytes:
        for cand in data.get("candidates") or []:
            for part in (cand.get("content") or {}).get("parts") or []:
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and inline.get("data"):
                    return base64.b64decode(inline["data"])
        raise RuntimeError("Gemini returned no image data in response")

    @staticmethod
    def _extract_text(data: dict) -> str:
        chunks: list[str] = []
        for cand in data.get("candidates") or []:
            for part in (cand.get("content") or {}).get("parts") or []:
                if isinstance(part.get("text"), str):
                    chunks.append(part["text"])
        return "".join(chunks) if chunks else "(no response)"

    # --- image capability methods ---
    def generate_image(self, prompt: str, model: str, **opts) -> bytes:
        data = self._generate(model, [{"text": prompt}], want_image=True)
        return self._extract_image(data)

    def edit_image(self, image_bytes: bytes, instruction: str, model: str, **opts) -> bytes:
        parts = [{"text": instruction}, _inline_part(image_bytes)]
        data = self._generate(model, parts, want_image=True)
        return self._extract_image(data)

    def compose_images(self, image_bytes_list: list, instruction: str, model: str, **opts) -> bytes:
        if not image_bytes_list:
            raise ValueError("compose_images requires at least one image")
        parts: list = [{"text": instruction}]
        parts.extend(_inline_part(b) for b in image_bytes_list)
        data = self._generate(model, parts, want_image=True)
        return self._extract_image(data)

    def analyze_image(self, image_bytes: bytes, instruction: str, model: str, **opts) -> dict:
        parts = [{"text": instruction}, _inline_part(image_bytes)]
        data = self._generate(model, parts, want_image=False)
        return {"text": self._extract_text(data)}


def register_into(register) -> None:
    register("image", "gemini", GeminiProvider)
