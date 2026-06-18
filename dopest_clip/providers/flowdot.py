"""FlowDot provider (aggregator).

FlowDot is the platform dopest-clip can optionally lean on instead of wiring every
vendor key individually — it fronts LLM, image, and audio capabilities behind one
account. The point of this module is to register FlowDot as an *option* for several
capabilities so users who already have a FlowDot key can route through it.

!!! UNVERIFIED ENDPOINTS — READ THIS !!!
At the time of writing this module I could NOT verify the exact FlowDot provider HTTP
API (path, auth scheme, request/response JSON) from the references available in this
package. Per project rules I do NOT invent endpoints silently:

  * LLM is implemented against a configurable, OpenAI-compatible base URL
    (FLOWDOT_BASE_URL) because FlowDot's model gateway is OpenAI-shaped in the
    reference material I do have. If FLOWDOT_BASE_URL is not set, complete() raises a
    clear error telling the operator to set it — it does NOT guess a hostname.
    TODO(verify): confirm FlowDot's exact chat endpoint path + auth header and pin the
    default base URL here once known.

  * IMAGE and AUDIO (tts/sfx/audio_qa) raise NotImplementedError with a message
    pointing at this TODO. They are registered so they appear in list_providers() (so
    the UI can show "FlowDot — not yet implemented" rather than hiding it), but calling
    them fails loudly instead of pretending.
    TODO(verify): implement once the FlowDot image + audio endpoints are confirmed.

Key: FLOWDOT_API_KEY, read at call time. Base URL: FLOWDOT_BASE_URL (no default —
empty/unset is treated as unset). Uses `requests`.
"""

from __future__ import annotations

import requests

from .registry import Provider, env_str

# Capabilities FlowDot is registered for. Only "llm" is implemented (against a
# configurable OpenAI-compatible base URL); the rest raise NotImplementedError.
_CAPS = ("llm", "image", "tts", "sfx", "audio_qa")

_NOT_IMPLEMENTED = (
    "FlowDot {cap} is not implemented yet: the exact FlowDot provider endpoint for this "
    "capability is unverified. See the TODO in dopest_clip/providers/flowdot.py. "
    "Select a different provider for {cap} via registry.set_provider({cap!r}, <name>)."
)


def _base_url() -> str | None:
    # No default: we will not invent a FlowDot hostname.
    return env_str("FLOWDOT_BASE_URL")


class FlowDotProvider(Provider):
    name = "flowdot"
    capabilities = _CAPS

    def validate(self) -> dict:
        key = env_str("FLOWDOT_API_KEY")
        base = _base_url()
        if not key:
            return {"ok": False, "detail": "FLOWDOT_API_KEY not set"}
        if not base:
            return {
                "ok": False,
                "detail": "FLOWDOT_BASE_URL not set (FlowDot endpoint is unverified; "
                          "set FLOWDOT_BASE_URL to an OpenAI-compatible gateway)",
            }
        return {"ok": True, "detail": "FLOWDOT_API_KEY and FLOWDOT_BASE_URL present"}

    def _key(self) -> str:
        key = env_str("FLOWDOT_API_KEY")
        if not key:
            raise RuntimeError("FLOWDOT_API_KEY is not set")
        return key

    # --- llm (OpenAI-compatible against FLOWDOT_BASE_URL) ---
    def complete(self, messages, model: str, **opts) -> dict:
        base = _base_url()
        if not base:
            raise RuntimeError(
                "FlowDot LLM requires FLOWDOT_BASE_URL to be set to an OpenAI-compatible "
                "chat-completions base (the FlowDot endpoint is unverified — see TODO in "
                "providers/flowdot.py). Refusing to guess a hostname."
            )
        payload = {"model": model, "messages": messages}
        for k in ("temperature", "top_p", "max_tokens", "stop", "response_format"):
            if k in opts and opts[k] is not None:
                payload[k] = opts[k]
        resp = requests.post(
            f"{base.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._key()}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=300,
        )
        if not resp.ok:
            raise RuntimeError(f"FlowDot {resp.status_code}: {resp.text}")
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return {"text": "(no response)"}
        return {"text": (choices[0].get("message") or {}).get("content") or "(no response)"}

    # --- image (unverified) ---
    def generate_image(self, prompt: str, model: str, **opts) -> bytes:
        raise NotImplementedError(_NOT_IMPLEMENTED.format(cap="image"))

    def edit_image(self, image_bytes: bytes, instruction: str, model: str, **opts) -> bytes:
        raise NotImplementedError(_NOT_IMPLEMENTED.format(cap="image"))

    def compose_images(self, image_bytes_list, instruction: str, model: str, **opts) -> bytes:
        raise NotImplementedError(_NOT_IMPLEMENTED.format(cap="image"))

    def analyze_image(self, image_bytes: bytes, instruction: str, model: str, **opts) -> dict:
        raise NotImplementedError(_NOT_IMPLEMENTED.format(cap="image"))

    # --- audio (unverified) ---
    def tts(self, text: str, voice=None, fmt: str = "mp3", **opts) -> bytes:
        raise NotImplementedError(_NOT_IMPLEMENTED.format(cap="tts"))

    def sfx(self, prompt: str, duration=None, **opts) -> bytes:
        raise NotImplementedError(_NOT_IMPLEMENTED.format(cap="sfx"))

    def audio_qa(self, audio_path, prompt=None, model=None) -> dict:
        raise NotImplementedError(_NOT_IMPLEMENTED.format(cap="audio_qa"))


def register_into(register) -> None:
    factory = FlowDotProvider
    for cap in _CAPS:
        register(cap, "flowdot", factory)
