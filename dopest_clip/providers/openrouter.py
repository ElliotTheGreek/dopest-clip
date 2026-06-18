"""OpenRouter provider: LLM only.

OpenRouter exposes an OpenAI-compatible chat-completions endpoint, so the request
shape mirrors openai.py — only the base URL, the API key env var, and a couple of
optional attribution headers differ.

    base    https://openrouter.ai/api/v1   (overridable via OPENROUTER_BASE_URL)
    auth    Authorization: Bearer <OPENROUTER_API_KEY>
    POST    /chat/completions  body { model, messages, ... } -> OpenAI-shaped response

Key: OPENROUTER_API_KEY, read at call time. Uses `requests`.
"""

from __future__ import annotations

import requests

from .registry import Provider, env_str

_DEFAULT_BASE = "https://openrouter.ai/api/v1"


def _base_url() -> str:
    return env_str("OPENROUTER_BASE_URL") or _DEFAULT_BASE


class OpenRouterProvider(Provider):
    name = "openrouter"
    capabilities = ("llm",)

    def validate(self) -> dict:
        if env_str("OPENROUTER_API_KEY"):
            return {"ok": True, "detail": "OPENROUTER_API_KEY present"}
        return {"ok": False, "detail": "OPENROUTER_API_KEY not set"}

    def _key(self) -> str:
        key = env_str("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        return key

    def complete(self, messages, model: str, **opts) -> dict:
        payload = {"model": model, "messages": messages}
        for k in ("temperature", "top_p", "max_tokens", "stop", "response_format"):
            if k in opts and opts[k] is not None:
                payload[k] = opts[k]

        headers = {
            "Authorization": f"Bearer {self._key()}",
            "Content-Type": "application/json",
        }
        # Optional OpenRouter attribution headers (no-op if unset).
        referer = env_str("OPENROUTER_REFERER")
        if referer:
            headers["HTTP-Referer"] = referer
        title = env_str("OPENROUTER_TITLE")
        if title:
            headers["X-Title"] = title

        resp = requests.post(
            f"{_base_url()}/chat/completions", headers=headers, json=payload, timeout=300
        )
        if not resp.ok:
            raise RuntimeError(f"OpenRouter {resp.status_code}: {resp.text}")
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return {"text": "(no response)"}
        return {"text": (choices[0].get("message") or {}).get("content") or "(no response)"}


def register_into(register) -> None:
    register("llm", "openrouter", OpenRouterProvider)
