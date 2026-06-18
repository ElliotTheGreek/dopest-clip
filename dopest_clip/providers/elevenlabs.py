"""ElevenLabs provider: sound-effects generation (sfx capability).

Ported from E:\\audio-mcp-servers\\elevenlabs-mcp-server\\src\\index.ts:
    POST https://api.elevenlabs.io/v1/sound-generation?output_format=...
    header  xi-api-key: <key>          Accept: audio/mpeg
    body    { text, model_id, [duration_seconds], [prompt_influence], [loop] }
    model   eleven_text_to_sound_v2

Returns raw mp3 bytes (the registry contract for sfx is bytes, not a file path — the
audio subsystem decides where to write).

Key: ELEVENLABS_API_KEY, read at call time. Uses `requests`.
"""

from __future__ import annotations

import urllib.parse

import requests

from .registry import Provider, env_str

_URL = "https://api.elevenlabs.io/v1/sound-generation"
_DEFAULT_MODEL = "eleven_text_to_sound_v2"
_DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"


class ElevenLabsProvider(Provider):
    name = "elevenlabs"
    capabilities = ("sfx",)

    def validate(self) -> dict:
        if env_str("ELEVENLABS_API_KEY"):
            return {"ok": True, "detail": "ELEVENLABS_API_KEY present"}
        return {"ok": False, "detail": "ELEVENLABS_API_KEY not set"}

    def _key(self) -> str:
        key = env_str("ELEVENLABS_API_KEY")
        if not key:
            raise RuntimeError("ELEVENLABS_API_KEY is not set")
        return key

    def sfx(self, prompt: str, duration: float | None = None, **opts) -> bytes:
        output_format = opts.get("output_format") or _DEFAULT_OUTPUT_FORMAT
        url = f"{_URL}?output_format={urllib.parse.quote(output_format)}"

        body: dict = {
            "text": prompt,
            "model_id": opts.get("model_id") or _DEFAULT_MODEL,
        }
        if duration is not None:
            body["duration_seconds"] = duration
        if opts.get("prompt_influence") is not None:
            body["prompt_influence"] = opts["prompt_influence"]
        if opts.get("loop") is not None:
            body["loop"] = opts["loop"]

        resp = requests.post(
            url,
            headers={
                "xi-api-key": self._key(),
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            json=body,
            timeout=300,
        )
        if not resp.ok:
            raise RuntimeError(f"ElevenLabs {resp.status_code}: {resp.text}")
        return resp.content


def register_into(register) -> None:
    register("sfx", "elevenlabs", ElevenLabsProvider)
