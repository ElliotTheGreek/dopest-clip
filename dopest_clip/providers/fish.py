"""Fish Audio provider: TTS, ASR, voice CRUD, account, health (+ LRU disk cache).

Ported from E:\\audio-mcp-servers\\fish-mcp-server\\src\\client\\FishAudioClient.ts and
AudioCache.ts:
    base    https://api.fish.audio   (overridable via FISH_AUDIO_BASE_URL)
    auth    Authorization: Bearer <key>
    tts     POST /v1/tts             header model: speech-1.6, body { text, ... } -> audio bytes
    asr     POST /v1/asr             multipart { audio, [language] } -> { text, segments }
    voices  GET/POST/DELETE /model   (list/get/create/delete)
    account GET /wallet/self/api-credit , /wallet/self/package
    health  GET /model?page_size=1

The LRU disk cache is ported faithfully: cache key = sha256 over (text, voice, fmt,
prosody) — carried gotcha — capped by file count and total bytes, evicting the
least-recently-accessed entry first. Cache survives across calls (and process restarts,
since it rehydrates entries from the cache dir on first use).

Key: FISH_AUDIO_API_KEY, read at call time. Uses `requests`.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import requests

from .. import config
from .registry import Provider, env_str

_DEFAULT_BASE = "https://api.fish.audio"
_TTS_MODEL = "speech-1.6"


def _base_url() -> str:
    return env_str("FISH_AUDIO_BASE_URL") or _DEFAULT_BASE


def _cache_dir() -> Path:
    env = env_str("FISH_AUDIO_CACHE_DIR")
    if env:
        return Path(env)
    return config.PROJECTS_ROOT.parent / ".cache" / "fish-audio"


class _LRUDiskCache:
    """LRU disk cache, ported from AudioCache.ts.

    Bounds: maxFiles (default 50) and maxSizeBytes (default 500MB). On set(), evicts the
    least-recently-accessed entry until both bounds hold. Files are named <key>.<format>.
    A reload() scans the dir so a fresh process sees previously-written files (the TS
    version started empty each run; we rehydrate so the cache is actually useful in a
    long-lived sidecar). Best-effort: filesystem errors degrade to a cache miss, never a
    crash of the TTS path.
    """

    def __init__(self, cache_dir: Path, max_files: int = 50, max_size_mb: int = 500):
        self.cache_dir = Path(cache_dir)
        self.max_files = max_files
        self.max_size_bytes = max_size_mb * 1024 * 1024
        # key -> {"path": Path, "size": int, "last": float}
        self._entries: dict[str, dict] = {}
        self._size = 0
        self._loaded = False

    @staticmethod
    def generate_key(text: str, voice, fmt, prosody) -> str:
        """sha256 over the salient TTS params. Carried gotcha: the key must include
        text, voice, fmt and prosody so a prosody/voice change busts the cache."""
        payload = json.dumps(
            {"text": text, "voice": voice, "format": fmt, "prosody": prosody},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            for child in self.cache_dir.iterdir():
                if not child.is_file():
                    continue
                key = child.stem
                try:
                    st = child.stat()
                except OSError:
                    continue
                self._entries[key] = {"path": child, "size": st.st_size, "last": st.st_mtime}
                self._size += st.st_size
        except OSError:
            pass

    def get(self, key: str) -> bytes | None:
        self._ensure_loaded()
        entry = self._entries.get(key)
        if not entry:
            return None
        try:
            data = entry["path"].read_bytes()
        except OSError:
            self._entries.pop(key, None)
            return None
        entry["last"] = time.time()
        return data

    def set(self, key: str, data: bytes, fmt: str) -> Path:
        self._ensure_loaded()
        self._evict_if_needed(len(data))
        path = self.cache_dir / f"{key}.{fmt}"
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        except OSError:
            # Cannot cache; return the intended path anyway, caller already has bytes.
            return path
        self._entries[key] = {"path": path, "size": len(data), "last": time.time()}
        self._size += len(data)
        return path

    def _evict_if_needed(self, incoming: int) -> None:
        while self._entries and (
            len(self._entries) >= self.max_files
            or self._size + incoming > self.max_size_bytes
        ):
            oldest_key = min(self._entries, key=lambda k: self._entries[k]["last"])
            entry = self._entries.pop(oldest_key)
            self._size -= entry["size"]
            try:
                entry["path"].unlink()
            except OSError:
                pass


class FishProvider(Provider):
    name = "fish"
    capabilities = ("tts", "stt")  # stt == ASR

    def __init__(self):
        self._cache = _LRUDiskCache(_cache_dir())

    def validate(self) -> dict:
        if env_str("FISH_AUDIO_API_KEY"):
            return {"ok": True, "detail": "FISH_AUDIO_API_KEY present"}
        return {"ok": False, "detail": "FISH_AUDIO_API_KEY not set"}

    def _key(self) -> str:
        key = env_str("FISH_AUDIO_API_KEY")
        if not key:
            raise RuntimeError("FISH_AUDIO_API_KEY is not set")
        return key

    def _auth(self) -> dict:
        return {"Authorization": f"Bearer {self._key()}"}

    # --- tts ---
    def tts(self, text: str, voice: str | None = None, fmt: str = "mp3", **opts) -> bytes:
        prosody = opts.get("prosody")
        cache_key = self._cache.generate_key(text, voice, fmt, prosody)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        body: dict = {"text": text, "format": fmt}
        if voice:
            body["reference_id"] = voice
        if opts.get("sample_rate") is not None:
            body["sample_rate"] = opts["sample_rate"]
        if opts.get("bitrate") is not None:
            body["mp3_bitrate"] = opts["bitrate"]
        if opts.get("temperature") is not None:
            body["temperature"] = opts["temperature"]
        if opts.get("top_p") is not None:
            body["top_p"] = opts["top_p"]
        if opts.get("latency") is not None:
            body["latency"] = opts["latency"]
        if opts.get("normalize") is not None:
            body["normalize"] = opts["normalize"]
        if opts.get("references"):
            body["references"] = opts["references"]
        if prosody is not None:
            body["prosody"] = prosody

        resp = requests.post(
            f"{_base_url()}/v1/tts",
            headers={**self._auth(), "Content-Type": "application/json", "model": _TTS_MODEL},
            json=body,
            timeout=300,
        )
        if not resp.ok:
            raise RuntimeError(f"Fish Audio {resp.status_code}: {resp.text}")
        data = resp.content
        self._cache.set(cache_key, data, fmt)
        return data

    # --- asr (stt) ---
    def asr(self, audio_path_or_bytes, **opts) -> dict:
        if isinstance(audio_path_or_bytes, (bytes, bytearray)):
            audio = bytes(audio_path_or_bytes)
        else:
            with open(audio_path_or_bytes, "rb") as fh:
                audio = fh.read()

        files = {"audio": ("audio.wav", audio)}
        data = {}
        if opts.get("language"):
            data["language"] = opts["language"]

        resp = requests.post(
            f"{_base_url()}/v1/asr",
            headers=self._auth(),
            files=files,
            data=data,
            timeout=300,
        )
        if not resp.ok:
            raise RuntimeError(f"Fish Audio {resp.status_code}: {resp.text}")
        body = resp.json()
        return {"text": body.get("text", ""), "segments": body.get("segments")}

    # alias so this can also serve the registry "stt" capability uniformly
    def transcribe(self, audio_path_or_bytes, **opts) -> dict:
        return self.asr(audio_path_or_bytes, **opts)

    # --- voice CRUD ---
    def list_voices(self, **params) -> dict:
        query = {}
        if params.get("page"):
            query["page_number"] = params["page"]
        if params.get("page_size"):
            query["page_size"] = params["page_size"]
        if params.get("language"):
            query["language"] = params["language"]
        if params.get("title"):
            query["title"] = params["title"]
        if params.get("sort_by"):
            query["sort_by"] = params["sort_by"]
        if params.get("self") is not None:
            query["self"] = "true" if params["self"] else "false"
        resp = requests.get(
            f"{_base_url()}/model", headers=self._auth(), params=query or None, timeout=60
        )
        if not resp.ok:
            raise RuntimeError(f"Fish Audio {resp.status_code}: {resp.text}")
        return resp.json()

    def get_voice(self, voice_id: str) -> dict:
        resp = requests.get(f"{_base_url()}/model/{voice_id}", headers=self._auth(), timeout=60)
        if not resp.ok:
            raise RuntimeError(f"Fish Audio {resp.status_code}: {resp.text}")
        return resp.json()

    def create_voice(self, title: str, audio_samples: list, **opts) -> dict:
        body = {"title": title, "audio_samples": audio_samples}
        for k in ("description", "language", "tags"):
            if opts.get(k) is not None:
                body[k] = opts[k]
        resp = requests.post(
            f"{_base_url()}/model",
            headers={**self._auth(), "Content-Type": "application/json"},
            json=body,
            timeout=120,
        )
        if not resp.ok:
            raise RuntimeError(f"Fish Audio {resp.status_code}: {resp.text}")
        return resp.json()

    def delete_voice(self, voice_id: str) -> None:
        resp = requests.delete(f"{_base_url()}/model/{voice_id}", headers=self._auth(), timeout=60)
        if not resp.ok:
            raise RuntimeError(f"Fish Audio {resp.status_code}: {resp.text}")

    # --- account ---
    def account(self) -> dict:
        credits = requests.get(
            f"{_base_url()}/wallet/self/api-credit", headers=self._auth(), timeout=60
        )
        package = requests.get(
            f"{_base_url()}/wallet/self/package", headers=self._auth(), timeout=60
        )
        out: dict = {}
        if credits.ok:
            out["credits"] = credits.json()
        else:
            out["credits_error"] = f"{credits.status_code}: {credits.text}"
        if package.ok:
            out["package"] = package.json()
        else:
            out["package_error"] = f"{package.status_code}: {package.text}"
        return out

    # --- health ---
    def health(self) -> dict:
        start = time.time()
        try:
            resp = requests.get(
                f"{_base_url()}/model",
                headers=self._auth(),
                params={"page_size": 1},
                timeout=30,
            )
        except requests.RequestException as e:
            return {"status": "unhealthy", "api_reachable": False, "auth_valid": False,
                    "error": str(e)}
        latency = int((time.time() - start) * 1000)
        if resp.ok:
            return {"status": "healthy", "api_reachable": True, "auth_valid": True,
                    "latency_ms": latency}
        auth_valid = resp.status_code not in (401, 403)
        return {
            "status": "unhealthy" if not auth_valid else "degraded",
            "api_reachable": True,
            "auth_valid": auth_valid,
            "latency_ms": latency,
            "error": f"{resp.status_code}: {resp.text}",
        }


def register_into(register) -> None:
    factory = FishProvider
    register("tts", "fish", factory)
    register("stt", "fish", factory)
