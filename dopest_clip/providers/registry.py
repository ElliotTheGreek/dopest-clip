"""The capability/provider registry.

A *capability* is something the studio needs done (an LLM completion, a TTS render,
an image edit). A *provider* is a concrete vendor implementation that supports one or
more capabilities. The registry tracks, per capability, which providers are available
and which one is currently active, and resolves a usable provider on demand.

Resolution order for the active provider of a capability:
    1. an in-memory selection made via set_provider(),
    2. config.PROVIDERS_TOML  ([active] table, key = capability),
    3. environment variable   DOPEST_PROVIDER_<CAP>  (e.g. DOPEST_PROVIDER_LLM),
    4. the code default        (DEFAULT_ACTIVE),
    5. otherwise, the first registered provider for that capability.

Selecting a provider does NOT require it to be configured (have a key) — selection is
just preference. get() is where a missing key turns into a loud error: it returns the
active provider only if that provider validates ok, otherwise it raises ProviderError
with a clear, actionable message. There are no silent fallbacks.
"""

from __future__ import annotations

import os
from typing import Callable

from .. import config

# TOML reading: stdlib tomllib on 3.11+, else the tomli backport if installed. The
# project targets >=3.10, where tomllib does not exist; tomli is present in the dev/test
# venv. If neither is importable we degrade to "no TOML override" (env + code defaults
# still work) rather than crashing the whole provider subsystem on import.
try:
    import tomllib as _tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - depends on interpreter version
    try:
        import tomli as _tomllib  # type: ignore
    except ModuleNotFoundError:
        _tomllib = None  # type: ignore

# The full set of capabilities the studio can route. STT lives in dopest_clip.stt for
# the local whisperx path, but the *cloud* STT (OpenAI Whisper / Fish ASR) is exposed
# here too so the registry is the single place to ask "who does X".
CAPABILITIES: tuple[str, ...] = ("llm", "stt", "tts", "sfx", "audio_qa", "image")

# Code-default active provider per capability. Chosen to be the free/local-friendly or
# most-broadly-available option; always overridable. A capability may map to a provider
# that is unconfigured by default — that is fine, it only errors at get() time.
DEFAULT_ACTIVE: dict[str, str] = {
    "llm": "openai",
    # Cloud STT exposed through the registry is Fish ASR. The OpenAI Whisper path lives
    # in dopest_clip.stt (local/whisperx + openai backends) and is selected there via
    # config.STT_BACKEND — it is NOT a registry provider, so it must not be the default
    # here or get("stt") would resolve to an unregistered name.
    "stt": "fish",
    "tts": "fish",
    "sfx": "elevenlabs",
    "audio_qa": "openai",
    "image": "gemini",
}


class ProviderError(RuntimeError):
    """Raised when a capability cannot be served — no provider selected, the selected
    provider is unknown, or the selected provider is not configured."""


def env_str(name: str) -> str | None:
    """Read an env var, treating empty/whitespace-only as absent.

    GOTCHA carried from the TS servers: an env var set to "" (common when a launcher
    exports every key whether or not it has a value) must behave as *unset*, not as a
    valid empty base URL. Returns None for missing or blank.
    """
    v = os.environ.get(name)
    if v is None:
        return None
    v = v.strip()
    return v or None


class Provider:
    """Base class. Subclasses set `name`, declare `capabilities`, and implement the
    capability methods they support.

    validate() MUST NOT raise — it reports configuration status as a dict so the UI /
    `list_providers` can show what is and isn't ready without crashing on a missing key.
    """

    name: str = "base"
    capabilities: tuple[str, ...] = ()

    def validate(self) -> dict:
        """Return {"ok": bool, "detail": str}. Never raises.

        Default implementation reports ok iff the provider's primary key env var is
        present. Subclasses override for richer checks (but must keep the no-raise
        contract).
        """
        return {"ok": False, "detail": "validate() not implemented"}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Provider {self.name} caps={','.join(self.capabilities)}>"


# Factory registry: capability -> {provider_name: zero-arg constructor}. Constructors
# are lazy so importing a provider module's heavy bits (none here, but keeps the door
# closed) and instantiating clients only happens on demand.
_FACTORIES: dict[str, dict[str, Callable[[], Provider]]] = {cap: {} for cap in CAPABILITIES}
# Cache of constructed singletons by (capability, name) so repeated get() calls reuse
# the same client (and its disk cache, in Fish's case).
_INSTANCES: dict[tuple[str, str], Provider] = {}
# In-memory active selection overrides (highest priority).
_ACTIVE_OVERRIDE: dict[str, str] = {}


def register(capability: str, name: str, factory: Callable[[], Provider]) -> None:
    """Register a provider factory for a capability. Idempotent per (capability, name)."""
    if capability not in CAPABILITIES:
        raise ValueError(f"unknown capability {capability!r}; expected one of {CAPABILITIES}")
    _FACTORIES[capability][name] = factory


def _instance(capability: str, name: str) -> Provider:
    key = (capability, name)
    inst = _INSTANCES.get(key)
    if inst is None:
        inst = _FACTORIES[capability][name]()
        _INSTANCES[key] = inst
    return inst


def _read_toml() -> dict:
    """Read PROVIDERS_TOML if it exists and is parseable, else {}. Never raises."""
    if _tomllib is None:
        return {}
    path = config.PROVIDERS_TOML
    try:
        if not path.exists():
            return {}
        with open(path, "rb") as fh:
            return _tomllib.load(fh)
    except (OSError, _tomllib.TOMLDecodeError):
        return {}


def _active_name(capability: str) -> str:
    """Resolve the active provider *name* for a capability (selection, not validation)."""
    if capability in _ACTIVE_OVERRIDE:
        return _ACTIVE_OVERRIDE[capability]

    toml = _read_toml()
    active_tbl = toml.get("active") or {}
    if isinstance(active_tbl, dict) and capability in active_tbl:
        return str(active_tbl[capability])

    env = env_str(f"DOPEST_PROVIDER_{capability.upper()}")
    if env:
        return env

    if capability in DEFAULT_ACTIVE:
        return DEFAULT_ACTIVE[capability]

    names = list(_FACTORIES.get(capability, {}).keys())
    if names:
        return names[0]
    raise ProviderError(
        f"no provider registered for capability {capability!r}; "
        f"available capabilities: {', '.join(CAPABILITIES)}"
    )


def get(capability: str) -> Provider:
    """Return the active, *configured* provider for a capability, or raise ProviderError.

    Raises when: the capability is unknown, the selected provider name isn't registered,
    or the selected provider's validate() reports not-ok (e.g. missing API key). The
    error message names the provider and the missing key so the fix is obvious. No
    silent fallback to another provider — selection is explicit.
    """
    if capability not in CAPABILITIES:
        raise ProviderError(
            f"unknown capability {capability!r}; expected one of {', '.join(CAPABILITIES)}"
        )

    name = _active_name(capability)
    factories = _FACTORIES.get(capability, {})
    if name not in factories:
        available = ", ".join(sorted(factories)) or "(none registered)"
        raise ProviderError(
            f"capability {capability!r} is set to provider {name!r}, which is not "
            f"registered. Registered providers for {capability!r}: {available}."
        )

    provider = _instance(capability, name)
    status = provider.validate()
    if not status.get("ok"):
        detail = status.get("detail", "not configured")
        raise ProviderError(
            f"provider {name!r} for capability {capability!r} is not usable: {detail}. "
            f"Configure it (set its API key env var) or select another provider with "
            f"registry.set_provider({capability!r}, <name>)."
        )
    return provider


def set_provider(capability: str, provider_name: str) -> None:
    """Select `provider_name` as active for `capability`.

    Records the selection in memory (effective immediately) and, if PROVIDERS_TOML's
    location is writable, persists it to the [active] table so the choice survives a
    restart. Selecting an unregistered provider raises so typos surface immediately;
    selecting an unconfigured (key-less) one is allowed — that only errors at get().
    """
    if capability not in CAPABILITIES:
        raise ProviderError(
            f"unknown capability {capability!r}; expected one of {', '.join(CAPABILITIES)}"
        )
    factories = _FACTORIES.get(capability, {})
    if provider_name not in factories:
        available = ", ".join(sorted(factories)) or "(none registered)"
        raise ProviderError(
            f"cannot select unknown provider {provider_name!r} for {capability!r}. "
            f"Registered: {available}."
        )
    _ACTIVE_OVERRIDE[capability] = provider_name
    _persist_active(capability, provider_name)


def _persist_active(capability: str, provider_name: str) -> None:
    """Write the active selection into PROVIDERS_TOML's [active] table.

    `toml` is not a dependency in the test venv, so we hand-roll a tiny writer that
    only knows how to emit a flat [active] string table (the one thing we persist).
    Best-effort: if the path isn't writable, the in-memory selection still stands and
    we do not raise — persistence is a convenience, not a guarantee.
    """
    path = config.PROVIDERS_TOML
    toml = _read_toml()
    active = toml.get("active")
    if not isinstance(active, dict):
        active = {}
    active = dict(active)
    active[capability] = provider_name
    toml["active"] = active
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_dump_toml(toml))
    except OSError:
        # Not writable (read-only fs, permissions). In-memory override remains valid.
        pass


def _dump_toml(data: dict) -> str:
    """Minimal TOML writer. Supports a flat top-level of string/number/bool scalars and
    one level of string-keyed tables of scalars — which is all the registry persists.
    Not a general TOML serializer; intentionally tiny and dependency-free.
    """
    def fmt_scalar(v) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return str(v)
        s = str(v).replace("\\", "\\\\").replace('"', '\\"')
        return f'"{s}"'

    lines: list[str] = []
    # top-level scalars first
    for k, v in data.items():
        if not isinstance(v, dict):
            lines.append(f"{k} = {fmt_scalar(v)}")
    # then tables
    for k, v in data.items():
        if isinstance(v, dict):
            lines.append("")
            lines.append(f"[{k}]")
            for sk, sv in v.items():
                lines.append(f"{sk} = {fmt_scalar(sv)}")
    return "\n".join(lines).lstrip("\n") + "\n"


def list_providers() -> dict:
    """Describe the full capability/provider map.

    Returns, per capability:
        {
          "active": <name resolved by precedence>,
          "providers": {
             <name>: {"configured": bool, "detail": str, "active": bool},
             ...
          }
        }

    `configured` reflects validate().ok (key present & shape valid). This is the data a
    settings UI renders. It never raises and never hits the network for the providers
    whose validate() is offline (all of ours are — they only check env presence/shape).
    """
    out: dict = {}
    for cap in CAPABILITIES:
        factories = _FACTORIES.get(cap, {})
        try:
            active = _active_name(cap)
        except ProviderError:
            active = None
        providers: dict = {}
        for name in sorted(factories):
            status = _instance(cap, name).validate()
            providers[name] = {
                "configured": bool(status.get("ok")),
                "detail": status.get("detail", ""),
                "active": (name == active),
            }
        out[cap] = {"active": active, "providers": providers}
    return out


def _reset_for_tests() -> None:
    """Clear in-memory selection + instance cache. Test helper only."""
    _ACTIVE_OVERRIDE.clear()
    _INSTANCES.clear()


# --- Wire up the built-in providers -----------------------------------------------
# Importing each module registers its factories via register(). Kept at the bottom so
# the registration API above is fully defined first. These imports are cheap (stdlib +
# requests only) and never touch the network or require keys.
from . import openai as _openai          # noqa: E402
from . import fish as _fish              # noqa: E402
from . import elevenlabs as _elevenlabs  # noqa: E402
from . import gemini as _gemini          # noqa: E402
from . import flowdot as _flowdot        # noqa: E402
from . import openrouter as _openrouter  # noqa: E402

_openai.register_into(register)
_fish.register_into(register)
_elevenlabs.register_into(register)
_gemini.register_into(register)
_flowdot.register_into(register)
_openrouter.register_into(register)


class _RegistryFacade:
    """Tiny object exposing the module-level functions as methods, so callers can do
    `registry.get(...)` per the contract."""

    get = staticmethod(get)
    set_provider = staticmethod(set_provider)
    list_providers = staticmethod(list_providers)
    register = staticmethod(register)
    env_str = staticmethod(env_str)
    CAPABILITIES = CAPABILITIES
    ProviderError = ProviderError
    Provider = Provider
    DEFAULT_ACTIVE = DEFAULT_ACTIVE

    def _reset_for_tests(self):  # pragma: no cover - test helper
        _reset_for_tests()


registry = _RegistryFacade()
