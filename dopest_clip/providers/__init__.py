"""Provider registry: the swappable spine that keeps dopest-clip from being locked
to any single vendor.

Every paid/cloud capability the studio uses — LLM, speech-to-text, text-to-speech,
sound effects, audio QA, image gen/edit — is expressed as an abstract *capability*.
Concrete providers (OpenAI, Fish Audio, ElevenLabs, Gemini, FlowDot, OpenRouter)
register the capabilities they implement, and the registry resolves which provider
is active for each capability based on, in order:

    1. code defaults,
    2. environment variables (DOPEST_PROVIDER_<CAP>),
    3. config.PROVIDERS_TOML.

Nothing here imports heavy deps. Provider modules use `requests` (a base dependency)
for HTTP and read their API keys from the environment *at call time*, never at import
— so importing this package, listing providers, and calling validate() all work in a
key-less, network-less test venv without raising.

Public surface:

    from dopest_clip.providers import registry
    registry.get("tts").tts("hello")
    registry.list_providers()
    registry.set_provider("llm", "openrouter")
"""

from .registry import (
    CAPABILITIES,
    Provider,
    ProviderError,
    registry,
)

__all__ = ["CAPABILITIES", "Provider", "ProviderError", "registry"]
