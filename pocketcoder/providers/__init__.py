"""
LLM Providers module.

Contains implementations for different LLM backends:

LOCAL (no API key required):
- Ollama (native API) - ollama
- OpenAI-compatible (vLLM, LM Studio, llama.cpp, etc.) - openai_compat

CLOUD (requires API key):
- OpenAI (GPT-4, GPT-3.5, etc.) - openai
- Anthropic (Claude 3.5, etc.) - anthropic
"""

from pocketcoder.providers.base import BaseProvider
from pocketcoder.providers.ollama import OllamaProvider
from pocketcoder.providers.openai_compat import OpenAICompatProvider
from pocketcoder.providers.openai import OpenAIProvider
from pocketcoder.providers.anthropic import AnthropicProvider


def create_provider(config: dict) -> BaseProvider:
    """
    Factory function to create a provider from config.

    Provider types:
    - "ollama": Local Ollama server (default)
    - "openai_compat": Any OpenAI-compatible API (vLLM, LM Studio, etc.)
    - "openai": Official OpenAI API (requires OPENAI_API_KEY)
    - "anthropic": Anthropic Claude API (requires ANTHROPIC_API_KEY)

    Example config:
        {"type": "ollama", "default_model": "qwen2.5-coder:7b"}
        {"type": "openai", "api_key": "sk-...", "default_model": "gpt-4o"}
        {"type": "anthropic", "default_model": "claude-sonnet-4-20250514"}
    """
    provider_type = config.get("type", "ollama")

    if provider_type == "ollama":
        return OllamaProvider(
            base_url=config.get("base_url", "http://localhost:11434"),
            default_model=config.get("default_model", "qwen2.5-coder:7b"),
        )
    elif provider_type == "openai_compat":
        return OpenAICompatProvider(
            base_url=config.get("base_url", "http://localhost:8000/v1"),
            default_model=config.get("default_model", ""),
            api_key=config.get("api_key"),
        )
    elif provider_type == "openai":
        return OpenAIProvider(
            api_key=config.get("api_key"),
            default_model=config.get("default_model", "gpt-4o-mini"),
        )
    elif provider_type == "anthropic":
        return AnthropicProvider(
            api_key=config.get("api_key"),
            default_model=config.get("default_model", "claude-sonnet-4-20250514"),
        )
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")


__all__ = [
    "BaseProvider",
    "OllamaProvider",
    "OpenAICompatProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "create_provider",
]
