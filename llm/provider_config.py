"""Shared API provider resolution for ChatQwen clients."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

SUPPORTED_CHAT_PROVIDERS = ("modelscope", "dashscope", "nvidia", "openai", "local")

_PROVIDER_ENV_MAP = {
    "modelscope": {
        "model": "MODELSCOPE_MODEL",
        "api_key": "MODELSCOPE_API_KEY",
        "base_url": "MODELSCOPE_BASE_URL",
    },
    "dashscope": {
        "model": "DASHSCOPE_MODEL",
        "api_key": "DASHSCOPE_API_KEY",
        "base_url": "DASHSCOPE_BASE_URL",
    },
    "nvidia": {
        "model": "NVIDIA_MODEL",
        "api_key": "NVIDIA_API_KEY",
        "base_url": "NVIDIA_BASE_URL",
    },
    "openai": {
        "model": "OPENAI_MODEL",
        "api_key": "OPENAI_API_KEY",
        "base_url": "OPENAI_BASE_URL",
    },
    "local": {
        "model": "LOCAL_MODEL",
        "api_key": "LOCAL_API_KEY",
        "base_url": "LOCAL_BASE_URL",
    },
}

DEFAULT_MODEL_BY_PROVIDER = {
    "modelscope": "Qwen/Qwen2.5-72B-Instruct",
    "dashscope": "qwen-plus",
    "nvidia": "minimaxai/minimax-m2.1",
    "openai": "gpt-4o-mini",
    "local": "Qwen/Qwen3-8B",
}

DEFAULT_BASE_URL_BY_PROVIDER = {
    "modelscope": "https://api-inference.modelscope.cn/v1",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "openai": "https://api.openai.com/v1",
    "local": "http://localhost:30000/v1",
}

DEFAULT_API_KEY_BY_PROVIDER = {
    "modelscope": "dummy",
    "local": "dummy",
}


@dataclass(frozen=True)
class ChatProviderConfig:
    provider: str
    model: str
    api_key: Optional[str]
    base_url: str


def resolve_chat_provider_config(
    provider: Optional[str] = None,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    default_models: Optional[Mapping[str, str]] = None,
    default_provider: str = "modelscope",
) -> ChatProviderConfig:
    """Resolve provider/model/api_key/base_url from overrides and environment."""
    resolved_provider = str(provider or os.getenv("API_PROVIDER", default_provider)).lower()
    if resolved_provider not in _PROVIDER_ENV_MAP:
        supported = ", ".join(SUPPORTED_CHAT_PROVIDERS)
        raise ValueError(f"Unsupported API_PROVIDER: {resolved_provider}. Expected one of: {supported}")

    env_keys = _PROVIDER_ENV_MAP[resolved_provider]
    models: Dict[str, str] = dict(DEFAULT_MODEL_BY_PROVIDER)
    if default_models:
        models.update(default_models)

    resolved_model = model or os.getenv(env_keys["model"], models[resolved_provider])
    resolved_api_key = api_key or os.getenv(env_keys["api_key"], DEFAULT_API_KEY_BY_PROVIDER.get(resolved_provider))
    resolved_base_url = base_url or os.getenv(env_keys["base_url"], DEFAULT_BASE_URL_BY_PROVIDER[resolved_provider])

    return ChatProviderConfig(
        provider=resolved_provider,
        model=resolved_model,
        api_key=resolved_api_key,
        base_url=resolved_base_url,
    )
