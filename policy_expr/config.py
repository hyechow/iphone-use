"""Configuration helpers for policy_expr."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from llm.provider_config import ChatProviderConfig, resolve_chat_provider_config

CONFIG_PATH = Path(__file__).with_name("config.yaml")


@lru_cache(maxsize=1)
def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def resolve_llm_config(name: str) -> ChatProviderConfig:
    """Resolve LLM config for policy, validator, or output."""

    llm_config = load_config().get("llm", {})
    section = llm_config.get(name, {}) if isinstance(llm_config, dict) else {}
    if not isinstance(section, dict):
        raise ValueError(f"policy_expr config llm.{name} must be a mapping")
    return resolve_chat_provider_config(
        provider=_optional_str(section.get("provider")),
        model=_optional_str(section.get("model")),
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None
