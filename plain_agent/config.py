"""Typed application settings loaded from environment variables."""

from collections.abc import Mapping
from dataclasses import dataclass, field
import math

from plain_agent.environment import (
    ENV_DEEPSEEK_API_KEY,
    ENV_LLM_API_KEY,
    ENV_LLM_BASE_URL,
    ENV_LLM_COMPACTION_AUTO_MAX_TOKENS,
    ENV_LLM_COMPACTION_KEEP_RECENT_EXCHANGES,
    ENV_LLM_COMPACTION_MODEL,
    ENV_LLM_MODEL,
    ENV_LLM_PROVIDER,
    ENV_LLM_TIMEOUT,
    ENV_OPENAI_API_KEY,
    ENV_PLAIN_AGENT_ENABLE_NETWORK,
)
from plain_agent.providers import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_DEFAULT_MODEL,
    OPENAI_BASE_URL,
    OPENAI_DEFAULT_MODEL,
)


@dataclass(frozen=True)
class LLMSettings:
    """Settings for the OpenAI-compatible language model client."""

    provider: str
    api_key: str | None = field(repr=False)
    base_url: str
    model: str
    timeout: float


@dataclass(frozen=True)
class CompactionSettings:
    """Settings for automatic and manual conversation compaction."""

    keep_recent_exchanges: int
    model: str | None
    auto_max_tokens: int | None


@dataclass(frozen=True)
class NetworkSettings:
    """Settings for optional network tools."""

    enabled: bool


@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration."""

    llm: LLMSettings
    compaction: CompactionSettings
    network: NetworkSettings

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "AppConfig":
        """Build and validate configuration from an environment mapping."""
        return cls(
            llm=_llm_settings(env),
            compaction=CompactionSettings(
                keep_recent_exchanges=_positive_int(
                    env.get(ENV_LLM_COMPACTION_KEEP_RECENT_EXCHANGES),
                    2,
                ),
                model=env.get(ENV_LLM_COMPACTION_MODEL),
                auto_max_tokens=_optional_positive_int(
                    env.get(ENV_LLM_COMPACTION_AUTO_MAX_TOKENS),
                    200_000,
                ),
            ),
            network=NetworkSettings(
                enabled=_bool_value(
                    env.get(ENV_PLAIN_AGENT_ENABLE_NETWORK),
                    True,
                ),
            ),
        )


def _llm_settings(env: Mapping[str, str]) -> LLMSettings:
    provider = env.get(ENV_LLM_PROVIDER, "deepseek").lower()
    if provider == "openai":
        api_key = env.get(ENV_OPENAI_API_KEY) or env.get(ENV_LLM_API_KEY)
        default_model = OPENAI_DEFAULT_MODEL
        default_base_url = OPENAI_BASE_URL
    elif provider == "deepseek":
        api_key = env.get(ENV_DEEPSEEK_API_KEY) or env.get(ENV_LLM_API_KEY)
        default_model = DEEPSEEK_DEFAULT_MODEL
        default_base_url = DEEPSEEK_BASE_URL
    else:
        raise ValueError(f"{ENV_LLM_PROVIDER} must be 'openai' or 'deepseek'")

    return LLMSettings(
        provider=provider,
        api_key=api_key,
        base_url=env.get(ENV_LLM_BASE_URL, default_base_url),
        model=env.get(ENV_LLM_MODEL, default_model),
        timeout=_positive_float(env.get(ENV_LLM_TIMEOUT), 60.0),
    )


def _non_negative_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    parsed = int(value)
    if parsed < 0:
        raise ValueError("integer environment settings must be non-negative")
    return parsed


def _positive_int(value: str | None, default: int) -> int:
    parsed = _non_negative_int(value, default)
    if parsed == 0:
        raise ValueError("integer environment settings must be positive")
    return parsed


def _optional_positive_int(value: str | None, default: int) -> int | None:
    parsed = _non_negative_int(value, default)
    if parsed == 0:
        return None
    return parsed


def _bool_value(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("boolean environment settings must be true or false")


def _positive_float(value: str | None, default: float) -> float:
    parsed = default if value is None else float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError("floating-point environment settings must be finite and positive")
    return parsed
