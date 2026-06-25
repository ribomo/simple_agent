import os

from dotenv import load_dotenv

from plain_agent.agent_loop import SimpleAgent
from plain_agent.compaction import (
    CompactionConfig,
    ConversationCompactor,
)
from plain_agent.llm_client import (
    DEEPSEEK_BASE_URL,
    OPENAI_BASE_URL,
    OpenAICompatibleClient,
)
from plain_agent.terminal_loop import approve_run_command, run_interactive_terminal


def main() -> None:
    load_dotenv()

    provider = os.environ.get("LLM_PROVIDER", "deepseek").lower()
    timeout = float(os.environ.get("LLM_TIMEOUT", "60"))

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
        default_model = "gpt-5.4-mini"
        default_base_url = OPENAI_BASE_URL
    elif provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("LLM_API_KEY")
        default_model = "deepseek-v4-flash"
        default_base_url = DEEPSEEK_BASE_URL
    else:
        raise ValueError("LLM_PROVIDER must be 'openai' or 'deepseek'")

    llm_client = OpenAICompatibleClient(
        api_key=api_key,
        base_url=os.environ.get("LLM_BASE_URL", default_base_url),
        timeout=timeout,
    )
    model = os.environ.get("LLM_MODEL", default_model)
    compactor = ConversationCompactor(
        llm_client=llm_client,
        model=model,
        config=CompactionConfig(
            keep_recent_exchanges=_env_int(
                os.environ.get("LLM_COMPACTION_KEEP_RECENT_EXCHANGES"),
                2,
            ),
            model=os.environ.get("LLM_COMPACTION_MODEL"),
        ),
    )
    agent = SimpleAgent(
        llm_client=llm_client,
        model=model,
        command_approver=approve_run_command,
        compactor=compactor,
        auto_compact_max_tokens=_env_optional_positive_int(
            os.environ.get("LLM_COMPACTION_AUTO_MAX_TOKENS"),
            200_000,
        ),
    )

    run_interactive_terminal(agent)


def _env_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    parsed = int(value)
    if parsed < 0:
        raise ValueError("integer environment settings must be non-negative")
    return parsed


def _env_optional_positive_int(value: str | None, default: int) -> int | None:
    parsed = _env_int(value, default)
    if parsed == 0:
        return None
    return parsed
