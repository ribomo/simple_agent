import os

from dotenv import load_dotenv

from simple_agent.agent_loop import SimpleAgent
from simple_agent.llm_client import (
    DEEPSEEK_BASE_URL,
    OPENAI_BASE_URL,
    OpenAICompatibleClient,
)
from simple_agent.terminal_loop import approve_run_command, run_interactive_terminal


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
    agent = SimpleAgent(
        llm_client=llm_client,
        model=model,
        command_approver=approve_run_command,
    )

    run_interactive_terminal(agent)
