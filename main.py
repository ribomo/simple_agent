import os

from dotenv import load_dotenv

from simple_agent.agent import SimpleAgent
from simple_agent.llm_client import (
    DEEPSEEK_BASE_URL,
    OPENAI_BASE_URL,
    OpenAICompatibleClient,
)
from simple_agent.streaming import TextDelta, ToolResult


def approve_run_command(command: str) -> bool:
    while True:
        answer = input(f"\nApprove command `{command}`? [y/N] ").strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"", "n", "no"}:
            return False
        print("Please answer y or n.")


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

    print("Simple agent client. Type 'exit' to quit.")

    while True:
        user_input = input("> ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        for event in agent.respond_stream(user_input):
            if isinstance(event, TextDelta):
                # Chat messages may arrive in multiple chunks
                # so we print them without a newline and flush after each chunk.
                print(event.content, end="", flush=True)
            elif isinstance(event, ToolResult):
                status = "ok" if event.ok else "error"
                # print tool call results
                print(f"\n[tool {event.name}: {status}]")
        print()


if __name__ == "__main__":
    main()
