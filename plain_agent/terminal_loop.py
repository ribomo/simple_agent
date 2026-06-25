"""Interactive terminal loop for the plain agent."""

from plain_agent.agent_loop import SimpleAgent
from plain_agent.streaming import TextDelta, ToolResult


def approve_run_command(command: str) -> bool:
    """Ask the user whether a requested shell command may run."""
    while True:
        answer = input(f"\nApprove command `{command}`? [y/N] ").strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"", "n", "no"}:
            return False
        print("Please answer y or n.")


def run_interactive_terminal(agent: SimpleAgent) -> None:
    """Read prompts, stream responses, show tool results, and repeat."""
    print("Simple agent client. Type 'exit' to quit.")

    while True:
        try:
            user_input = input("> ").strip()
        except EOFError:
            print()
            break

        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue
        if user_input == "/compact":
            if agent.compact_history():
                print("[conversation compacted]")
                _print_context_size(agent)
            else:
                print("[conversation compact: nothing to compact]")
            print()
            continue

        for event in agent.respond_stream(user_input):
            if isinstance(event, TextDelta):
                print(event.content, end="", flush=True)
            elif isinstance(event, ToolResult):
                status = "ok" if event.ok else "error"
                print(f"\n[tool {event.name}: {status}]")

        print()
        _print_context_size(agent)
        print()


def _print_context_size(agent: SimpleAgent) -> None:
    size = agent.context_size()
    print(
        f"[conversation history: {size.message_count} messages, "
        f"{size.char_count} chars, {size.byte_count} bytes]"
    )
