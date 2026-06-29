"""Interactive terminal loop for the plain agent."""
import sys

from plain_agent.agent_loop import SimpleAgent
from plain_agent.sandbox import CommandRequest
from plain_agent.streaming import AutoCompaction, TextDelta, ToolResult
from plain_agent.ui.terminal_renderer import TerminalRenderer


def approve_run_command(request: CommandRequest) -> bool:
    """Ask the user whether a requested sandboxed command may run."""
    while True:
        answer = input(
            f"\nApprove [{request.mode.value}] command `{request.display}`? [y/N] "
        ).strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"", "n", "no"}:
            return False
        print("Please answer y or n.")


def run_interactive_terminal(agent: SimpleAgent, renderer: TerminalRenderer | None = None) -> None:
    """Read prompts, stream responses, show tool results, and repeat."""
    renderer = renderer or TerminalRenderer()
    if renderer.console.is_terminal and sys.stdin.isatty():
        from plain_agent.ui.textual_terminal import run_textual_terminal

        run_textual_terminal(agent)
        return

    _run_basic_interactive_terminal(agent, renderer)


def _run_basic_interactive_terminal(agent: SimpleAgent, renderer: TerminalRenderer) -> None:
    """Run the simple line-oriented terminal loop."""
    renderer.print_welcome()
    for warning in agent.startup_warnings:
        renderer.print_status("warning", warning, "yellow")

    while True:
        try:
            user_input = input("> ").strip()
        except EOFError:
            renderer.print_blank_line()
            break

        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue
        if user_input == "/compact":
            if agent.compact_history():
                renderer.print_status("conversation", "compacted", "green")
                renderer.print_context_size(agent.context_size())
            else:
                renderer.print_status("conversation compact", "nothing to compact", "yellow")
            renderer.print_blank_line()
            continue

        try:
            for event in agent.respond_stream(user_input):
                if isinstance(event, TextDelta):
                    renderer.update_assistant(event.content)
                elif isinstance(event, ToolResult):
                    renderer.finish_assistant()
                    renderer.print_tool_result(event)
                elif isinstance(event, AutoCompaction):
                    renderer.print_auto_compaction(event.before, event.after)
        finally:
            renderer.finish_assistant()

        renderer.print_blank_line()
        renderer.print_context_size(agent.context_size())
        renderer.print_blank_line()
