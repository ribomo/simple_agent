"""Rendering helpers for the interactive terminal UI."""

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

from plain_agent.conversation_history import ContextSize, estimate_token_count
from plain_agent.streaming import ToolResult


class TerminalRenderer:
    """Render assistant output and agent status messages."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console(soft_wrap=True)
        self._assistant_live: Live | None = None
        self._assistant_text = ""

    def print_welcome(self) -> None:
        self.console.print(Text.assemble(("Plain Agent", "bold"), (" Type 'exit' to quit.", "dim")))

    def print_blank_line(self) -> None:
        self.console.file.write("\n")
        self.console.file.flush()

    def start_assistant(self) -> None:
        if self._assistant_live is not None:
            return
        self._assistant_text = ""
        self._assistant_live = Live(
            Markdown(""),
            console=self.console,
            auto_refresh=False,
            transient=False,
        )
        self._assistant_live.start()

    def update_assistant(self, content_delta: str) -> None:
        if self._assistant_live is None:
            self.start_assistant()
        self._assistant_text += content_delta
        if self._assistant_live is not None:
            self._assistant_live.update(self._assistant_markdown(), refresh=True)

    def finish_assistant(self) -> None:
        if self._assistant_live is None:
            return

        has_content = bool(self._assistant_text.strip())
        self._assistant_live.update(self._assistant_markdown(), refresh=True)
        self._assistant_live.stop()
        self._assistant_live = None

        if has_content and not self.console.is_terminal:
            self.print_blank_line()

    def _assistant_markdown(self) -> Markdown:
        return Markdown(self._assistant_text.rstrip())

    def print_tool_result(self, event: ToolResult) -> None:
        status = "ok" if event.ok else "error"
        style = "green" if event.ok else "red"
        self.print_status(f"tool {event.name}", status, style)

    def print_auto_compaction(self, before: ContextSize, after: ContextSize) -> None:
        before_tokens = format_token_count(estimate_token_count(before.char_count))
        after_tokens = format_token_count(estimate_token_count(after.char_count))
        self.print_status(
            "conversation auto-compacted",
            f"~{before_tokens} -> ~{after_tokens} tokens",
            "cyan",
        )

    def print_context_size(self, size: ContextSize) -> None:
        self.print_status(
            "conversation history",
            f"{size.message_count} messages, ~{format_token_count(estimate_token_count(size.char_count))} tokens",
            "dim",
        )

    def print_status(self, label: str, value: str, value_style: str) -> None:
        self.console.print(_status_text(label, value, value_style))


def format_token_count(token_count: int) -> str:
    if token_count >= 1_000:
        return f"{token_count / 1_000:.1f}k"
    return str(token_count)


def _status_text(label: str, value: str, value_style: str) -> Text:
    return Text.assemble(
        ("[", "dim"),
        (label, "dim"),
        (": ", "dim"),
        (value, value_style),
        ("]", "dim"),
    )
