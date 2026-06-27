"""Rich renderables for the Textual terminal UI."""

from rich.text import Text

from plain_agent.conversation_history import ContextSize, estimate_token_count
from plain_agent.streaming import AutoCompaction, ToolResult
from plain_agent.ui.terminal_renderer import format_token_count

WELCOME_STYLE = "#7dd3c7"
USER_PROMPT_STYLE = "bold #8bd5ff"
STATUS_LABEL_STYLE = "#6f7a88"
STATUS_BRACKET_STYLE = "#4f5b68"
APPROVAL_STYLE = "bold #f2c572"


def format_welcome() -> Text:
    return Text.assemble(
        ("Plain Agent", f"bold {WELCOME_STYLE}"),
        (" Type 'exit' to quit.", STATUS_LABEL_STYLE),
    )


def format_context_size(size: ContextSize) -> Text:
    return status_text(
        "conversation history",
        f"{size.message_count} messages, ~{format_token_count(estimate_token_count(size.char_count))} tokens",
        "dim",
    )


def format_tool_result(event: ToolResult) -> Text:
    status = "ok" if event.ok else "error"
    style = "bold #8fbc8f" if event.ok else "bold #ff7b7b"
    return status_text(f"tool {event.name}", status, style)


def format_auto_compaction(event: AutoCompaction) -> Text:
    before_tokens = format_token_count(estimate_token_count(event.before.char_count))
    after_tokens = format_token_count(estimate_token_count(event.after.char_count))
    return status_text(
        "conversation auto-compacted",
        f"~{before_tokens} -> ~{after_tokens} tokens",
        "#7dd3c7",
    )


def status_text(label: str, value: str, value_style: str) -> Text:
    return Text.assemble(
        ("[", STATUS_BRACKET_STYLE),
        (label, STATUS_LABEL_STYLE),
        (": ", STATUS_BRACKET_STYLE),
        (value, value_style),
        ("]", STATUS_BRACKET_STYLE),
    )
