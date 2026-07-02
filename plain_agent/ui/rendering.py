"""Rich renderables for the terminal UI."""

from rich.text import Text

from plain_agent.conversation_history import ContextSize, estimate_token_count
from plain_agent.display import escape_display_text
from plain_agent.streaming import AutoCompaction, ToolResult
from plain_agent.tools.permissions.network_permission import NetworkPermissionRequest
from plain_agent.tools.permissions.request import CommandPermissionRequest

WELCOME_STYLE = "#7dd3c7"
USER_PROMPT_STYLE = "bold #8bd5ff"
STATUS_LABEL_STYLE = "#6f7a88"
STATUS_BRACKET_STYLE = "#4f5b68"
APPROVAL_STYLE = "bold #f2c572"


def format_token_count(token_count: int) -> str:
    if token_count >= 1_000:
        return f"{token_count / 1_000:.1f}k"
    return str(token_count)


def format_welcome() -> Text:
    return Text.assemble(
        ("Plain Agent", f"bold {WELCOME_STYLE}"),
        (" Type 'exit' to quit.", STATUS_LABEL_STYLE),
    )


def format_context_size(size: ContextSize) -> Text:
    return status_text(
        "context",
        f"~{format_token_count(estimate_token_count(size.char_count))} tokens",
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


def format_command_approval(request: CommandPermissionRequest) -> Text:
    command = request.command
    return Text(
        f"[approval required: {command.mode.value}] {command.display}\n"
        f"[reason] {escape_display_text(request.justification)}",
        style=APPROVAL_STYLE,
    )


def format_network_approval(request: NetworkPermissionRequest) -> Text:
    return Text(
        f"[approval required: network] {escape_display_text(request.tool)} "
        f"-> {escape_display_text(request.destination)}\n"
        f"[target] {request.display}",
        style=APPROVAL_STYLE,
    )


def status_text(label: str, value: str, value_style: str) -> Text:
    return Text.assemble(
        ("[", STATUS_BRACKET_STYLE),
        (label, STATUS_LABEL_STYLE),
        (": ", STATUS_BRACKET_STYLE),
        (value, value_style),
        ("]", STATUS_BRACKET_STYLE),
    )
