"""Textual application for the full-screen terminal UI."""

import os
import threading
from collections.abc import Callable

# Disable Kitty keyboard handling before importing Textual; on affected
# terminals it can introduce a noticeable delay between key presses.
os.environ.setdefault("TEXTUAL_DISABLE_KITTY_KEY", "1")

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import Input, Static

from plain_agent.agent_loop import SimpleAgent
from plain_agent.streaming import AutoCompaction, TextDelta, ToolResult
from plain_agent.ui.textual_terminal.approval import PendingApproval, parse_approval_answer
from plain_agent.ui.textual_terminal.rendering import (
    APPROVAL_STYLE,
    USER_PROMPT_STYLE,
    format_auto_compaction,
    format_context_size,
    format_tool_result,
    format_welcome,
    status_text,
)
from plain_agent.ui.textual_terminal.transcript import TextualTranscript


class TranscriptView(VerticalScroll):
    """Scrollable transcript that can be selected without stealing click focus."""

    FOCUS_ON_CLICK = False


class PromptInput(Input):
    """Input that highlights the surrounding prompt row while focused."""

    def action_copy(self) -> None:
        """Prefer an active transcript selection over an old prompt selection."""
        selected_text = self.screen.get_selected_text()
        if selected_text is not None:
            self.app.copy_to_clipboard(selected_text)
        else:
            super().action_copy()

    def on_focus(self) -> None:
        self.app._set_prompt_active(True)

    def on_blur(self) -> None:
        self.app._set_prompt_active(False)


class PlainAgentTextualApp(App[None]):
    """Full-screen Textual UI for Plain Agent."""

    CSS_PATH = "terminal.tcss"
    BINDINGS = [("ctrl+d", "quit", "Quit")]

    def __init__(self, agent: SimpleAgent) -> None:
        super().__init__()
        self.agent = agent
        self.transcript: TextualTranscript
        self.status: Static
        self.prompt_row: Horizontal
        self.prompt_label: Static
        self.prompt_input: PromptInput
        self._responding = False
        self._compacting = False
        self._pending_approval: PendingApproval | None = None
        self._old_command_approver = agent.command_approver

    def compose(self) -> ComposeResult:
        yield TranscriptView(id="transcript")
        with Container(id="bottom-dock"):
            with Horizontal(id="prompt-row"):
                yield Static("> ", id="prompt-label")
                yield PromptInput(id="prompt-input", compact=True)
            yield Static("", id="status")

    def on_mount(self) -> None:
        self.transcript = TextualTranscript(self.query_one("#transcript", VerticalScroll))
        self.status = self.query_one("#status", Static)
        self.prompt_row = self.query_one("#prompt-row", Horizontal)
        self.prompt_label = self.query_one("#prompt-label", Static)
        self.prompt_input = self.query_one("#prompt-input", PromptInput)
        self._set_status("")
        self.transcript.append(format_welcome())
        self.prompt_input.focus()
        self.agent.command_approver = self._approve_run_command

    async def on_unmount(self) -> None:
        if self._pending_approval is not None:
            self._pending_approval.done.set()
        if hasattr(self, "transcript"):
            await self.transcript.finish_assistant()
        self.agent.command_approver = self._old_command_approver

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self._submit_prompt(event.value.strip())

    def _submit_prompt(self, text: str) -> None:
        if self._pending_approval is not None:
            self.prompt_input.value = ""
            self._submit_approval(text)
            return

        if not text:
            self.prompt_input.value = ""
            return
        if self._responding:
            self._set_status("Assistant is still responding.")
            return
        if self._compacting:
            self._set_status("Conversation is still being compacted.")
            return

        self.prompt_input.value = ""
        if text.lower() in {"exit", "quit"}:
            self.exit()
            return
        if text == "/compact":
            self._compact_history()
            return

        self._responding = True
        self._set_status("Assistant is responding...")
        self.transcript.append(Text(f"> {text}", style=USER_PROMPT_STYLE))
        self._start_worker(self._handle_agent_events, text)

    def _submit_approval(self, answer: str) -> None:
        approved = parse_approval_answer(answer)
        if approved is None:
            self._set_status("Please answer y or n.")
            return

        self._resolve_approval(approved)
        self._set_status("")

    def _compact_history(self) -> None:
        self._compacting = True
        self._set_status("Compacting conversation...")
        self._start_worker(self._handle_compaction)

    def _handle_compaction(self) -> None:
        try:
            compacted = self.agent.compact_history()
        except Exception as exc:
            self._call_from_agent_thread(self._finish_compaction, False, _format_exception(exc))
        else:
            self._call_from_agent_thread(self._finish_compaction, compacted, None)

    def _finish_compaction(self, compacted: bool, error_message: str | None) -> None:
        self._compacting = False
        if error_message is not None:
            self.transcript.append(status_text("conversation compact error", error_message, "bold red"))
            self._set_status("Conversation compaction failed.")
        elif compacted:
            self.transcript.append(status_text("conversation", "compacted", "green"))
            self.transcript.append(format_context_size(self.agent.context_size()))
            self._set_status("")
        else:
            self.transcript.append(status_text("conversation compact", "nothing to compact", "yellow"))
            self._set_status("")

    def _handle_agent_events(self, user_input: str) -> None:
        error_message: str | None = None
        try:
            for event in self.agent.respond_stream(user_input):
                if isinstance(event, TextDelta):
                    self._call_from_agent_thread(self.transcript.update_assistant, event.content)
                elif isinstance(event, ToolResult):
                    self._call_from_agent_thread(self.transcript.finish_assistant)
                    self._call_from_agent_thread(self.transcript.append, format_tool_result(event))
                elif isinstance(event, AutoCompaction):
                    self._call_from_agent_thread(self.transcript.finish_assistant)
                    self._call_from_agent_thread(self.transcript.append, format_auto_compaction(event))
        except Exception as exc:
            error_message = _format_exception(exc)
        finally:
            self._call_from_agent_thread(self._finish_response, error_message)

    def _start_worker(self, target: Callable[..., None], *args: object) -> None:
        threading.Thread(target=target, args=args, daemon=True).start()

    def _approve_run_command(self, command: str) -> bool:
        pending = PendingApproval()

        async def ask() -> None:
            await self.transcript.finish_assistant()
            self._pending_approval = pending
            self.prompt_input.value = ""
            self.transcript.append(Text(f"[approval required] {command}", style=APPROVAL_STYLE))
            self.prompt_label.update("Approve command? [y/N] ")
            self._set_status("Command approval required")
            self.prompt_input.focus()

        try:
            self.call_from_thread(ask)
        except RuntimeError:
            return False
        pending.done.wait()
        return pending.approved

    def _call_from_agent_thread(self, callback: Callable[..., object], *args: object) -> None:
        try:
            self.call_from_thread(callback, *args)
        except RuntimeError:
            return

    def _resolve_approval(self, approved: bool) -> None:
        pending = self._pending_approval
        if pending is None:
            return
        pending.approved = approved
        self._pending_approval = None
        self.prompt_label.update("> ")
        pending.done.set()

    async def _finish_response(self, error_message: str | None) -> None:
        await self.transcript.finish_assistant()
        if error_message is None:
            self.transcript.append(format_context_size(self.agent.context_size()))
            self._set_status("")
        else:
            self.transcript.append(status_text("assistant error", error_message, "bold red"))
            self._set_status("Assistant response failed.")
        self._responding = False

    def _set_status(self, text: str) -> None:
        self.status.update(text.strip())

    def _set_prompt_active(self, active: bool) -> None:
        prompt_row = getattr(self, "prompt_row", None)
        if prompt_row is not None:
            prompt_row.set_class(active, "active")


def _format_exception(exc: Exception) -> str:
    detail = str(exc).strip()
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
