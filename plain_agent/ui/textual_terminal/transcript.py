"""Incremental transcript rendering for the Textual terminal UI."""

from rich.text import Text
from textual.await_complete import AwaitComplete
from textual.containers import VerticalScroll
from textual.widgets import Markdown, Static


class TranscriptEntry(Static):
    """A selectable, pre-rendered transcript entry."""

    def __init__(self, content: Text) -> None:
        super().__init__(content, classes="transcript-entry", markup=False)
        self.plain_text = content.plain


class AssistantResponse(Markdown):
    """A live Markdown response that supports Textual selection."""

    def __init__(self) -> None:
        super().__init__(None, classes="assistant-response")
        self._stream = None
        self._skip_initial_empty_update = True

    def update(self, markdown: str) -> AwaitComplete:
        """Skip the native widget's redundant initial empty parse."""
        if self._skip_initial_empty_update and not markdown:
            self._skip_initial_empty_update = False
            return AwaitComplete.nothing()
        return super().update(markdown)

    async def append_delta(self, content_delta: str) -> None:
        """Append streamed Markdown without touching completed entries."""
        if self._stream is None:
            self._stream = Markdown.get_stream(self)
        await self._stream.write(content_delta)

    async def finish_stream(self) -> None:
        """Flush all queued Markdown fragments and stop the stream."""
        if self._stream is not None:
            await self._stream.stop()
            self._stream = None


class TextualTranscript:
    """Manage transcript entries and in-flight assistant text."""

    def __init__(self, log: VerticalScroll) -> None:
        self.log = log
        self.log.anchor()
        self._assistant: AssistantResponse | None = None

    def append(self, entry: Text) -> None:
        self.log.mount(TranscriptEntry(entry))

    async def update_assistant(self, content_delta: str) -> None:
        if self._assistant is None:
            self._assistant = AssistantResponse()
            await self.log.mount(self._assistant)
        await self._assistant.append_delta(content_delta)

    async def finish_assistant(self) -> None:
        if self._assistant is None:
            return

        await self._assistant.finish_stream()
        if not self._assistant.source.strip():
            await self._assistant.remove()
        self._assistant = None
