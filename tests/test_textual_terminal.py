import threading
import unittest

from rich.text import Text
from textual.selection import SELECT_ALL

from plain_agent.conversation_history import ContextSize
from plain_agent.streaming import TextDelta, ToolResult
from plain_agent.ui.textual_terminal.approval import parse_approval_answer
from plain_agent.ui.textual_terminal.app import PlainAgentTextualApp
from plain_agent.ui.textual_terminal.rendering import (
    format_context_size,
    format_tool_result,
)
from plain_agent.ui.textual_terminal.transcript import AssistantResponse, TranscriptEntry


class TextualTerminalTest(unittest.TestCase):
    def test_format_tool_result_uses_status_text(self) -> None:
        rendered = format_tool_result(
            ToolResult(
                call_id="call_1",
                name="list_files",
                result='{"ok": true}',
                ok=True,
            )
        ).plain

        self.assertEqual(rendered, "[tool list_files: ok]")

    def test_format_context_size_matches_basic_terminal(self) -> None:
        rendered = format_context_size(ContextSize(message_count=4, char_count=8_400, byte_count=8_400)).plain

        self.assertEqual(rendered, "[conversation history: 4 messages, ~2.1k tokens]")

    def test_parse_approval_answer_accepts_yes(self) -> None:
        self.assertIs(parse_approval_answer("y"), True)
        self.assertIs(parse_approval_answer("yes"), True)
        self.assertIs(parse_approval_answer(" YES "), True)

    def test_parse_approval_answer_rejects_no_and_empty(self) -> None:
        self.assertIs(parse_approval_answer(""), False)
        self.assertIs(parse_approval_answer("n"), False)
        self.assertIs(parse_approval_answer("no"), False)

    def test_parse_approval_answer_returns_none_for_invalid_input(self) -> None:
        self.assertIsNone(parse_approval_answer("maybe"))


class TextualTerminalAppTest(unittest.IsolatedAsyncioTestCase):
    async def test_prompt_can_release_focus(self) -> None:
        app = PlainAgentTextualApp(_FakeAgent())

        async with app.run_test() as pilot:
            self.assertIs(app.focused, app.prompt_input)
            self.assertTrue(app.prompt_row.has_class("active"))

            await pilot.press("shift+tab")

            self.assertIs(app.focused, app.transcript.log)
            self.assertFalse(app.prompt_row.has_class("active"))

    async def test_busy_submission_preserves_prompt_text(self) -> None:
        app = PlainAgentTextualApp(_FakeAgent())

        async with app.run_test():
            app._responding = True
            app.prompt_input.value = "keep this draft"

            app._submit_prompt(app.prompt_input.value.strip())

            self.assertEqual(app.prompt_input.value, "keep this draft")
            self.assertEqual(app.status.render().plain, "Assistant is still responding.")

    async def test_wrapped_entry_selection_uses_logical_offsets(self) -> None:
        app = PlainAgentTextualApp(_FakeAgent())

        async with app.run_test(size=(30, 12)) as pilot:
            app.transcript.append(Text("abcdefghijklmnopqrstuvwxyz" * 4))
            await pilot.pause()
            entry = list(app.query(TranscriptEntry))[-1]

            await _drag(pilot, entry, (0, 1), entry, (4, 1))

            self.assertGreater(entry.content_region.height, 1)
            self.assertEqual(app.screen.get_selected_text(), "abcde")
            self.assertIs(app.focused, app.prompt_input)

    async def test_selection_handles_wide_unicode_characters(self) -> None:
        app = PlainAgentTextualApp(_FakeAgent())

        async with app.run_test(size=(30, 12)) as pilot:
            app.transcript.append(Text("🙂abcdef"))
            await pilot.pause()
            entry = list(app.query(TranscriptEntry))[-1]

            await _drag(pilot, entry, (2, 0), entry, (3, 0))

            self.assertEqual(app.screen.get_selected_text(), "ab")

    async def test_markdown_selection_is_partial_and_visible(self) -> None:
        app = PlainAgentTextualApp(_FakeAgent())

        async with app.run_test(size=(50, 12)) as pilot:
            await app.transcript.update_assistant("**assistant bold text**")
            await app.transcript.finish_assistant()
            await pilot.pause()
            assistant = app.query_one(AssistantResponse)
            block = list(assistant.query("MarkdownBlock"))[0]

            await _drag(pilot, block, (0, 0), block, (8, 0))

            selected_style = app.screen.get_style_at(block.region.x, block.region.y)
            unselected_style = app.screen.get_style_at(block.region.x + 10, block.region.y)
            self.assertEqual(app.screen.get_selected_text(), "assistant")
            self.assertNotEqual(selected_style.bgcolor, unselected_style.bgcolor)
            self.assertIs(app.focused, app.prompt_input)

    async def test_selection_spans_plain_and_markdown_entries(self) -> None:
        app = PlainAgentTextualApp(_FakeAgent())

        async with app.run_test(size=(50, 14)) as pilot:
            app.transcript.append(Text("before entry"))
            await app.transcript.update_assistant("**assistant bold text**")
            await app.transcript.finish_assistant()
            await pilot.pause()
            entry = list(app.query(TranscriptEntry))[-1]
            assistant = app.query_one(AssistantResponse)
            block = list(assistant.query("MarkdownBlock"))[0]

            await _drag(pilot, entry, (3, 0), block, (8, 0))

            self.assertEqual(app.screen.get_selected_text(), "ore entry\nassistan")

    async def test_transcript_copy_wins_while_prompt_keeps_focus(self) -> None:
        app = PlainAgentTextualApp(_FakeAgent())

        async with app.run_test(size=(50, 12)) as pilot:
            app.prompt_input.value = "prompt selection"
            app.prompt_input.select_all()
            app.transcript.append(Text("transcript selection"))
            await pilot.pause()
            entry = list(app.query(TranscriptEntry))[-1]

            await _drag(pilot, entry, (0, 0), entry, (9, 0))
            await pilot.press("ctrl+c")

            self.assertIs(app.focused, app.prompt_input)
            self.assertEqual(app._clipboard, "transcript")

            app.screen.clear_selection()
            await pilot.press("ctrl+c")

            self.assertEqual(app._clipboard, "prompt selection")

    async def test_streaming_does_not_override_manual_scroll(self) -> None:
        app = PlainAgentTextualApp(_StreamingAgent())

        async with app.run_test(size=(40, 10)) as pilot:
            for index in range(30):
                app.transcript.append(Text(f"line {index}"))
            await app.transcript.update_assistant("first")
            await pilot.pause()
            app.transcript.log.scroll_home(animate=False, immediate=True)
            await pilot.pause()

            await app.transcript.update_assistant(" second")
            await app._finish_response(None)
            await pilot.pause()

            self.assertEqual(app.transcript.log.scroll_y, 0)

    async def test_compaction_runs_without_blocking_the_ui(self) -> None:
        agent = _BlockingCompactionAgent()
        app = PlainAgentTextualApp(agent)

        async with app.run_test() as pilot:
            app._submit_prompt("/compact")
            await pilot.pause()

            self.assertTrue(agent.started.wait(timeout=1))
            self.assertTrue(app._compacting)
            self.assertEqual(app.status.render().plain, "Compacting conversation...")

            app.prompt_input.value = "keep this draft"
            app._submit_prompt(app.prompt_input.value.strip())
            self.assertEqual(app.prompt_input.value, "keep this draft")
            self.assertEqual(app.status.render().plain, "Conversation is still being compacted.")

            agent.release.set()
            for _ in range(20):
                await pilot.pause(0.01)
                if not app._compacting:
                    break

            self.assertFalse(app._compacting)
            entries = [entry.plain_text for entry in app.query(TranscriptEntry)]
            self.assertIn("[conversation: compacted]", entries)
            self.assertEqual(app.status.render().plain, "")

    async def test_response_failure_is_rendered_in_transcript(self) -> None:
        app = PlainAgentTextualApp(_FailingAgent())

        async with app.run_test() as pilot:
            app._submit_prompt("Hello")
            for _ in range(20):
                await pilot.pause(0.01)
                if not app._responding:
                    break

            entries = [entry.plain_text for entry in app.query(TranscriptEntry)]
            self.assertIn("[assistant error: RuntimeError: network unavailable]", entries)
            self.assertEqual(app.status.render().plain, "Assistant response failed.")

    async def test_background_stream_updates_incremental_markdown(self) -> None:
        app = PlainAgentTextualApp(_StreamingAgent())

        async with app.run_test() as pilot:
            welcome = app.query_one(TranscriptEntry)
            app._submit_prompt("Hello")
            for _ in range(20):
                await pilot.pause(0.01)
                if not app._responding:
                    break

            assistant = app.query_one(AssistantResponse)
            block = list(assistant.query("MarkdownBlock"))[0]
            self.assertEqual(assistant.source, "Streamed response")
            self.assertEqual(block.get_selection(SELECT_ALL)[0], "Streamed response")
            self.assertIs(app.query_one(TranscriptEntry), welcome)
            self.assertEqual(welcome.get_selection(SELECT_ALL)[0], "Plain Agent Type 'exit' to quit.")
            self.assertEqual(len(app.query(AssistantResponse)), 1)
            self.assertEqual(app.status.render().plain, "")


class _FakeAgent:
    command_approver = None


class _FailingAgent(_FakeAgent):
    def respond_stream(self, user_input: str):
        raise RuntimeError("network unavailable")
        yield


class _StreamingAgent(_FakeAgent):
    def respond_stream(self, user_input: str):
        yield TextDelta("Streamed")
        yield TextDelta(" response")

    def context_size(self) -> ContextSize:
        return ContextSize(message_count=2, char_count=100, byte_count=100)


class _BlockingCompactionAgent(_StreamingAgent):
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def compact_history(self) -> bool:
        self.started.set()
        self.release.wait(timeout=1)
        return True


async def _drag(pilot, start, start_offset, end, end_offset) -> None:
    await pilot.mouse_down(start, offset=start_offset)
    await pilot.hover(end, offset=end_offset)
    await pilot.mouse_up(end, offset=end_offset)
    await pilot.pause()


if __name__ == "__main__":
    unittest.main()
