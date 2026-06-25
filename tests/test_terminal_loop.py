from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import patch

from plain_agent.conversation_history import ContextSize
from plain_agent.streaming import TextDelta, ToolResult
from plain_agent.terminal_loop import approve_run_command, run_interactive_terminal


class FakeAgent:
    def __init__(self, compact_result: bool = False) -> None:
        self.prompts = []
        self.compact_result = compact_result
        self.compact_calls = 0

    def respond_stream(self, user_input: str):
        self.prompts.append(user_input)
        yield TextDelta("Hello")
        yield TextDelta(" there")
        yield ToolResult(
            call_id="call_1",
            name="list_files",
            result='{"ok": true}',
            ok=True,
        )
        yield TextDelta("Done")

    def context_size(self) -> ContextSize:
        return ContextSize(message_count=4, char_count=84, byte_count=84)

    def compact_history(self) -> bool:
        self.compact_calls += 1
        return self.compact_result


class TerminalLoopTest(unittest.TestCase):
    def test_run_interactive_terminal_streams_text_and_tool_results(self) -> None:
        agent = FakeAgent()
        output = StringIO()

        with patch("builtins.input", side_effect=["", "Hi", "exit"]) as mock_input:
            with redirect_stdout(output):
                run_interactive_terminal(agent)

        self.assertEqual([call.args[0] for call in mock_input.call_args_list], ["> ", "> ", "> "])
        self.assertEqual(agent.prompts, ["Hi"])
        self.assertIn("Simple agent client. Type 'exit' to quit.\n", output.getvalue())
        self.assertIn(
            "Hello there\n[tool list_files: ok]\nDone\n",
            output.getvalue(),
        )
        self.assertIn("[conversation history: 4 messages, 84 chars, 84 bytes]\n\n", output.getvalue())

    def test_run_interactive_terminal_compacts_on_command(self) -> None:
        agent = FakeAgent(compact_result=True)
        output = StringIO()

        with patch("builtins.input", side_effect=["/compact", "exit"]):
            with redirect_stdout(output):
                run_interactive_terminal(agent)

        self.assertEqual(agent.prompts, [])
        self.assertEqual(agent.compact_calls, 1)
        self.assertIn("[conversation compacted]\n", output.getvalue())
        self.assertIn("[conversation history: 4 messages, 84 chars, 84 bytes]\n\n", output.getvalue())

    def test_run_interactive_terminal_reports_when_compact_has_nothing_to_do(self) -> None:
        agent = FakeAgent(compact_result=False)
        output = StringIO()

        with patch("builtins.input", side_effect=["/compact", "exit"]):
            with redirect_stdout(output):
                run_interactive_terminal(agent)

        self.assertEqual(agent.prompts, [])
        self.assertEqual(agent.compact_calls, 1)
        self.assertIn("[conversation compact: nothing to compact]\n\n", output.getvalue())

    def test_run_interactive_terminal_handles_eof(self) -> None:
        output = StringIO()

        with patch("builtins.input", side_effect=EOFError):
            with redirect_stdout(output):
                run_interactive_terminal(FakeAgent())

        self.assertTrue(output.getvalue().endswith("\n\n"))

    def test_approve_run_command_accepts_yes(self) -> None:
        output = StringIO()

        with patch("builtins.input", side_effect=["maybe", "yes"]):
            with redirect_stdout(output):
                approved = approve_run_command("pwd")

        self.assertTrue(approved)
        self.assertEqual(output.getvalue(), "Please answer y or n.\n")

    def test_approve_run_command_rejects_default(self) -> None:
        with patch("builtins.input", return_value=""):
            approved = approve_run_command("pwd")

        self.assertFalse(approved)


if __name__ == "__main__":
    unittest.main()
