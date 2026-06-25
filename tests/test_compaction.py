import sys
import types
import unittest
from types import SimpleNamespace

fake_openai_module = types.ModuleType("openai")
fake_openai_types_module = types.ModuleType("openai.types")
fake_openai_chat_module = types.ModuleType("openai.types.chat")
fake_openai_chat_chunk_module = types.ModuleType("openai.types.chat.chat_completion_chunk")
fake_openai_chat_module.ChatCompletionMessage = object
fake_openai_chat_module.ChatCompletionMessageToolCall = object
fake_openai_chat_chunk_module.ChatCompletionChunk = object
fake_openai_chat_chunk_module.ChoiceDeltaToolCall = object
sys.modules.setdefault("openai", fake_openai_module)
sys.modules.setdefault("openai.types", fake_openai_types_module)
sys.modules.setdefault("openai.types.chat", fake_openai_chat_module)
sys.modules.setdefault("openai.types.chat.chat_completion_chunk", fake_openai_chat_chunk_module)

from plain_agent.compaction import (  # noqa: E402
    CompactionConfig,
    ConversationCompactor,
    build_compaction_window,
)
from plain_agent.conversation_history import ConversationHistory  # noqa: E402
from plain_agent.prompt import COMPACTION_SUMMARY_PREFIX  # noqa: E402


class FakeCompletions:
    def __init__(self, summary: str) -> None:
        self.summary = summary
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.summary),
                )
            ]
        )


class FakeLLMClient:
    def __init__(self, summary: str) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(summary))


class ConversationCompactorTest(unittest.TestCase):
    def test_build_compaction_window_splits_old_and_recent_exchanges(self) -> None:
        history = ConversationHistory("System")
        history.append_user("First")
        history.append_assistant("First answer")
        history.append_user("Second")
        history.append_assistant("Second answer")
        history.append_user("Third")
        history.append_assistant("Third answer")

        window = build_compaction_window(history, keep_recent_exchanges=1)

        self.assertEqual([message["role"] for message in window.system_messages], ["system"])
        self.assertEqual(len(window.old_exchanges), 2)
        self.assertEqual(len(window.recent_exchanges), 1)
        self.assertEqual(window.old_exchanges[0].messages[0]["content"], "First")
        self.assertEqual(window.recent_exchanges[0].messages[0]["content"], "Third")

    def test_build_compaction_window_rejects_zero_recent_exchanges(self) -> None:
        history = ConversationHistory("System")
        history.append_user("First")
        history.append_assistant("First answer")

        with self.assertRaisesRegex(ValueError, "keep_recent_exchanges must be 1 or greater"):
            build_compaction_window(history, keep_recent_exchanges=0)

    def test_compaction_config_rejects_zero_recent_exchanges(self) -> None:
        with self.assertRaisesRegex(ValueError, "keep_recent_exchanges must be 1 or greater"):
            CompactionConfig(keep_recent_exchanges=0)

    def test_compaction_config_rejects_negative_recent_exchanges(self) -> None:
        with self.assertRaisesRegex(ValueError, "keep_recent_exchanges must be 1 or greater"):
            CompactionConfig(keep_recent_exchanges=-1)

    def test_build_compaction_window_handles_short_history(self) -> None:
        history = ConversationHistory("System")
        history.append_user("First")
        history.append_assistant("First answer")

        window = build_compaction_window(history, keep_recent_exchanges=3)

        self.assertEqual(window.old_exchanges, [])
        self.assertEqual(len(window.recent_exchanges), 1)

    def test_compact_skips_when_no_old_exchanges_exist(self) -> None:
        llm_client = FakeLLMClient("summary")
        history = ConversationHistory("System")
        history.append_user("Hi")
        history.append_assistant("Hello")
        compactor = ConversationCompactor(
            llm_client=llm_client,
            model="test-model",
            config=CompactionConfig(keep_recent_exchanges=1),
        )

        compacted = compactor.compact(history)

        self.assertFalse(compacted)
        self.assertEqual(llm_client.chat.completions.calls, [])

    def test_compact_replaces_old_exchanges_with_summary(self) -> None:
        llm_client = FakeLLMClient("Old exchange summary")
        history = ConversationHistory("System")
        history.append_user("Old question")
        history.append_assistant("Old answer")
        history.append_user("Recent question")
        history.append_assistant("Recent answer")
        compactor = ConversationCompactor(
            llm_client=llm_client,
            model="test-model",
            config=CompactionConfig(keep_recent_exchanges=1),
        )

        compacted = compactor.compact(history)

        self.assertTrue(compacted)
        call = llm_client.chat.completions.calls[0]
        self.assertEqual(call["model"], "test-model")
        self.assertFalse(call["stream"])
        self.assertNotIn("tools", call)
        self.assertIn("Old question", call["messages"][1]["content"])
        self.assertNotIn("Recent question", call["messages"][1]["content"])

        messages = history.to_messages()
        self.assertEqual(
            [message["role"] for message in messages],
            ["system", "system", "user", "assistant"],
        )
        self.assertEqual(messages[0]["content"], "System")
        self.assertEqual(
            messages[1]["content"],
            f"{COMPACTION_SUMMARY_PREFIX}\nOld exchange summary",
        )
        self.assertEqual(messages[2]["content"], "Recent question")
        self.assertEqual(messages[3]["content"], "Recent answer")

    def test_compact_rolls_previous_summary_forward(self) -> None:
        llm_client = FakeLLMClient("Updated summary")
        history = ConversationHistory("System")
        history.append({"role": "system", "content": f"{COMPACTION_SUMMARY_PREFIX}\nEarlier summary"})
        history.append_user("Old question")
        history.append_assistant("Old answer")
        history.append_user("Recent question")
        history.append_assistant("Recent answer")
        compactor = ConversationCompactor(
            llm_client=llm_client,
            model="test-model",
            config=CompactionConfig(keep_recent_exchanges=1),
        )

        compacted = compactor.compact(history)

        self.assertTrue(compacted)
        call = llm_client.chat.completions.calls[0]
        self.assertIn("Earlier summary", call["messages"][1]["content"])

        messages = history.to_messages()
        summaries = [
            message
            for message in messages
            if message["role"] == "system"
            and message["content"].startswith(COMPACTION_SUMMARY_PREFIX)
        ]
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["content"], f"{COMPACTION_SUMMARY_PREFIX}\nUpdated summary")


if __name__ == "__main__":
    unittest.main()
