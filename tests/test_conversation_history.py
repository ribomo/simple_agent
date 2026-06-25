import json
import unittest

from plain_agent.conversation_history import ConversationExchange, ConversationHistory


class ConversationHistoryTest(unittest.TestCase):
    def test_context_size_uses_exact_stable_json_serialization(self) -> None:
        history = ConversationHistory("System")
        history.append_user("Hello café")
        history.append_assistant("Hi")

        messages = history.to_messages()
        serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        size = history.context_size()

        self.assertEqual(size.message_count, len(messages))
        self.assertEqual(size.char_count, len(serialized))
        self.assertEqual(size.byte_count, len(serialized.encode("utf-8")))
        self.assertGreater(size.byte_count, size.char_count)

    def test_exchanges_group_user_led_message_sequences(self) -> None:
        history = ConversationHistory("System")
        history.append_user("First")
        history.append_assistant("First answer")
        history.append_user("Second")
        history.append_assistant("Second answer")

        exchanges = history.exchanges()

        self.assertEqual(len(exchanges), 2)
        self.assertEqual(
            [[message["role"] for message in exchange.messages] for exchange in exchanges],
            [["user", "assistant"], ["user", "assistant"]],
        )
        self.assertEqual(exchanges[0].messages[0]["content"], "First")
        self.assertEqual(exchanges[1].messages[0]["content"], "Second")

    def test_exchanges_keep_tool_call_messages_together(self) -> None:
        history = ConversationHistory("System")
        history.append_user("Read a file")
        history.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path":"main.py"}'},
                    }
                ],
            }
        )
        history.append_tool("call_1", '{"ok": true, "content": "print()"}')
        history.append_assistant("Done")

        exchanges = history.exchanges()

        self.assertEqual(len(exchanges), 1)
        self.assertEqual(
            [message["role"] for message in exchanges[0].messages],
            ["user", "assistant", "tool", "assistant"],
        )
        self.assertEqual(exchanges[0].messages[2]["tool_call_id"], "call_1")

    def test_conversation_exchange_from_messages_snapshots_messages(self) -> None:
        messages = [{"role": "user", "content": "Original"}]

        exchange = ConversationExchange.from_messages(messages)
        messages[0]["content"] = "Changed"

        self.assertEqual(exchange.messages[0]["content"], "Original")


if __name__ == "__main__":
    unittest.main()
