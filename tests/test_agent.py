import json
import sys
import tempfile
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

from simple_agent.agent import SimpleAgent
from simple_agent.streaming import TextDelta, ToolResult


def stream_chunk(content=None, tool_calls=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content, tool_calls=tool_calls),
            )
        ]
    )


def empty_choices_chunk():
    return SimpleNamespace(choices=[])


def stream_response(text_chunks: list[str]):
    for content in text_chunks:
        yield stream_chunk(content=content)


def tool_call_delta(
    index: int,
    call_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
    call_type: str | None = None,
):
    return SimpleNamespace(
        index=index,
        id=call_id,
        type=call_type,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def stream_response_with_tool_calls(tool_call_chunks: list[list[SimpleNamespace]], text_chunks=None):
    for content in text_chunks or []:
        yield stream_chunk(content=content)
    for tool_calls in tool_call_chunks:
        yield stream_chunk(tool_calls=tool_calls)


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeLLMClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


class SimpleAgentTest(unittest.TestCase):
    def test_respond_stream_yields_text_deltas(self) -> None:
        llm_client = FakeLLMClient([[
            stream_chunk(content="Hello"),
            empty_choices_chunk(),
            stream_chunk(content=" there"),
        ]])
        agent = SimpleAgent(llm_client=llm_client, model="test-model")

        events = list(agent.respond_stream("Hi"))

        self.assertEqual(events, [TextDelta("Hello"), TextDelta(" there")])
        self.assertEqual(llm_client.chat.completions.calls[0]["model"], "test-model")
        self.assertTrue(llm_client.chat.completions.calls[0]["stream"])
        self.assertIn("tools", llm_client.chat.completions.calls[0])
        self.assertEqual([item["role"] for item in agent.conversation_history], ["system", "user", "assistant"])
        self.assertEqual(agent.conversation_history[-1]["content"], "Hello there")

    def test_respond_stream_runs_tool_call_then_streams_final_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            main_path = f"{temp_dir}/main.py"
            with open(main_path, "w", encoding="utf-8") as file:
                file.write("print('hello')\n")
            llm_client = FakeLLMClient(
                [
                    stream_response_with_tool_calls([
                        [
                            tool_call_delta(
                                index=0,
                                call_id="call_1",
                                name="read",
                                arguments="",
                                call_type="function",
                            )
                        ],
                        [tool_call_delta(index=0, name="_file", arguments='{"path"')],
                        [tool_call_delta(index=0, arguments=': "main.py"}')],
                    ]),
                    stream_response(["main.py", " prints hello"]),
                ]
            )
            agent = SimpleAgent(
                llm_client=llm_client,
                model="test-model",
                workspace=temp_dir,
            )

            events = list(agent.respond_stream("Read main.py"))

            self.assertIsInstance(events[0], ToolResult)
            self.assertEqual(events[0].call_id, "call_1")
            self.assertEqual(events[0].name, "read_file")
            self.assertTrue(events[0].ok)
            self.assertEqual(events[1:], [TextDelta("main.py"), TextDelta(" prints hello")])
            self.assertEqual(len(llm_client.chat.completions.calls), 2)
            self.assertTrue(all(call["stream"] for call in llm_client.chat.completions.calls))

            assistant_message = agent.conversation_history[2]
            self.assertIsNone(assistant_message["content"])
            self.assertEqual(assistant_message["tool_calls"][0]["id"], "call_1")
            self.assertEqual(assistant_message["tool_calls"][0]["function"]["name"], "read_file")
            self.assertEqual(
                json.loads(assistant_message["tool_calls"][0]["function"]["arguments"]),
                {"path": "main.py"},
            )

            tool_message = agent.conversation_history[3]
            self.assertEqual(tool_message["role"], "tool")
            self.assertEqual(tool_message["tool_call_id"], "call_1")
            self.assertIn("print('hello')", tool_message["content"])

    def test_unknown_tool_yields_error_tool_result(self) -> None:
        llm_client = FakeLLMClient(
            [
                stream_response_with_tool_calls([
                    [
                        tool_call_delta(
                            index=0,
                            call_id="call_1",
                            name="unknown_tool",
                            arguments="{}",
                            call_type="function",
                        )
                    ]
                ]),
                stream_response(["The tool was unavailable."]),
            ]
        )
        agent = SimpleAgent(llm_client=llm_client, model="test-model")

        events = list(agent.respond_stream("Use a mystery tool"))

        self.assertIsInstance(events[0], ToolResult)
        self.assertFalse(events[0].ok)
        self.assertEqual(events[1], TextDelta("The tool was unavailable."))
        tool_result = json.loads(agent.conversation_history[3]["content"])
        self.assertFalse(tool_result["ok"])
        self.assertIn("unknown tool", tool_result["error"])

    def test_invalid_tool_arguments_yields_error_tool_result(self) -> None:
        llm_client = FakeLLMClient(
            [
                stream_response_with_tool_calls([
                    [
                        tool_call_delta(
                            index=0,
                            call_id="call_1",
                            name="read_file",
                            arguments="{not-json",
                            call_type="function",
                        )
                    ]
                ]),
                stream_response(["The arguments were invalid."]),
            ]
        )
        agent = SimpleAgent(llm_client=llm_client, model="test-model")

        events = list(agent.respond_stream("Read badly"))

        self.assertIsInstance(events[0], ToolResult)
        self.assertFalse(events[0].ok)
        self.assertEqual(events[1], TextDelta("The arguments were invalid."))
        tool_result = json.loads(agent.conversation_history[3]["content"])
        self.assertFalse(tool_result["ok"])
        self.assertIn("invalid JSON arguments", tool_result["error"])

    def test_interleaved_tool_calls_are_ordered_by_index(self) -> None:
        llm_client = FakeLLMClient(
            [
                stream_response_with_tool_calls([
                    [
                        tool_call_delta(
                            index=1,
                            call_id="call_2",
                            name="list_files",
                            arguments='{"path"',
                            call_type="function",
                        ),
                        tool_call_delta(
                            index=0,
                            call_id="call_1",
                            name="list_files",
                            arguments='{"path"',
                            call_type="function",
                        ),
                    ],
                    [
                        tool_call_delta(index=1, arguments=': "."}'),
                        tool_call_delta(index=0, arguments=': "."}'),
                    ],
                ]),
                stream_response(["Done"]),
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = SimpleAgent(llm_client=llm_client, model="test-model", workspace=temp_dir)

            events = list(agent.respond_stream("List twice"))

        tool_results = [event for event in events if isinstance(event, ToolResult)]
        self.assertEqual([event.call_id for event in tool_results], ["call_1", "call_2"])
        assistant_tool_calls = agent.conversation_history[2]["tool_calls"]
        self.assertEqual([tool_call["id"] for tool_call in assistant_tool_calls], ["call_1", "call_2"])

    def test_respond_stream_max_turns_stops_loop(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            llm_client = FakeLLMClient(
                [
                    stream_response_with_tool_calls([
                        [
                            tool_call_delta(
                                index=0,
                                call_id="call_1",
                                name="list_files",
                                arguments="{}",
                                call_type="function",
                            )
                        ]
                    ]),
                ]
            )
            agent = SimpleAgent(
                llm_client=llm_client,
                model="test-model",
                workspace=temp_dir,
                max_turns=1,
            )

            events = list(agent.respond_stream("Keep listing files"))

        self.assertIsInstance(events[0], ToolResult)
        self.assertTrue(events[0].ok)
        self.assertEqual(events[-1], TextDelta("I stopped because the tool loop reached the max turn limit."))
        self.assertEqual(len(llm_client.chat.completions.calls), 1)
        self.assertEqual(agent.conversation_history[-1], {"role": "assistant", "content": events[-1].content})


if __name__ == "__main__":
    unittest.main()
