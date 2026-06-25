"""Helpers for Chat Completions streaming responses."""

from dataclasses import dataclass

from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    ChoiceDeltaToolCall,
)

from plain_agent.conversation_history import ContextSize
from plain_agent.message_types import ASSISTANT_ROLE, AssistantMessageDict, ToolCallDict


@dataclass
class TextDelta:
    """A streamed assistant text chunk."""

    content: str


@dataclass
class ToolResult:
    """A completed tool call event."""

    call_id: str
    name: str
    result: str
    ok: bool


@dataclass
class AutoCompaction:
    """Conversation history was automatically compacted before an LLM call."""

    before: ContextSize
    after: ContextSize


def merge_tool_call_delta_into(tool_call: ToolCallDict, tool_call_delta: ChoiceDeltaToolCall) -> None:
    # Each streamed delta may contain only one fragment of the final tool call.
    call_id = tool_call_delta.id
    if call_id:
        tool_call["id"] = call_id

    call_type = tool_call_delta.type
    if call_type:
        tool_call["type"] = call_type

    function_delta = tool_call_delta.function
    if function_delta is None:
        return

    name = function_delta.name
    if name:
        tool_call["function"]["name"] += name

    arguments = function_delta.arguments
    if arguments:
        tool_call["function"]["arguments"] += arguments


class ChatCompletionStreamAccumulator:
    """Build the final assistant message from streamed Chat Completions chunks."""

    def __init__(self) -> None:
        self.full_text = ""
        self.tool_calls_by_index: dict[int, ToolCallDict] = {}

    def add_chunk(self, chunk: ChatCompletionChunk) -> list[TextDelta]:
        # OpenAI can send usage/bookkeeping chunks with choices=[].
        choices = chunk.choices
        if not choices:
            return []

        delta = choices[0].delta
        events = []

        content = delta.content
        if content:
            self.full_text += content
            events.append(TextDelta(content))

        for tool_call_delta in delta.tool_calls or []:
            index = tool_call_delta.index
            if index not in self.tool_calls_by_index:
                self.tool_calls_by_index[index] = {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }

            tool_call = self.tool_calls_by_index[index]
            # Later deltas usually add more argument text for this index.
            merge_tool_call_delta_into(tool_call, tool_call_delta)

        return events

    def assistant_message(self) -> AssistantMessageDict:
        message_dict: AssistantMessageDict = {"role": ASSISTANT_ROLE, "content": self.full_text or None}
        if self.tool_calls_by_index:
            message_dict["tool_calls"] = [
                self.tool_calls_by_index[index]
                for index in sorted(self.tool_calls_by_index)
            ]
        return message_dict
