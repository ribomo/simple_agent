"""Typed chat messages and conversation history storage."""

from collections.abc import Iterator
from copy import deepcopy
from typing import overload

from simple_agent.message_types import AssistantMessageDict, ChatMessage, ToolMessage, UserMessage


class ConversationHistory:
    """Stores chat messages while exposing list-like read access."""
    _messages: list[ChatMessage]

    def __init__(self, system_prompt: str) -> None:
        self._messages = [
            {"role": "system", "content": system_prompt},
        ]

    def append(self, message: ChatMessage) -> None:
        self._messages.append(message)

    def append_user(self, content: str) -> None:
        message: UserMessage = {"role": "user", "content": content}
        self.append(message)

    def append_assistant(self, content: str) -> None:
        message: AssistantMessageDict = {"role": "assistant", "content": content}
        self.append(message)

    def append_tool(self, tool_call_id: str, content: str) -> None:
        message: ToolMessage = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }
        self.append(message)

    def replace(self, messages: list[ChatMessage]) -> None:
        self._messages = messages

    def to_messages(self) -> list[ChatMessage]:
        return deepcopy(self._messages)

    def __iter__(self) -> Iterator[ChatMessage]:
        return iter(self._messages)

    def __len__(self) -> int:
        return len(self._messages)

    @overload
    def __getitem__(self, index: int) -> ChatMessage:
        ...

    @overload
    def __getitem__(self, index: slice) -> list[ChatMessage]:
        ...

    def __getitem__(self, index: int | slice) -> ChatMessage | list[ChatMessage]:
        return self._messages[index]
