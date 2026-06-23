"""TypedDict shapes for OpenAI-compatible chat messages."""

from typing import Literal, NotRequired, TypedDict


class SystemMessage(TypedDict):
    role: Literal["system"]
    content: str


class UserMessage(TypedDict):
    role: Literal["user"]
    content: str


class ToolMessage(TypedDict):
    role: Literal["tool"]
    tool_call_id: str
    content: str


class FunctionCallDict(TypedDict):
    name: str
    arguments: str


class ToolCallDict(TypedDict):
    id: str
    type: Literal["function"]
    function: FunctionCallDict


class AssistantMessageDict(TypedDict):
    role: Literal["assistant"]
    content: str | None
    tool_calls: NotRequired[list[ToolCallDict]]


ChatMessage = SystemMessage | UserMessage | AssistantMessageDict | ToolMessage
