"""Simple tool loop for the educational agent."""

from collections.abc import Iterable
import json
from typing import Any, Generator

from simple_agent.prompt import INITIAL_PROMPT
from simple_agent.streaming import (
    ChatCompletionStreamAccumulator,
    TextDelta,
    ToolCallDict,
    ToolResult,
)
from simple_agent.tools.tools import Tools
from simple_agent.tools.utils import error


class SimpleAgent:
    """A tiny Chat Completions agent loop with workspace tools."""

    def __init__(
        self,
        llm_client: Any,
        model: str,
        workspace: str = ".",
        max_turns: int = 5,
    ) -> None:
        self.llm_client = llm_client
        self.model = model
        self.max_turns = max_turns
        self.tools = Tools(workspace)
        self.conversation_history: list[dict[str, Any]] = [
            {"role": "system", "content": INITIAL_PROMPT},
        ]

    def respond_stream(self, user_input: str) -> Generator[TextDelta | ToolResult, None, None]:
        self.conversation_history.append({"role": "user", "content": user_input})

        for _ in range(self.max_turns):
            accumulator = ChatCompletionStreamAccumulator()
            for chunk in self._create_llm_stream():
                for event in accumulator.add_chunk(chunk):
                    yield event

            message_dict = accumulator.assistant_message()
            self.conversation_history.append(message_dict)

            tool_calls = message_dict.get("tool_calls")
            if not tool_calls:
                return

            for tool_call_dict in tool_calls:
                result = self._handle_tool_call(tool_call_dict)
                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call_dict["id"],
                    "content": result,
                })
                yield ToolResult(
                    call_id=tool_call_dict["id"],
                    name=tool_call_dict["function"]["name"],
                    result=result,
                    ok=self._tool_result_ok(result),
                )

        final_text = "I stopped because the tool loop reached the max turn limit."
        self.conversation_history.append({"role": "assistant", "content": final_text})
        yield TextDelta(final_text)

    def _create_llm_stream(self) -> Iterable[Any]:
        return self.llm_client.chat.completions.create(
            model=self.model,
            messages=self.conversation_history,
            tools=self.tools.definitions(),
            tool_choice="auto",
            stream=True,
        )

    def _handle_tool_call(self, tool_call: ToolCallDict) -> str:
        name = tool_call["function"]["name"]
        try:
            arguments = self._parse_tool_arguments(tool_call["function"]["arguments"])
        except ValueError as exc:
            return error(str(exc))
        return self.tools.run(name, arguments)

    def _tool_result_ok(self, result: str) -> bool:
        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            return False
        if not isinstance(parsed, dict):
            return False
        return parsed.get("ok") is True

    def _parse_tool_arguments(self, raw_arguments: str) -> dict[str, object]:
        try:
            arguments = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON arguments: {exc}")
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be a JSON object")
        return arguments
