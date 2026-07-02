"""Simple tool loop for the educational agent."""

from collections.abc import Iterable
import json
from typing import Any, Generator

from plain_agent.compaction import ConversationCompactor
from plain_agent.conversation_history import ContextSize, ConversationHistory, estimate_token_count
from plain_agent.message_types import ToolCallDict
from plain_agent.prompt import INITIAL_PROMPT
from plain_agent.streaming import (
    AutoCompaction,
    ChatCompletionStreamAccumulator,
    TextDelta,
    ToolResult,
)
from plain_agent.tools.permissions.controller import PermissionController
from plain_agent.tools.registry import ToolRegistry
from plain_agent.tools.utils import error


class SimpleAgent:
    """A tiny Chat Completions agent loop with workspace tools."""

    def __init__(
        self,
        llm_client: Any,
        model: str,
        workspace: str = ".",
        max_turns: int = 10,
        compactor: ConversationCompactor | None = None,
        auto_compact_max_tokens: int | None = None,
        permission_controller: PermissionController | None = None,
        enable_network: bool = True,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        if auto_compact_max_tokens is not None and auto_compact_max_tokens < 1:
            raise ValueError("auto_compact_max_tokens must be positive when provided")
        self.llm_client = llm_client
        self.model = model
        self.max_turns = max_turns
        self.compactor = compactor
        self.auto_compact_max_tokens = auto_compact_max_tokens
        self.permission_controller = (
            permission_controller
            if permission_controller is not None
            else PermissionController()
        )
        self.tool_registry = ToolRegistry(
            workspace,
            permission_controller=self.permission_controller,
            enable_network=enable_network,
        )
        self.startup_warnings = self.tool_registry.startup_warnings
        self.conversation_history = ConversationHistory(INITIAL_PROMPT)

    def respond_stream(self, user_input: str) -> Generator[TextDelta | ToolResult | AutoCompaction, None, None]:
        self.conversation_history.append_user(user_input)

        for _ in range(self.max_turns):
            auto_compaction = self._compact_history_if_over_token_limit()
            if auto_compaction is not None:
                yield auto_compaction

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
                self.conversation_history.append_tool(tool_call_dict["id"], result)
                yield ToolResult(
                    call_id=tool_call_dict["id"],
                    name=tool_call_dict["function"]["name"],
                    result=result,
                    ok=self._tool_result_ok(result),
                )

        final_text = "I stopped because the tool loop reached the max turn limit."
        self.conversation_history.append_assistant(final_text)
        yield TextDelta(final_text)

    def context_size(self) -> ContextSize:
        return self.conversation_history.context_size()

    def compact_history(self) -> bool:
        if self.compactor is None:
            return False
        return self.compactor.compact(self.conversation_history)

    def _compact_history_if_over_token_limit(self) -> AutoCompaction | None:
        if self.compactor is None or self.auto_compact_max_tokens is None:
            return None

        current_size = self.context_size()
        if estimate_token_count(current_size.char_count) < self.auto_compact_max_tokens:
            return None

        # Run compaction
        compacted = self.compactor.compact(self.conversation_history)
        if not compacted:
            return None

        after = self.context_size()
        return AutoCompaction(before=current_size, after=after)

    def _create_llm_stream(self) -> Iterable[Any]:
        return self.llm_client.chat.completions.create(
            model=self.model,
            messages=self.conversation_history.to_messages(),
            tools=self.tool_registry.definitions(),
            tool_choice="auto",
            stream=True,
        )

    def _handle_tool_call(self, tool_call: ToolCallDict) -> str:
        name = tool_call["function"]["name"]
        try:
            arguments = self._parse_tool_arguments(tool_call["function"]["arguments"])
        except ValueError as exc:
            return error(str(exc))
        return self.tool_registry.run(name, arguments)

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
