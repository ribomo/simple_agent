"""Conversation history compaction using an OpenAI-compatible chat model."""

from dataclasses import dataclass
import json
from typing import Any

from plain_agent.conversation_history import (
    ConversationExchange,
    ConversationHistory,
)
from plain_agent.message_types import (
    ChatMessage,
    SYSTEM_ROLE,
    USER_ROLE,
)
from plain_agent.prompt import (
    COMPACTION_SUMMARY_PREFIX,
    COMPACTION_SYSTEM_PROMPT,
    COMPACTION_USER_PROMPT_TEMPLATE,
)


@dataclass(frozen=True)
class CompactionConfig:
    """Settings for manual conversation history compaction."""

    keep_recent_exchanges: int = 2
    model: str | None = None

    def __post_init__(self) -> None:
        if self.keep_recent_exchanges < 1:
            raise ValueError("keep_recent_exchanges must be 1 or greater")


@dataclass
class CompactionWindow:
    """Conversation history divided into old and recent exchange windows."""

    system_messages: list[ChatMessage]
    old_exchanges: list[ConversationExchange]
    recent_exchanges: list[ConversationExchange]


def build_compaction_window(
    history: ConversationHistory,
    keep_recent_exchanges: int,
) -> CompactionWindow:
    if keep_recent_exchanges < 1:
        raise ValueError("keep_recent_exchanges must be 1 or greater")

    exchanges = history.exchanges()
    system_messages = [message for message in history if message["role"] == SYSTEM_ROLE]

    # Summarize only the older exchanges; the most recent exchanges stay verbatim.
    split_index = max(0, len(exchanges) - keep_recent_exchanges)
    return CompactionWindow(
        system_messages=system_messages,
        old_exchanges=exchanges[:split_index],
        recent_exchanges=exchanges[split_index:],
    )


class ConversationCompactor:
    """Summarize old exchanges and preserve recent exchanges verbatim."""

    def __init__(
        self,
        llm_client: Any,
        model: str,
        config: CompactionConfig | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.model = model
        self.config = config or CompactionConfig()

    def compact(self, history: ConversationHistory) -> bool:
        window = build_compaction_window(history, self.config.keep_recent_exchanges)
        if not window.old_exchanges:
            return False

        summary = self._create_summary(window)
        history.replace(self._compacted_messages(window, summary))
        return True

    def _create_summary(self, window: CompactionWindow) -> str:
        response = self.llm_client.chat.completions.create(
            model=self.config.model or self.model,
            messages=self._summary_request_messages(window),
            stream=False,
        )
        return self._extract_response_text(response)

    def _summary_request_messages(self, window: CompactionWindow) -> list[ChatMessage]:
        # Carry prior summaries into the next prompt so repeated compactions retain context.
        previous_summary = self._previous_summary_text(window.system_messages)
        compacted_history = self._serialize_exchanges(window.old_exchanges)
        user_content = COMPACTION_USER_PROMPT_TEMPLATE.format(
            previous_summary=previous_summary or "(none)",
            compacted_history=compacted_history,
        )
        return [
            {
                "role": SYSTEM_ROLE,
                "content": COMPACTION_SYSTEM_PROMPT,
            },
            {"role": USER_ROLE, "content": user_content},
        ]

    def _compacted_messages(self, window: CompactionWindow, summary: str) -> list[ChatMessage]:
        messages = self._base_system_messages(window.system_messages)
        messages.append(
            {
                "role": SYSTEM_ROLE,
                "content": f"{COMPACTION_SUMMARY_PREFIX}\n{summary}",
            }
        )
        for exchange in window.recent_exchanges:
            messages.extend(exchange.messages)
        return messages

    def _base_system_messages(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        # Replace old compaction summaries with the freshly generated summary.
        return [message for message in messages if not self._is_compaction_summary(message)]

    def _previous_summary_text(self, messages: list[ChatMessage]) -> str:
        summaries = []
        for message in messages:
            if self._is_compaction_summary(message):
                summaries.append(self._compaction_summary_body(message))
        return "\n\n".join(summaries)

    def _compaction_summary_body(self, message: ChatMessage) -> str:
        return str(message["content"])[len(COMPACTION_SUMMARY_PREFIX) :].strip()

    def _is_compaction_summary(self, message: ChatMessage) -> bool:
        content = message.get("content")
        return (
            message["role"] == SYSTEM_ROLE
            and isinstance(content, str)
            and content.startswith(COMPACTION_SUMMARY_PREFIX)
        )

    def _serialize_exchanges(self, exchanges: list[ConversationExchange]) -> str:
        return json.dumps(
            [exchange.messages for exchange in exchanges],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def _extract_response_text(self, response: Any) -> str:
        # Compaction reads from a non-streaming Chat Completions response.
        choices = response.choices
        if not choices:
            raise ValueError("compaction response did not include choices")

        message = choices[0].message
        content = message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("compaction response did not include summary content")
        return content.strip()
