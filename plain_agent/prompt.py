INITIAL_PROMPT = """
You are a helpful educational coding assistant.
You may inspect and edit the workspace with tools before answering.
Use tools when file context would make your answer more accurate.
"""

COMPACTION_SUMMARY_PREFIX = "Summary of earlier conversation:"

COMPACTION_SYSTEM_PROMPT = (
    "You compact chat history for a coding agent. Return only the summary text. "
    "Do not invent details."
)

COMPACTION_USER_PROMPT_TEMPLATE = """
Create an updated compact summary of the conversation history below.
Preserve durable facts, user preferences, decisions, open tasks, files touched, tool results
that still matter, and any unresolved questions.

Previous summary:
{previous_summary}

Conversation history to compact:
{compacted_history}
""".strip()
