"""Command approval state for the Textual terminal UI."""

import threading


class PendingApproval:
    """A command approval request waiting for bottom-prompt input."""

    def __init__(self) -> None:
        self.done = threading.Event()
        self.approved = False


def parse_approval_answer(answer: str) -> bool | None:
    """Parse command approval input."""
    normalized = answer.strip().lower()
    if normalized in {"y", "yes"}:
        return True
    if normalized in {"", "n", "no"}:
        return False
    return None
