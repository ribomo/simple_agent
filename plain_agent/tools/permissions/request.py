"""Typed requests and decisions for user-approved actions."""

from dataclasses import dataclass
from enum import Enum

from plain_agent.sandbox import CommandRequest


class ApprovalDecision(str, Enum):
    """A user's decision for one approval request."""

    ALLOW_ONCE = "allow_once"
    REJECT = "reject"


@dataclass(frozen=True)
class CommandPermissionRequest:
    """A sandboxed command requiring user approval before execution."""

    command: CommandRequest
    justification: str
