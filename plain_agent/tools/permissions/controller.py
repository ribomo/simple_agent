"""User approval flow for protected tool actions."""

from collections.abc import Callable

from plain_agent.tools.permissions.network_permission import NetworkPermissionRequest
from plain_agent.tools.permissions.request import (
    ApprovalDecision,
    CommandPermissionRequest,
)

PermissionRequest = CommandPermissionRequest | NetworkPermissionRequest
ApprovalHandler = Callable[[PermissionRequest], ApprovalDecision]


class ApprovalDeniedError(PermissionError):
    """Raised when a tool action is not approved."""


class PermissionController:
    """Require the configured handler to approve protected tool actions."""

    def __init__(self, approval_handler: ApprovalHandler | None = None) -> None:
        self.set_approval_handler(approval_handler)

    def set_approval_handler(self, approval_handler: ApprovalHandler | None) -> None:
        """Bind or unbind the callback that approves protected tool actions."""
        self.approval_handler = approval_handler

    def require_approval(self, request: PermissionRequest) -> None:
        """Require approval before the caller continues its side effect."""
        if (
            self.approval_handler is None
            or self.approval_handler(request) is not ApprovalDecision.ALLOW_ONCE
        ):
            raise ApprovalDeniedError("tool action was not approved")
