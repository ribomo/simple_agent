import unittest
from pathlib import Path

from plain_agent.sandbox import CommandRequest, SandboxMode
from plain_agent.tools.permissions.controller import (
    PermissionController,
    ApprovalDeniedError,
)
from plain_agent.tools.permissions.request import ApprovalDecision, CommandPermissionRequest


class RecordingApprovalHandler:
    def __init__(self, decision: ApprovalDecision) -> None:
        self.decision = decision
        self.requests: list[CommandPermissionRequest] = []

    def __call__(self, request: CommandPermissionRequest) -> ApprovalDecision:
        self.requests.append(request)
        return self.decision


class PermissionControllerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.request = CommandPermissionRequest(
            command=CommandRequest(
                argv=("pwd",),
                mode=SandboxMode.READ_ONLY,
                workspace=Path.cwd(),
            ),
            justification="Inspect the workspace path",
        )

    def test_denies_when_no_approval_handler_is_configured(self) -> None:
        with self.assertRaises(ApprovalDeniedError):
            PermissionController().require_approval(self.request)

    def test_uses_configured_approval_handler(self) -> None:
        approval_handler = RecordingApprovalHandler(ApprovalDecision.ALLOW_ONCE)
        controller = PermissionController(approval_handler)

        controller.require_approval(self.request)

        self.assertEqual(approval_handler.requests, [self.request])

    def test_raises_when_approval_handler_rejects_request(self) -> None:
        controller = PermissionController(RecordingApprovalHandler(ApprovalDecision.REJECT))

        with self.assertRaises(ApprovalDeniedError):
            controller.require_approval(self.request)


if __name__ == "__main__":
    unittest.main()
