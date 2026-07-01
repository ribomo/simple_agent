"""Tool for running OS-sandboxed commands in the workspace."""

import json
from pathlib import Path

from plain_agent.sandbox import CommandRequest, SandboxBackend, SandboxConfigurationError
from plain_agent.tools.base_tool import BaseTool
from plain_agent.tools.command_runtime import CommandRuntime, CommandRuntimeError
from plain_agent.tools.permissions.controller import (
    PermissionController,
    ApprovalDeniedError,
)
from plain_agent.tools.permissions.request import CommandPermissionRequest
from plain_agent.tools.utils import error


class RunCommandTool(BaseTool):
    """Run an argv vector through the configured OS sandbox."""

    name = "run_command"
    description = (
        "Run an offline command in an OS sandbox. Commands receive read-only workspace access "
        "unless mode is workspace-write. Pass an exact argv array; shell syntax works only when "
        "you explicitly invoke a shell, for example ['bash', '-lc', '...']."
    )
    parameters = {
        "type": "object",
        "properties": {
            "argv": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
                "description": "Exact executable and arguments. No implicit shell parsing is performed.",
            },
            "mode": {
                "type": "string",
                "enum": ["read-only", "workspace-write"],
                "default": "read-only",
                "description": "Filesystem access granted to the workspace.",
            },
            "justification": {
                "type": "string",
                "minLength": 1,
                "description": "Why this command is needed.",
            },
        },
        "required": ["argv", "justification"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        sandbox: SandboxBackend,
        permission_controller: PermissionController | None = None,
        timeout_seconds: float = 30,
        max_output_chars: int = 12_000,
    ) -> None:
        self.permission_controller = (
            permission_controller
            if permission_controller is not None
            else PermissionController()
        )
        self.runtime = CommandRuntime(
            sandbox=sandbox,
            timeout_seconds=timeout_seconds,
            max_output_chars=max_output_chars,
        )

    def run(self, root: Path, arguments: dict[str, object]) -> str:
        try:
            command = CommandRequest.from_arguments(root, arguments)
            justification = arguments.get("justification")
            if not isinstance(justification, str) or not justification.strip():
                raise SandboxConfigurationError("justification must be a non-empty string")
            permission_request = CommandPermissionRequest(
                command=command,
                justification=justification.strip(),
            )
            self.permission_controller.require_approval(permission_request)
            result = self.runtime.run(command)
        except ApprovalDeniedError:
            return error("run_command was not approved")
        except (CommandRuntimeError, SandboxConfigurationError) as exc:
            return error(str(exc))

        return json.dumps(result.to_dict())
