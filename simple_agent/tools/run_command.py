"""Tool for running safe inspection commands in the workspace."""

import json
from pathlib import Path

from simple_agent.tools.base_tool import BaseTool
from simple_agent.tools.command_runtime import CommandRuntime, CommandRuntimeError
from simple_agent.tools.definitions import RUN_COMMAND_DEFINITION
from simple_agent.tools.utils import error


class RunCommandTool(BaseTool):
    """Run a small allowlisted command without shell syntax."""

    # This tool is for trusted workspace inspection, not a full sandbox. It
    # runs outside the WorkspacePermission file filters, so keep the command
    # allowlist and argument checks conservative.

    name = "run_command"
    definition = RUN_COMMAND_DEFINITION

    def __init__(self, timeout_seconds: float = 30, max_output_chars: int = 12_000) -> None:
        self.runtime = CommandRuntime(
            timeout_seconds=timeout_seconds,
            max_output_chars=max_output_chars,
        )

    def run(self, root: Path, arguments: dict[str, object]) -> str:
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return error("command is required")

        try:
            result = self.runtime.run(root, command)
        except CommandRuntimeError as exc:
            return error(str(exc))

        return json.dumps(result.to_dict())
