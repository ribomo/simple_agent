"""Runtime for safe workspace inspection commands."""

from dataclasses import asdict, dataclass
from pathlib import Path
import shlex
import subprocess

from simple_agent.tools.command_policy import (
    RUN_COMMAND_ALLOWED_COMMANDS,
    RUN_COMMAND_ALLOWED_GIT_SUBCOMMANDS,
)
from simple_agent.tools.permissions.file_permission import WorkspacePermission

ALLOWED_COMMANDS = set(RUN_COMMAND_ALLOWED_COMMANDS)
ALLOWED_GIT_SUBCOMMANDS = set(RUN_COMMAND_ALLOWED_GIT_SUBCOMMANDS)
SHELL_OPERATORS = ("|", ">", "<", "&&", "||", ";", "$(", "`", "&")
MUTATING_FLAGS = {
    "-delete",
    "-exec",
    "-execdir",
    "-ok",
    "-okdir",
}


class CommandRuntimeError(ValueError):
    """Raised when a command cannot be safely run."""


@dataclass
class CommandResult:
    ok: bool
    command: str
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    truncated: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class CommandRuntime:
    """Runs allowlisted commands from a workspace without shell syntax."""

    def __init__(self, timeout_seconds: float = 30, max_output_chars: int = 12_000) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars

    def run(self, workspace: Path, command: str) -> CommandResult:
        argv = self._parse_command(command)
        self._require_command_access(workspace, argv)

        try:
            completed = subprocess.run(
                argv,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                shell=False,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = self._output_text(exc.stdout)
            stderr = self._output_text(exc.stderr)
            stdout, stderr, truncated = self._truncate_outputs(stdout, stderr)
            return CommandResult(
                ok=False,
                command=command,
                exit_code=None,
                stdout=stdout,
                stderr=stderr,
                timed_out=True,
                truncated=truncated,
            )
        except OSError as exc:
            raise CommandRuntimeError(f"could not run command: {exc}") from exc

        stdout, stderr, truncated = self._truncate_outputs(completed.stdout, completed.stderr)
        return CommandResult(
            ok=completed.returncode == 0,
            command=command,
            exit_code=completed.returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=False,
            truncated=truncated,
        )

    def _parse_command(self, command: str) -> list[str]:
        self._require_no_shell_syntax(command)
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            raise CommandRuntimeError(f"invalid command: {exc}") from exc
        if not argv:
            raise CommandRuntimeError("command is required")
        return argv

    def _require_no_shell_syntax(self, command: str) -> None:
        for operator in SHELL_OPERATORS:
            if operator in command:
                raise CommandRuntimeError(f"shell syntax is not allowed: {operator}")

    def _require_command_access(self, workspace: Path, argv: list[str]) -> None:
        self._check_supported_command(argv)
        self._check_command_paths(workspace, argv)

    def _check_supported_command(self, argv: list[str]) -> None:
        executable = argv[0]
        if executable in ALLOWED_COMMANDS:
            return

        if executable == "git":
            if len(argv) < 2:
                raise CommandRuntimeError("git subcommand is required")
            subcommand = argv[1]
            if subcommand in ALLOWED_GIT_SUBCOMMANDS:
                return
            raise CommandRuntimeError(f"git subcommand is not allowed: {subcommand}")

        raise CommandRuntimeError(f"command is not allowed: {executable}")

    def _check_command_paths(self, workspace: Path, argv: list[str]) -> None:
        permissions = WorkspacePermission(workspace)
        for argument in argv[1:]:
            if argument in MUTATING_FLAGS:
                raise CommandRuntimeError(f"command flag is not allowed: {argument}")
            if argument.startswith("-"):
                continue

            argument_path = Path(argument)
            if argument_path.is_absolute() or ".." in argument_path.parts:
                raise CommandRuntimeError(f"path argument is outside workspace: {argument}")

            candidate = permissions.workspace / argument_path
            if candidate.exists() and not permissions.contains_path(candidate.resolve()):
                raise CommandRuntimeError(f"path argument is outside workspace: {argument}")

    def _truncate_outputs(self, stdout: str, stderr: str) -> tuple[str, str, bool]:
        if len(stdout) + len(stderr) <= self.max_output_chars:
            return stdout, stderr, False

        stdout = stdout[:self.max_output_chars]
        stderr = stderr[:self.max_output_chars - len(stdout)]
        return stdout, stderr, True

    def _output_text(self, output: str | bytes | None) -> str:
        if output is None:
            return ""
        if isinstance(output, bytes):
            return output.decode(errors="replace")
        return output
