"""Shared command sandbox types."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol
import shlex


class SandboxMode(str, Enum):
    """Filesystem access granted to a sandboxed command."""

    READ_ONLY = "read-only"
    WORKSPACE_WRITE = "workspace-write"


class SandboxConfigurationError(ValueError):
    """Raised when sandbox configuration is invalid."""


class SandboxUnavailableError(RuntimeError):
    """Raised when the requested platform sandbox cannot be used."""


@dataclass(frozen=True)
class CommandRequest:
    """Validated command and sandbox policy requested by the model."""

    argv: tuple[str, ...]
    mode: SandboxMode
    workspace: Path

    @property
    def display(self) -> str:
        return shlex.join(_escape_display_argument(value) for value in self.argv)

    @classmethod
    def from_arguments(cls, workspace: Path, arguments: dict[str, object]) -> "CommandRequest":
        raw_argv = arguments.get("argv")
        if not isinstance(raw_argv, list) or not raw_argv:
            raise SandboxConfigurationError("argv must be a non-empty array of strings")

        argv: list[str] = []
        for value in raw_argv:
            if not isinstance(value, str) or not value:
                raise SandboxConfigurationError("argv must contain only non-empty strings")
            if "\x00" in value:
                raise SandboxConfigurationError("argv entries must not contain NUL bytes")
            argv.append(value)

        raw_mode = arguments.get("mode", SandboxMode.READ_ONLY.value)
        if not isinstance(raw_mode, str):
            raise SandboxConfigurationError("mode must be 'read-only' or 'workspace-write'")
        try:
            mode = SandboxMode(raw_mode)
        except ValueError as exc:
            raise SandboxConfigurationError(
                "mode must be 'read-only' or 'workspace-write'"
            ) from exc

        return cls(argv=tuple(argv), mode=mode, workspace=workspace.resolve())


class SandboxBackend(Protocol):
    """Build an OS-enforced command invocation."""

    def verify_usable(self, timeout_seconds: float = 2) -> None:
        """Raise when the backend cannot enforce its policy on this host."""

    def build_command(self, request: CommandRequest) -> list[str]:
        """Return the host argv used to launch a sandboxed command."""


def _escape_display_argument(value: str) -> str:
    """Make an argv entry safe and unambiguous for terminal display."""
    escaped: list[str] = []
    for character in value:
        if character == "\\":
            escaped.append("\\\\")
        elif character.isprintable():
            escaped.append(character)
        else:
            # ascii() returns a quoted escape such as '\\x1b'; strip only the quotes.
            escaped.append(ascii(character)[1:-1])
    return "".join(escaped)
