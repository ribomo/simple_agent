"""Workspace mount protections for the Bubblewrap command sandbox."""

import os
from pathlib import Path
import stat

from plain_agent.sandbox.base import SandboxConfigurationError, SandboxMode
from plain_agent.tools.permissions.file_permission import (
    SENSITIVE_FILE_NAMES,
    SENSITIVE_FILE_SUFFIXES,
)

HIDDEN_WORKSPACE_DIRS = (".agents", ".codex", ".sandbox")
READ_ONLY_WORKSPACE_PATHS = (".git", ".venv")


def build_workspace_protection_arguments(workspace: Path, mode: SandboxMode) -> list[str]:
    """Build mounts that hide or make protected workspace paths read-only."""
    args: list[str] = []
    for name in HIDDEN_WORKSPACE_DIRS:
        path = workspace / name
        _reject_protected_symlink(path)
        if path.is_dir():
            args.extend(("--tmpfs", str(path)))
        elif path.exists():
            args.extend(("--ro-bind", "/dev/null", str(path)))

    if mode is SandboxMode.WORKSPACE_WRITE:
        for name in READ_ONLY_WORKSPACE_PATHS:
            path = workspace / name
            _reject_protected_symlink(path)
            if path.exists():
                args.extend(("--ro-bind", str(path), str(path)))

    for path in masked_workspace_paths(workspace):
        args.extend(("--ro-bind", "/dev/null", str(path)))
    return args


def masked_workspace_paths(workspace: Path) -> list[Path]:
    """Find sensitive files and sockets that must be hidden from the sandbox."""
    masked: list[Path] = []
    skipped_dirs = set(HIDDEN_WORKSPACE_DIRS)
    for root, dirs, files in os.walk(
        workspace,
        followlinks=False,
        onerror=_raise_walk_error,
    ):
        dirs[:] = [name for name in dirs if name not in skipped_dirs]
        root_path = Path(root)
        for name in files:
            lower_name = name.lower()
            path = root_path / name
            try:
                path_mode = path.lstat().st_mode
            except OSError as exc:
                raise SandboxConfigurationError(
                    f"could not inspect workspace protection path: {path}: {exc}"
                ) from exc

            is_sensitive = (
                lower_name in SENSITIVE_FILE_NAMES
                or path.suffix.lower() in SENSITIVE_FILE_SUFFIXES
            )
            if is_sensitive and stat.S_ISLNK(path_mode):
                raise SandboxConfigurationError(
                    f"protected sandbox path must not be a symlink: {path}"
                )
            if is_sensitive or stat.S_ISSOCK(path_mode):
                masked.append(path)
    return sorted(masked)


def _reject_protected_symlink(path: Path) -> None:
    if path.is_symlink():
        raise SandboxConfigurationError(f"protected sandbox path must not be a symlink: {path}")


def _raise_walk_error(error: OSError) -> None:
    raise SandboxConfigurationError(f"could not inspect workspace protections: {error}") from error
