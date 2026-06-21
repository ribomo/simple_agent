"""Shared helpers for workspace tools."""

import json
from pathlib import Path

IGNORED_DIRS = {".agents", ".codex", ".git", ".sandbox", ".venv", "__pycache__"}
SENSITIVE_FILE_NAMES = {".env", "id_dsa", "id_ecdsa", "id_ed25519", "id_rsa"}
SENSITIVE_FILE_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}


class ToolError(ValueError):
    """Raised when a tool cannot safely handle the request."""


def ok(data: dict[str, object]) -> str:
    return json.dumps({"ok": True, **data})


def error(message: str) -> str:
    return json.dumps({"ok": False, "error": message})


def is_sensitive_path(relative_path: Path) -> bool:
    for part in relative_path.parts:
        lower_part = part.lower()
        if lower_part in IGNORED_DIRS or lower_part in SENSITIVE_FILE_NAMES:
            return True
    return relative_path.suffix.lower() in SENSITIVE_FILE_SUFFIXES


def get_workspace_path(root: Path, path: str, must_exist: bool = True) -> Path:
    workspace_path = (root / path).resolve()
    # Keep tool access inside the configured workspace directory.
    if workspace_path != root and root not in workspace_path.parents:
        raise ToolError(f"path is outside workspace: {path}")
    if workspace_path != root and is_sensitive_path(workspace_path.relative_to(root)):
        raise ToolError(f"path is blocked: {path}")
    if must_exist and not workspace_path.exists():
        raise ToolError(f"path does not exist: {path}")
    return workspace_path


def walk_workspace(path: Path, ignored_dirs: set[str]) -> list[Path]:
    """Collect files recursively using DFS, skipping symlinks and ignored dirs."""
    files: list[Path] = []
    for child in sorted(path.iterdir()):
        if child.is_symlink():
            continue
        if child.name in ignored_dirs or is_sensitive_path(child.relative_to(path)):
            continue
        if child.is_dir():
            files += walk_workspace(child, ignored_dirs)
        elif child.is_file():
            files.append(child)
    return files


def get_files_under_path(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return walk_workspace(path, IGNORED_DIRS)
