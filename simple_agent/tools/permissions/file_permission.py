"""File permission helpers for workspace tools."""

from pathlib import Path

IGNORED_DIRS = {".agents", ".codex", ".git", ".sandbox", ".venv", "__pycache__"}
SENSITIVE_FILE_NAMES = {".env", "id_dsa", "id_ecdsa", "id_ed25519", "id_rsa"}
SENSITIVE_FILE_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}


class FilePermissionError(PermissionError):
    """Raised when a tool cannot safely access a workspace path."""


class WorkspacePermission:
    """Checks whether paths are safe to access inside a workspace."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()

    def _resolve_to_absolute_path(self, path: Path) -> Path:
        try:
            if path.is_absolute():
                return path.resolve()
            return (self.workspace / path).resolve()
        except (OSError, RuntimeError) as exc:
            raise FilePermissionError(f"path could not be resolved: {path}") from exc

    def is_sensitive_path(self, relative_path: Path) -> bool:
        for part in relative_path.parts:
            lower_part = part.lower()
            if lower_part in IGNORED_DIRS or lower_part in SENSITIVE_FILE_NAMES:
                return True
        return relative_path.suffix.lower() in SENSITIVE_FILE_SUFFIXES

    def require_access(self, path: str | Path, must_exist: bool = True) -> Path:
        try:
            user_path = Path(path)
        except TypeError as exc:
            raise FilePermissionError("path must be a string") from exc

        absolute_path = self._resolve_to_absolute_path(user_path)
        self._check_inside_workspace(absolute_path, path)
        self._check_not_sensitive(absolute_path, path)
        self._check_exists(absolute_path, path, must_exist=must_exist)
        return absolute_path

    def contains_path(self, path: Path) -> bool:
        return path == self.workspace or self.workspace in path.parents

    def get_files_under_path(self, path: Path) -> list[Path]:
        if path.is_file():
            return [path]
        return self._walk_files(path)

    def _walk_files(self, path: Path) -> list[Path]:
        """Collect workspace files recursively, skipping symlinks and blocked paths."""
        files: list[Path] = []
        for child in sorted(path.iterdir()):
            if child.is_symlink():
                continue

            relative_path = child.relative_to(self.workspace)
            if child.name in IGNORED_DIRS or self.is_sensitive_path(relative_path):
                continue

            if child.is_dir():
                files += self._walk_files(child)
            elif child.is_file():
                files.append(child)
        return files

    def _check_inside_workspace(self, absolute_path: Path, original_path: str | Path) -> None:
        if not self.contains_path(absolute_path):
            raise FilePermissionError(f"path is outside workspace: {original_path}")

    def _check_not_sensitive(self, absolute_path: Path, original_path: str | Path) -> None:
        if absolute_path != self.workspace and self.is_sensitive_path(absolute_path.relative_to(self.workspace)):
            raise FilePermissionError(f"path is blocked: {original_path}")

    def _check_exists(
        self,
        absolute_path: Path,
        original_path: str | Path,
        must_exist: bool = True,
    ) -> None:
        try:
            path_exists = absolute_path.exists()
        except OSError as exc:
            raise FilePermissionError(f"path could not be checked: {original_path}") from exc
        if must_exist and not path_exists:
            raise FilePermissionError(f"path does not exist: {original_path}")

    def relative_to_workspace(self, path: Path) -> str:
        return str(path.relative_to(self.workspace))
