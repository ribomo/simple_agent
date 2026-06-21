"""Tool for listing workspace files."""

from pathlib import Path

from simple_agent.tools.definitions import LIST_FILES_DEFINITION
from simple_agent.tools.base_tool import BaseTool
from simple_agent.tools.utils import ToolError, error, get_workspace_path, is_sensitive_path, ok


class ListFilesTool(BaseTool):
    """List files and directories inside the workspace."""

    name = "list_files"
    definition = LIST_FILES_DEFINITION

    def run(self, root: Path, arguments: dict[str, object]) -> str:
        path = arguments.get("path", ".")
        try:
            workspace_path = get_workspace_path(root, path)
        except ToolError as exc:
            return error(str(exc))

        if not workspace_path.is_dir():
            return error(f"path is not a directory: {path}")

        entries = []
        for child in sorted(workspace_path.iterdir()):
            if is_sensitive_path(child.relative_to(root)):
                continue
            suffix = "/" if child.is_dir() else ""
            entries.append(f"{child.relative_to(root)}{suffix}")

        return ok({"path": str(workspace_path.relative_to(root)), "entries": entries})
