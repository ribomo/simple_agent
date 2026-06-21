"""Tool for creating or overwriting workspace files."""

from pathlib import Path

from simple_agent.tools.definitions import WRITE_FILE_DEFINITION
from simple_agent.tools.base_tool import BaseTool
from simple_agent.tools.utils import ToolError, error, get_workspace_path, ok


class WriteFileTool(BaseTool):
    """Create or overwrite a file inside the workspace."""

    name = "write_file"
    definition = WRITE_FILE_DEFINITION

    def run(self, root: Path, arguments: dict[str, object]) -> str:
        path = arguments.get("path")
        content = arguments.get("content")
        if not isinstance(path, str):
            return error("path is required")
        if not isinstance(content, str) or not content:
            return error("content is required")

        try:
            workspace_path = get_workspace_path(root, path, must_exist=False)
        except ToolError as exc:
            return error(str(exc))

        if workspace_path.is_dir():
            return error(f"path is not a file: {path}")

        try:
            workspace_path.write_text(content, encoding="utf-8")
        except OSError as exc:
            return error(f"could not write file: {exc}")

        return ok({
            "path": str(workspace_path.relative_to(root)),
            "written": len(content),
        })
