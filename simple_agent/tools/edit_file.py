"""Tool for editing workspace files with exact string replacement."""

from pathlib import Path

from simple_agent.tools.definitions import EDIT_FILE_DEFINITION
from simple_agent.tools.base_tool import BaseTool
from simple_agent.tools.utils import ToolError, error, get_workspace_path, ok


class EditFileTool(BaseTool):
    """Replace an exact string inside an existing workspace file."""

    name = "edit_file"
    definition = EDIT_FILE_DEFINITION

    def run(self, root: Path, arguments: dict[str, object]) -> str:
        path = arguments.get("path")
        old_string = arguments.get("old_string")
        new_string = arguments.get("new_string")
        if not isinstance(path, str):
            return error("path is required")
        if not isinstance(old_string, str) or not old_string:
            return error("old_string is required")
        if not isinstance(new_string, str):
            return error("new_string is required")

        try:
            workspace_path = get_workspace_path(root, path)
        except ToolError as exc:
            return error(str(exc))

        if not workspace_path.is_file():
            return error(f"path is not a file: {path}")

        try:
            text = workspace_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return error(f"file is not valid UTF-8 text: {path}")
        except OSError as exc:
            return error(f"could not read file: {exc}")

        count = text.count(old_string)
        if count == 0:
            return error(f"old_string not found in {path}")
        if count > 1:
            return error(f"old_string found {count} times in {path} — provide more context")

        try:
            workspace_path.write_text(text.replace(old_string, new_string), encoding="utf-8")
        except OSError as exc:
            return error(f"could not write file: {exc}")

        return ok({
            "path": str(workspace_path.relative_to(root)),
            "replaced": True,
        })
