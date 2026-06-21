"""Tool for reading workspace files."""

from pathlib import Path

from simple_agent.tools.definitions import READ_FILE_DEFINITION
from simple_agent.tools.base_tool import BaseTool
from simple_agent.tools.utils import ToolError, error, get_workspace_path, ok


class ReadFileTool(BaseTool):
    """Read a UTF-8 text file inside the workspace."""

    name = "read_file"
    definition = READ_FILE_DEFINITION

    def __init__(self, max_chars: int = 12_000) -> None:
        self.max_chars = max_chars

    def run(self, root: Path, arguments: dict[str, object]) -> str:
        path = arguments.get("path")
        if not isinstance(path, str):
            return error("path is required")

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

        truncated = len(text) > self.max_chars
        if truncated:
            text = text[: self.max_chars]

        return ok(
            {
                "path": str(workspace_path.relative_to(root)),
                "content": text,
                "truncated": truncated,
            }
        )
