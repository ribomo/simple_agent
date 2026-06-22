"""Tool for searching workspace text."""

from pathlib import Path

from simple_agent.tools.definitions import SEARCH_TEXT_DEFINITION
from simple_agent.tools.base_tool import BaseTool
from simple_agent.tools.permissions.file_permission import FilePermissionError, WorkspacePermission
from simple_agent.tools.utils import error, ok


class SearchTextTool(BaseTool):
    """Search for exact text inside workspace files."""

    name = "search_text"
    definition = SEARCH_TEXT_DEFINITION

    def __init__(self, max_results: int = 20) -> None:
        self.max_results = max_results

    def run(self, root: Path, arguments: dict[str, object]) -> str:
        query = arguments.get("query")
        if not isinstance(query, str) or not query:
            return error("query is required")

        path = str(arguments.get("path", "."))
        try:
            permissions = WorkspacePermission(root)
            workspace_path = permissions.require_access(path)
        except FilePermissionError as exc:
            return error(str(exc))

        files = permissions.get_files_under_path(workspace_path)
        results = []
        for file_path in files:
            try:
                lines = file_path.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, OSError):
                continue

            for line_number, line in enumerate(lines, start=1):
                if query in line:
                    results.append(
                        {
                            "path": permissions.relative_to_workspace(file_path),
                            "line": line_number,
                            "text": line.strip(),
                        }
                    )
                    if len(results) >= self.max_results:
                        return ok({"query": query, "results": results, "truncated": True})

        return ok({"query": query, "results": results, "truncated": False})
