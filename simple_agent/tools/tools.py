"""Dispatcher for workspace tools."""

from pathlib import Path

from simple_agent.tools.base_tool import BaseTool
from simple_agent.tools.utils import error
from simple_agent.tools.list_files import ListFilesTool
from simple_agent.tools.read_file import ReadFileTool
from simple_agent.tools.search_text import SearchTextTool
from simple_agent.tools.write_file import WriteFileTool
from simple_agent.tools.edit_file import EditFileTool
from simple_agent.tools.run_command import RunCommandTool


class Tools:
    """Small tool dispatcher scoped to a workspace directory."""

    def __init__(
        self,
        root: str | Path = ".",
        max_read_chars: int = 12_000,
        max_search_results: int = 20,
    ) -> None:
        self.root = Path(root).resolve()
        self.max_read_chars = max_read_chars
        self.max_search_results = max_search_results
        self.tools: dict[str, BaseTool] = {
            tool.name: tool
            for tool in [
                ListFilesTool(),
                ReadFileTool(max_chars=self.max_read_chars),
                SearchTextTool(max_results=self.max_search_results),
                WriteFileTool(),
                EditFileTool(),
                RunCommandTool(),
            ]
        }

    def definitions(self) -> list[dict[str, object]]:
        return [tool.definition for tool in self.tools.values()]

    def run(self, name: str, arguments: dict[str, object]) -> str:
        tool = self.tools.get(name)
        if tool is None:
            return error(f"unknown tool: {name}")
        return tool.run(self.root, arguments)
