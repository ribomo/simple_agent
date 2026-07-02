"""Registry and dispatcher for workspace tools."""

from pathlib import Path

from plain_agent.sandbox import discover_linux_sandbox
from plain_agent.tools.base_tool import BaseTool
from plain_agent.tools.edit_file import EditFileTool
from plain_agent.tools.list_files import ListFilesTool
from plain_agent.tools.permissions.controller import PermissionController
from plain_agent.tools.read_file import ReadFileTool
from plain_agent.tools.run_command import RunCommandTool
from plain_agent.tools.search_text import SearchTextTool
from plain_agent.tools.utils import error
from plain_agent.tools.web_search import WebSearchTool
from plain_agent.tools.write_file import WriteFileTool


class ToolRegistry:
    """Registry and dispatcher scoped to a workspace directory."""

    def __init__(
        self,
        root: str | Path = ".",
        max_read_chars: int = 12_000,
        max_search_results: int = 20,
        enable_commands: bool = True,
        permission_controller: PermissionController | None = None,
        enable_network: bool = True,
    ) -> None:
        self.root = Path(root).resolve()
        self.permission_controller = (
            permission_controller
            if permission_controller is not None
            else PermissionController()
        )
        registered_tools: list[BaseTool] = [
            ListFilesTool(),
            ReadFileTool(max_chars=max_read_chars),
            SearchTextTool(max_results=max_search_results),
            WriteFileTool(),
            EditFileTool(),
        ]
        self.startup_warnings: list[str] = []
        if enable_network:
            registered_tools.append(WebSearchTool(self.permission_controller))
        if enable_commands:
            discovery = discover_linux_sandbox()
            if discovery.warning is not None:
                self.startup_warnings.append(discovery.warning)
            if discovery.backend is not None:
                registered_tools.append(
                    RunCommandTool(discovery.backend, self.permission_controller)
                )

        self._tools: dict[str, BaseTool] = {
            tool.name: tool
            for tool in registered_tools
        }

    def definitions(self) -> list[dict[str, object]]:
        return [tool.definition for tool in self._tools.values()]

    def run(self, name: str, arguments: dict[str, object]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return error(f"unknown tool: {name}")
        return tool.run(self.root, arguments)

    def has(self, name: str) -> bool:
        return name in self._tools
