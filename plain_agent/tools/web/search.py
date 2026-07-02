"""Model-facing web search tool."""

from pathlib import Path

from plain_agent.tools.base_tool import BaseTool
from plain_agent.tools.permissions.controller import (
    ApprovalDeniedError,
    PermissionController,
)
from plain_agent.tools.permissions.network_permission import NetworkPermissionRequest
from plain_agent.tools.utils import error, ok
from plain_agent.tools.web.providers.exa import (
    EXA_DESTINATION,
    ExaSearchClient,
    ExaSearchError,
)

MAX_QUERY_CHARS = 2_000


class WebSearchTool(BaseTool):
    """Search the web after receiving explicit user approval."""

    name = "web_search"
    description = "Search the web and return relevant links with concise excerpts."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_QUERY_CHARS,
                "description": "Natural-language web search query.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        permission_controller: PermissionController | None = None,
        client: ExaSearchClient | None = None,
    ) -> None:
        self.permission_controller = (
            permission_controller
            if permission_controller is not None
            else PermissionController()
        )
        self.client = client if client is not None else ExaSearchClient()

    def run(self, root: Path, arguments: dict[str, object]) -> str:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return error("query is required")
        query = query.strip()
        if len(query) > MAX_QUERY_CHARS:
            return error(f"query must not exceed {MAX_QUERY_CHARS} characters")

        request = NetworkPermissionRequest(
            tool=self.name,
            destination=EXA_DESTINATION,
            target=query,
        )
        try:
            self.permission_controller.require_approval(request)
            content = self.client.search(query)
        except ApprovalDeniedError:
            return error("web_search was not approved")
        except ExaSearchError as exc:
            return error(str(exc))

        return ok({"query": query, "content": content})
