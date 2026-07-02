"""Model-facing webpage fetch tool."""

from pathlib import Path
from urllib.parse import urlsplit

from plain_agent.tools.base_tool import BaseTool
from plain_agent.tools.permissions.controller import (
    ApprovalDeniedError,
    PermissionController,
)
from plain_agent.tools.permissions.network_permission import NetworkPermissionRequest
from plain_agent.tools.utils import error, ok
from plain_agent.tools.web.providers.exa import (
    EXA_DESTINATION,
    ExaFetchClient,
    ExaFetchError,
)

MAX_URL_CHARS = 2_000


class WebFetchTool(BaseTool):
    """Fetch a webpage through Exa after explicit user approval."""

    name = "web_fetch"
    description = "Fetch a webpage and return its clean, bounded Markdown content."
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_URL_CHARS,
                "description": "Absolute HTTP or HTTPS URL to fetch.",
            },
        },
        "required": ["url"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        permission_controller: PermissionController | None = None,
        client: ExaFetchClient | None = None,
    ) -> None:
        self.permission_controller = (
            permission_controller
            if permission_controller is not None
            else PermissionController()
        )
        self.client = client if client is not None else ExaFetchClient()

    def run(self, root: Path, arguments: dict[str, object]) -> str:
        url = arguments.get("url")
        validation_error = _validate_url(url)
        if validation_error is not None:
            return error(validation_error)
        assert isinstance(url, str)
        url = url.strip()

        request = NetworkPermissionRequest(
            tool=self.name,
            destination=EXA_DESTINATION,
            target=url,
        )
        try:
            self.permission_controller.require_approval(request)
            content = self.client.fetch(url)
        except ApprovalDeniedError:
            return error("web_fetch was not approved")
        except ExaFetchError as exc:
            return error(str(exc))

        return ok({"url": url, "content": content})


def _validate_url(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return "url is required"

    url = value.strip()
    if len(url) > MAX_URL_CHARS:
        return f"url must not exceed {MAX_URL_CHARS} characters"
    if any(character.isspace() or not character.isprintable() for character in url):
        return "url must not contain whitespace or control characters"

    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        return "url is invalid"

    if parsed.scheme not in {"http", "https"}:
        return "url must use http or https"
    if parsed.hostname is None:
        return "url must include a hostname"
    if parsed.username is not None or parsed.password is not None:
        return "url must not include credentials"
    if port is not None and not 1 <= port <= 65535:
        return "url is invalid"
    return None
