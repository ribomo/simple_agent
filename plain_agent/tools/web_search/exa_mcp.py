"""Client for Exa's public MCP web-search endpoint."""

import json

import httpx

EXA_MCP_URL = "https://mcp.exa.ai/mcp"
EXA_DESTINATION = "mcp.exa.ai"
EXA_SEARCH_TOOL = "web_search_exa"
MAX_RESULTS = 5
MAX_CONTENT_CHARS = 5_000
MAX_RESPONSE_BYTES = 1024 * 1024


class ExaSearchError(RuntimeError):
    """Raised when Exa cannot provide a safe, valid search response."""


class ExaSearchClient:
    """Small synchronous client for Exa's fixed MCP endpoint."""

    def __init__(
        self,
        timeout_seconds: float = 25.0,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_response_bytes < 1:
            raise ValueError("max_response_bytes must be positive")

        self.timeout = httpx.Timeout(timeout_seconds, connect=5.0)
        self.max_response_bytes = max_response_bytes
        self.transport = transport

    def search(self, query: str) -> str:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": EXA_SEARCH_TOOL,
                "arguments": {
                    "query": query,
                    "type": "auto",
                    "numResults": MAX_RESULTS,
                    "livecrawl": "fallback",
                    "contextMaxCharacters": MAX_CONTENT_CHARS,
                },
            },
        }

        try:
            with httpx.Client(
                timeout=self.timeout,
                follow_redirects=False,
                transport=self.transport,
            ) as client:
                with client.stream(
                    "POST",
                    EXA_MCP_URL,
                    headers={"Accept": "application/json, text/event-stream"},
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    body = self._read_response(response)
        except httpx.TimeoutException as exc:
            raise ExaSearchError("web search timed out") from exc
        except httpx.HTTPStatusError as exc:
            raise ExaSearchError(
                f"web search failed with HTTP status {exc.response.status_code}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ExaSearchError("web search request failed") from exc

        return self._parse_response(body)

    def _read_response(self, response: httpx.Response) -> bytes:
        content_length = response.headers.get("content-length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError:
                declared_length = None
            if (
                declared_length is not None
                and declared_length > self.max_response_bytes
            ):
                raise ExaSearchError("web search response exceeded the size limit")

        body = bytearray()
        for chunk in response.iter_bytes():
            if len(body) + len(chunk) > self.max_response_bytes:
                raise ExaSearchError("web search response exceeded the size limit")
            body.extend(chunk)
        return bytes(body)

    def _parse_response(self, body: bytes) -> str:
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ExaSearchError("web search returned invalid JSON") from exc

        candidates = [text.strip()]
        candidates.extend(
            line.removeprefix("data: ")
            for line in text.splitlines()
            if line.startswith("data: ")
        )
        parsed_payload = False
        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            parsed_payload = True
            content = _text_content(payload)
            if content is not None:
                return content[:MAX_CONTENT_CHARS]

        if not parsed_payload:
            raise ExaSearchError("web search returned invalid JSON")
        raise ExaSearchError("web search returned an invalid response")


def _text_content(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    result = payload.get("result")
    if not isinstance(result, dict):
        return None
    items = result.get("content")
    if not isinstance(items, list):
        return None
    for item in items:
        if not isinstance(item, dict) or item.get("type") != "text":
            continue
        text = item.get("text")
        if isinstance(text, str):
            return text
    return None
