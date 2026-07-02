import json
from pathlib import Path
import unittest

import httpx

from plain_agent.tools.permissions.controller import PermissionController
from plain_agent.tools.permissions.request import ApprovalDecision
from plain_agent.tools.web_search import ExaSearchClient, WebSearchTool


def permission_controller(decision: ApprovalDecision) -> PermissionController:
    return PermissionController(lambda request: decision)


def mcp_payload(text: str) -> dict[str, object]:
    return {
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": text,
                }
            ]
        },
        "jsonrpc": "2.0",
        "id": 1,
    }


class ExaSearchClientTest(unittest.TestCase):
    def test_search_sends_mcp_request_and_parses_event_stream(self) -> None:
        captured_requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_requests.append(request)
            payload = json.dumps(mcp_payload("Title: Python\nURL: https://python.org"))
            return httpx.Response(200, text=f"event: message\ndata: {payload}\n\n")

        client = ExaSearchClient(transport=httpx.MockTransport(handler))

        content = client.search("current Python release")

        request = captured_requests[0]
        self.assertEqual(request.method, "POST")
        self.assertEqual(str(request.url), "https://mcp.exa.ai/mcp")
        self.assertEqual(request.headers["content-type"], "application/json")
        self.assertNotIn("x-api-key", request.headers)
        self.assertEqual(
            json.loads(request.content),
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "web_search_exa",
                    "arguments": {
                        "query": "current Python release",
                        "type": "auto",
                        "numResults": 5,
                        "livecrawl": "fallback",
                        "contextMaxCharacters": 5000,
                    },
                },
            },
        )
        self.assertEqual(content, "Title: Python\nURL: https://python.org")

    def test_search_parses_direct_json_response(self) -> None:
        client = ExaSearchClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=mcp_payload("result"))
            )
        )

        self.assertEqual(client.search("query"), "result")

    def test_search_reports_timeout_without_request_details(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("included-sensitive-detail", request=request)

        client = ExaSearchClient(transport=httpx.MockTransport(handler))

        with self.assertRaisesRegex(RuntimeError, "web search timed out") as raised:
            client.search("query")

        self.assertNotIn("sensitive", str(raised.exception))

    def test_search_reports_connection_failure_without_request_details(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("included-sensitive-detail", request=request)

        client = ExaSearchClient(transport=httpx.MockTransport(handler))

        with self.assertRaisesRegex(RuntimeError, "web search request failed") as raised:
            client.search("query")

        self.assertNotIn("sensitive", str(raised.exception))

    def test_search_reports_http_status_without_response_body(self) -> None:
        client = ExaSearchClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(429, text="secret response body")
            )
        )

        with self.assertRaisesRegex(RuntimeError, "HTTP status 429") as raised:
            client.search("query")

        self.assertNotIn("secret response body", str(raised.exception))

    def test_search_rejects_malformed_json(self) -> None:
        client = ExaSearchClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, content=b"not-json")
            )
        )

        with self.assertRaisesRegex(RuntimeError, "invalid JSON"):
            client.search("query")

    def test_search_rejects_json_without_text_content(self) -> None:
        client = ExaSearchClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"result": {}})
            )
        )

        with self.assertRaisesRegex(RuntimeError, "invalid response"):
            client.search("query")

    def test_search_rejects_oversized_response(self) -> None:
        client = ExaSearchClient(
            max_response_bytes=5,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, content=b"123456")
            ),
        )

        with self.assertRaisesRegex(RuntimeError, "size limit"):
            client.search("query")

    def test_search_bounds_returned_content(self) -> None:
        client = ExaSearchClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=mcp_payload("x" * 6_000))
            )
        )

        self.assertEqual(len(client.search("query")), 5_000)


class WebSearchToolTest(unittest.TestCase):
    def test_denial_does_not_send_request(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(200, json=mcp_payload("result"))

        client = ExaSearchClient(transport=httpx.MockTransport(handler))
        tool = WebSearchTool(
            permission_controller(ApprovalDecision.REJECT),
            client,
        )

        result = json.loads(tool.run(Path.cwd(), {"query": "query"}))

        self.assertFalse(result["ok"])
        self.assertIn("not approved", result["error"])
        self.assertEqual(calls, 0)

    def test_approved_search_returns_tool_result(self) -> None:
        client = ExaSearchClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=mcp_payload("result"))
            )
        )
        tool = WebSearchTool(
            permission_controller(ApprovalDecision.ALLOW_ONCE),
            client,
        )

        result = json.loads(tool.run(Path.cwd(), {"query": "  query  "}))

        self.assertTrue(result["ok"])
        self.assertEqual(result["query"], "query")
        self.assertEqual(result["content"], "result")

    def test_invalid_query_is_rejected_before_approval(self) -> None:
        requests = []

        def approve(request):
            requests.append(request)
            return ApprovalDecision.ALLOW_ONCE

        tool = WebSearchTool(PermissionController(approve))

        empty = json.loads(tool.run(Path.cwd(), {"query": "   "}))
        oversized = json.loads(tool.run(Path.cwd(), {"query": "x" * 2001}))

        self.assertFalse(empty["ok"])
        self.assertFalse(oversized["ok"])
        self.assertEqual(requests, [])


if __name__ == "__main__":
    unittest.main()
