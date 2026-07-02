import json
from pathlib import Path
import unittest

import httpx

from plain_agent.tools.permissions.controller import PermissionController
from plain_agent.tools.permissions.request import ApprovalDecision
from plain_agent.tools.web import WebFetchTool, WebSearchTool
from plain_agent.tools.web.providers import (
    ExaFetchClient,
    ExaSearchClient,
)


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


class ExaFetchClientTest(unittest.TestCase):
    def test_fetch_sends_mcp_request_and_returns_content(self) -> None:
        captured_requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_requests.append(request)
            return httpx.Response(200, json=mcp_payload("# Example\nContent"))

        client = ExaFetchClient(transport=httpx.MockTransport(handler))

        content = client.fetch("https://example.com/page")

        request = captured_requests[0]
        self.assertEqual(str(request.url), "https://mcp.exa.ai/mcp")
        self.assertNotIn("x-api-key", request.headers)
        self.assertEqual(
            json.loads(request.content),
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "web_fetch_exa",
                    "arguments": {
                        "urls": ["https://example.com/page"],
                        "maxCharacters": 12_000,
                    },
                },
            },
        )
        self.assertEqual(content, "# Example\nContent")

    def test_fetch_bounds_returned_content(self) -> None:
        client = ExaFetchClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json=mcp_payload("x" * 13_000),
                )
            )
        )

        self.assertEqual(len(client.fetch("https://example.com")), 12_000)

    def test_fetch_reports_http_status_without_response_body(self) -> None:
        client = ExaFetchClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(500, text="secret response body")
            )
        )

        with self.assertRaisesRegex(RuntimeError, "HTTP status 500") as raised:
            client.fetch("https://example.com")

        self.assertNotIn("secret response body", str(raised.exception))


class WebFetchToolTest(unittest.TestCase):
    def test_denial_does_not_send_request(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(200, json=mcp_payload("result"))

        tool = WebFetchTool(
            permission_controller(ApprovalDecision.REJECT),
            ExaFetchClient(transport=httpx.MockTransport(handler)),
        )

        result = json.loads(tool.run(Path.cwd(), {"url": "https://example.com"}))

        self.assertFalse(result["ok"])
        self.assertIn("not approved", result["error"])
        self.assertEqual(calls, 0)

    def test_approved_fetch_returns_tool_result_and_approval_target(self) -> None:
        approval_requests = []

        def approve(request):
            approval_requests.append(request)
            return ApprovalDecision.ALLOW_ONCE

        client = ExaFetchClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=mcp_payload("content"))
            )
        )
        tool = WebFetchTool(PermissionController(approve), client)

        result = json.loads(
            tool.run(Path.cwd(), {"url": "  https://example.com/page  "})
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["url"], "https://example.com/page")
        self.assertEqual(result["content"], "content")
        self.assertEqual(approval_requests[0].tool, "web_fetch")
        self.assertEqual(approval_requests[0].destination, "mcp.exa.ai")
        self.assertEqual(approval_requests[0].target, "https://example.com/page")

    def test_invalid_urls_are_rejected_before_approval(self) -> None:
        approval_requests = []

        def approve(request):
            approval_requests.append(request)
            return ApprovalDecision.ALLOW_ONCE

        tool = WebFetchTool(PermissionController(approve))
        invalid_urls = [
            "",
            "example.com",
            "file:///etc/passwd",
            "https://user:password@example.com",
            "https://example.com/a b",
            "https://[invalid",
            "https://example.com:99999",
            "https://example.com/" + "x" * 2_000,
        ]

        for url in invalid_urls:
            with self.subTest(url=url):
                result = json.loads(tool.run(Path.cwd(), {"url": url}))
                self.assertFalse(result["ok"])

        self.assertEqual(approval_requests, [])


if __name__ == "__main__":
    unittest.main()
