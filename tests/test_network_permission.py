import unittest

from plain_agent.tools.permissions.network_permission import NetworkPermissionRequest


class NetworkPermissionRequestTest(unittest.TestCase):
    def test_display_escapes_untrusted_text(self) -> None:
        request = NetworkPermissionRequest(
            tool="web_search",
            destination="mcp.exa.ai",
            target="query\x1b[2K\rspoof\\text",
        )

        self.assertEqual(request.tool, "web_search")
        self.assertEqual(request.destination, "mcp.exa.ai")
        self.assertEqual(request.target, "query\x1b[2K\rspoof\\text")
        self.assertEqual(request.display, r"query\x1b[2K\rspoof\\text")
        self.assertTrue(request.display.isprintable())


if __name__ == "__main__":
    unittest.main()
