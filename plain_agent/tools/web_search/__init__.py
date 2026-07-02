"""Approved web search tools and provider clients."""

from plain_agent.tools.web_search.exa_mcp import ExaSearchClient, ExaSearchError
from plain_agent.tools.web_search.tool import WebSearchTool

__all__ = [
    "ExaSearchClient",
    "ExaSearchError",
    "WebSearchTool",
]
