"""Permission requests for tools that access the network."""

from dataclasses import dataclass

from plain_agent.display import escape_display_text


@dataclass(frozen=True)
class NetworkPermissionRequest:
    """A validated network action that requires user approval."""

    tool: str
    destination: str
    target: str

    @property
    def display(self) -> str:
        return escape_display_text(self.target)
