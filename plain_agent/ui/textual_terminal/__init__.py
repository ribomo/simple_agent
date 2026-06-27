"""Full-screen Textual terminal UI."""

from plain_agent.agent_loop import SimpleAgent
from plain_agent.ui.textual_terminal.app import PlainAgentTextualApp


def run_textual_terminal(agent: SimpleAgent) -> None:
    """Run a full-screen terminal loop with Textual."""
    PlainAgentTextualApp(agent).run()
