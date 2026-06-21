"""Base class for callable tools."""

from pathlib import Path


class BaseTool:
    """Base class for a tool the model can call."""

    name = ""
    definition: dict[str, object]

    def run(self, root: Path, arguments: dict[str, object]) -> str:
        raise NotImplementedError
