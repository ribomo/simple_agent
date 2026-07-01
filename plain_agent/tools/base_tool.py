"""Base class for callable tools."""

from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path


class BaseTool(ABC):
    """Base class for a tool the model can call."""

    name = ""
    description = ""
    parameters: dict[str, object] = {
        "type": "object",
        "properties": {},
    }

    @property
    def definition(self) -> dict[str, object]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": deepcopy(self.parameters),
            },
        }

    @abstractmethod
    def run(self, root: Path, arguments: dict[str, object]) -> str:
        """Run the tool and return its serialized result."""
