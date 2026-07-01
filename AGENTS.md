# Project preferences

- Do not add `from __future__ import annotations`; use normal imports and annotation behavior.
- Avoid local imports when a module-level import is possible.

# Testing

- Run the full test suite from the repository root with `uv run python -m unittest discover`.
- See [docs/running-tests.md](docs/running-tests.md) for setup and targeted test commands.
