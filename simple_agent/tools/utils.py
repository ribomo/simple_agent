"""Shared helpers for workspace tools."""

import json


def ok(data: dict[str, object]) -> str:
    return json.dumps({"ok": True, **data})


def error(message: str) -> str:
    return json.dumps({"ok": False, "error": message})
