"""OpenAI-compatible tool definitions."""

from simple_agent.tools.command_policy import RUN_COMMAND_ALLOWED_COMMANDS_TEXT

LIST_FILES_DEFINITION = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "List files and directories inside the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative directory path.",
                    "default": ".",
                }
            },
        },
    },
}

READ_FILE_DEFINITION = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a UTF-8 text file inside the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative file path.",
                }
            },
            "required": ["path"],
        },
    },
}

SEARCH_TEXT_DEFINITION = {
    "type": "function",
    "function": {
        "name": "search_text",
        "description": "Search for exact text inside workspace files.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Exact text to search for.",
                },
                "path": {
                    "type": "string",
                    "description": "Workspace-relative file or directory path.",
                    "default": ".",
                },
            },
            "required": ["query"],
        },
    },
}

WRITE_FILE_DEFINITION = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Create or overwrite a file inside the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative file path.",
                },
                "content": {
                    "type": "string",
                    "description": "Full file content.",
                },
            },
            "required": ["path", "content"],
        },
    },
}

EDIT_FILE_DEFINITION = {
    "type": "function",
    "function": {
        "name": "edit_file",
        "description": "Replace an exact string inside an existing workspace file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative file path.",
                },
                "old_string": {
                    "type": "string",
                    "description": "Exact text to replace.",
                },
                "new_string": {
                    "type": "string",
                    "description": "Replacement text.",
                },
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
}

RUN_COMMAND_DEFINITION = {
    "type": "function",
    "function": {
        "name": "run_command",
        "description": (
            "Run a safe inspection command inside the workspace. "
            f"{RUN_COMMAND_ALLOWED_COMMANDS_TEXT}"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "Simple command to run. Shell syntax is not supported. "
                        f"{RUN_COMMAND_ALLOWED_COMMANDS_TEXT}"
                    ),
                },
            },
            "required": ["command"],
        },
    },
}
