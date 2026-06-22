"""Allowlist policy for safe workspace commands."""

RUN_COMMAND_ALLOWED_COMMANDS = ("pwd", "ls", "find", "rg", "grep", "cat", "head", "tail", "wc")
RUN_COMMAND_ALLOWED_GIT_SUBCOMMANDS = (
    "status",
    "diff",
    "log",
    "show",
    "branch",
    "rev-parse",
    "ls-files",
    "grep",
)

RUN_COMMAND_ALLOWED_COMMANDS_TEXT = (
    "Allowed commands: "
    f"{', '.join(RUN_COMMAND_ALLOWED_COMMANDS)}. "
    "Allowed git subcommands: "
    f"{', '.join(f'git {subcommand}' for subcommand in RUN_COMMAND_ALLOWED_GIT_SUBCOMMANDS)}."
)
