import json
import os
import subprocess
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import mock_open, patch

from plain_agent.sandbox import CommandRequest
from plain_agent.sandbox.bubblewrap import SandboxDiscovery
from plain_agent.tools.base_tool import BaseTool
from plain_agent.tools.list_files import ListFilesTool
from plain_agent.tools.read_file import ReadFileTool
from plain_agent.tools.search_text import SearchTextTool
from plain_agent.tools.write_file import WriteFileTool
from plain_agent.tools.edit_file import EditFileTool
from plain_agent.tools.run_command import RunCommandTool
from plain_agent.tools.command_runtime import CommandRuntime
from plain_agent.tools.permissions.controller import PermissionController
from plain_agent.tools.permissions.request import ApprovalDecision, CommandPermissionRequest
from plain_agent.tools.registry import ToolRegistry


class PassthroughSandbox:
    """Test backend that makes runtime behavior observable without Bubblewrap."""

    def __init__(self) -> None:
        self.requests: list[CommandRequest] = []

    def build_command(self, request: CommandRequest) -> list[str]:
        self.requests.append(request)
        return list(request.argv)


def approve_request(request: CommandPermissionRequest) -> ApprovalDecision:
    return ApprovalDecision.ALLOW_ONCE


def approved_permissions() -> PermissionController:
    return PermissionController(approve_request)


def run_command_tool(sandbox, **kwargs) -> RunCommandTool:
    return RunCommandTool(sandbox, approved_permissions(), **kwargs)


def command_arguments(argv, **kwargs) -> dict[str, object]:
    return {
        "argv": argv,
        "justification": "Exercise command execution",
        **kwargs,
    }


class ToolRegistryTest(unittest.TestCase):
    def test_base_tool_is_abstract(self) -> None:
        with self.assertRaises(TypeError):
            BaseTool()

    def test_read_file_returns_file_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "notes.txt").write_text("hello from a file", encoding="utf-8")
            tools = ToolRegistry(root=root)

            result = json.loads(tools.run("read_file", {"path": "notes.txt"}))

            self.assertTrue(result["ok"])
            self.assertEqual(result["path"], "notes.txt")
            self.assertEqual(result["content"], "hello from a file")
            self.assertFalse(result["truncated"])

    def test_read_file_reports_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tools = ToolRegistry(root=temp_dir)

            result = json.loads(tools.run("read_file", {"path": "missing.txt"}))

            self.assertFalse(result["ok"])
            self.assertIn("does not exist", result["error"])

    def test_read_file_only_reads_through_the_character_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "large.txt"
            path.write_text("placeholder", encoding="utf-8")
            opened = mock_open(read_data="abcdef")

            with patch.object(Path, "open", opened):
                result = json.loads(
                    ReadFileTool(max_chars=3).run(root, {"path": "large.txt"})
                )

            opened().read.assert_called_once_with(4)
            self.assertEqual(result["content"], "abc")
            self.assertTrue(result["truncated"])

    def test_rejects_paths_outside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tools = ToolRegistry(root=temp_dir)

            result = json.loads(tools.run("read_file", {"path": "../outside.txt"}))

            self.assertFalse(result["ok"])
            self.assertIn("outside workspace", result["error"])

    def test_absolute_paths_must_stay_inside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inside = root / "notes.txt"
            inside.write_text("inside", encoding="utf-8")
            tools = ToolRegistry(root=root)

            inside_result = json.loads(tools.run("read_file", {"path": str(inside)}))
            outside_result = json.loads(tools.run("read_file", {"path": "/tmp/outside.txt"}))

            self.assertTrue(inside_result["ok"])
            self.assertEqual(inside_result["content"], "inside")
            self.assertFalse(outside_result["ok"])
            self.assertIn("outside workspace", outside_result["error"])

    def test_symlink_targets_outside_workspace_are_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            outside = root.parent / "outside-permission-target.txt"
            outside.write_text("secret", encoding="utf-8")
            (root / "link.txt").symlink_to(outside)
            tools = ToolRegistry(root=root)

            try:
                result = json.loads(tools.run("read_file", {"path": "link.txt"}))
            finally:
                outside.unlink(missing_ok=True)

            self.assertFalse(result["ok"])
            self.assertIn("outside workspace", result["error"])

    def test_list_files_rejects_invalid_path_argument(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tools = ToolRegistry(root=temp_dir)

            result = json.loads(tools.run("list_files", {"path": 123}))

            self.assertFalse(result["ok"])
            self.assertIn("path must be a string", result["error"])

    def test_sensitive_paths_are_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ".env").write_text("SECRET=value\n", encoding="utf-8")
            (root / ".env.example").write_text("PLACEHOLDER=value\n", encoding="utf-8")
            (root / "private.key").write_text("SECRET=value\n", encoding="utf-8")
            (root / ".agents").mkdir()
            (root / ".agents" / "notes.txt").write_text("SECRET=value\n", encoding="utf-8")
            tools = ToolRegistry(root=root)

            read_env = json.loads(tools.run("read_file", {"path": ".env"}))
            read_env_example = json.loads(tools.run("read_file", {"path": ".env.example"}))
            read_key = json.loads(tools.run("read_file", {"path": "private.key"}))
            write_env_local = json.loads(tools.run("write_file", {"path": ".env.local", "content": "x"}))
            edit_env = json.loads(tools.run("edit_file", {"path": ".env", "old_string": "SECRET", "new_string": "x"}))
            listed = json.loads(tools.run("list_files", {}))
            searched = json.loads(tools.run("search_text", {"query": "SECRET"}))

            self.assertFalse(read_env["ok"])
            self.assertIn("blocked", read_env["error"])
            self.assertTrue(read_env_example["ok"])
            self.assertFalse(read_key["ok"])
            self.assertIn("blocked", read_key["error"])
            self.assertTrue(write_env_local["ok"])
            self.assertFalse(edit_env["ok"])
            self.assertIn("blocked", edit_env["error"])
            self.assertEqual(listed["entries"], [".env.example", ".env.local"])
            self.assertEqual(searched["results"], [])

    def test_search_text_returns_matching_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "one.txt").write_text("alpha\nneedle here\n", encoding="utf-8")
            (root / "two.txt").write_text("nothing\n", encoding="utf-8")
            tools = ToolRegistry(root=root)

            result = json.loads(tools.run("search_text", {"query": "needle"}))

            self.assertTrue(result["ok"])
            self.assertEqual(
                result["results"],
                [{"path": "one.txt", "line": 2, "text": "needle here"}],
            )

    def test_search_text_reports_directory_traversal_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patch.object(Path, "iterdir", side_effect=OSError("denied")):
                result = json.loads(SearchTextTool().run(root, {"query": "needle"}))

            self.assertFalse(result["ok"])
            self.assertIn("could not list directory", result["error"])

    def test_search_text_rejects_invalid_path_argument(self) -> None:
        result = json.loads(
            SearchTextTool().run(Path.cwd(), {"query": "needle", "path": 123})
        )

        self.assertFalse(result["ok"])
        self.assertIn("path must be a string", result["error"])

    def test_list_files_returns_workspace_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "src").mkdir()
            (root / "README.md").write_text("# Test", encoding="utf-8")
            tools = ToolRegistry(root=root)

            result = json.loads(tools.run("list_files", {}))

            self.assertTrue(result["ok"])
            self.assertEqual(result["entries"], ["README.md", "src/"])

    def test_tools_expose_definitions(self) -> None:
        discovery = SandboxDiscovery(PassthroughSandbox(), None)
        with patch(
            "plain_agent.tools.registry.discover_linux_sandbox",
            return_value=discovery,
        ):
            tools = ToolRegistry()

        definitions = tools.definitions()
        run_command_definition = definitions[-1]["function"]

        self.assertEqual(
            [definition["function"]["name"] for definition in definitions],
            [
                "list_files",
                "read_file",
                "search_text",
                "write_file",
                "edit_file",
                "web_search",
                "web_fetch",
                "run_command",
            ],
        )
        self.assertEqual(
            run_command_definition["parameters"]["required"],
            ["argv", "justification"],
        )
        self.assertEqual(
            run_command_definition["parameters"]["properties"]["mode"]["enum"],
            ["read-only", "workspace-write"],
        )

    def test_registry_exposes_web_tools_by_default_with_opt_out(self) -> None:
        with_network = ToolRegistry(enable_commands=False)
        without_network = ToolRegistry(
            enable_commands=False,
            enable_network=False,
        )

        self.assertFalse(without_network.has("web_search"))
        self.assertFalse(without_network.has("web_fetch"))
        self.assertTrue(with_network.has("web_search"))
        self.assertTrue(with_network.has("web_fetch"))

    def test_run_command_runs_argv_in_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sandbox = PassthroughSandbox()
            discovery = SandboxDiscovery(sandbox, None)
            with patch(
                "plain_agent.tools.registry.discover_linux_sandbox",
                return_value=discovery,
            ):
                tools = ToolRegistry(
                    root=root,
                    permission_controller=approved_permissions(),
                )

            result = json.loads(
                tools.run("run_command", command_arguments(["/bin/pwd"]))
            )

            self.assertTrue(result["ok"])
            self.assertEqual(result["exit_code"], 0)
            self.assertEqual(result["stdout"].strip(), str(root))
            self.assertEqual(result["stderr"], "")
            self.assertFalse(result["timed_out"])
            self.assertEqual(sandbox.requests[0].mode.value, "read-only")

    def test_run_command_does_not_inherit_stdin(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tool = run_command_tool(PassthroughSandbox())

            with patch(
                "plain_agent.tools.command_runtime.subprocess.Popen",
                wraps=subprocess.Popen,
            ) as popen:
                result = json.loads(
                    tool.run(Path(temp_dir), command_arguments(["/bin/true"]))
                )

            self.assertTrue(result["ok"])
            self.assertEqual(popen.call_args.kwargs["stdin"], subprocess.DEVNULL)

    def test_run_command_reports_nonzero_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tool = run_command_tool(PassthroughSandbox())

            result = json.loads(
                tool.run(
                    Path(temp_dir),
                    command_arguments(["/bin/sh", "-c", "exit 7"]),
                )
            )

            self.assertFalse(result["ok"])
            self.assertEqual(result["exit_code"], 7)
            self.assertFalse(result["timed_out"])

    def test_run_command_rejects_invalid_argv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tool = run_command_tool(PassthroughSandbox())

            invalid_values = (None, [], [""], [1], ["echo", "bad\x00arg"])
            results = [
                json.loads(tool.run(Path(temp_dir), command_arguments(value)))
                for value in invalid_values
            ]

            self.assertTrue(all(not result["ok"] for result in results))
            self.assertTrue(all("argv" in result["error"] for result in results))

    def test_run_command_requires_justification(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tool = run_command_tool(PassthroughSandbox())

            result = json.loads(
                tool.run(Path(temp_dir), {"argv": ["/bin/true"]})
            )

            self.assertFalse(result["ok"])
            self.assertIn("justification", result["error"])

    def test_run_command_passes_arguments_without_shell_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tool = run_command_tool(PassthroughSandbox())

            result = json.loads(
                tool.run(
                    Path(temp_dir),
                    command_arguments(["/bin/echo", "one | two", ">", "out"]),
                )
            )

            self.assertTrue(result["ok"])
            self.assertEqual(result["stdout"], "one | two > out\n")
            self.assertFalse((Path(temp_dir) / "out").exists())

    def test_run_command_accepts_explicit_shell_argv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tool = run_command_tool(PassthroughSandbox())

            result = json.loads(
                tool.run(
                    Path(temp_dir),
                    command_arguments(
                        ["/bin/bash", "-lc", "printf explicit-shell"]
                    ),
                )
            )

            self.assertTrue(result["ok"])
            self.assertEqual(result["stdout"], "explicit-shell")

    def test_run_command_preserves_requested_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sandbox = PassthroughSandbox()
            tool = run_command_tool(sandbox)

            result = json.loads(
                tool.run(
                    Path(temp_dir),
                    command_arguments(["/bin/true"], mode="workspace-write"),
                )
            )

            self.assertTrue(result["ok"])
            self.assertEqual(sandbox.requests[0].mode.value, "workspace-write")

    def test_run_command_rejects_unknown_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tool = run_command_tool(PassthroughSandbox())

            result = json.loads(
                tool.run(
                    Path(temp_dir),
                    command_arguments(["/bin/true"], mode="unsafe"),
                )
            )

            self.assertFalse(result["ok"])
            self.assertIn("mode", result["error"])

    def test_run_command_times_out(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            fifo_path = Path(temp_dir) / "stream"
            os.mkfifo(fifo_path)
            tool = run_command_tool(PassthroughSandbox(), timeout_seconds=0.001)

            result = json.loads(
                tool.run(
                    Path(temp_dir),
                    command_arguments(["/bin/cat", "stream"]),
                )
            )

            self.assertFalse(result["ok"])
            self.assertIsNone(result["exit_code"])
            self.assertTrue(result["timed_out"])

    def test_run_command_truncates_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "long.txt").write_text("abcdef", encoding="utf-8")
            tool = run_command_tool(PassthroughSandbox(), max_output_chars=3)

            result = json.loads(
                tool.run(root, command_arguments(["/bin/cat", "long.txt"]))
            )

            self.assertTrue(result["ok"])
            self.assertEqual(result["stdout"], "abc")
            self.assertTrue(result["truncated"])

    def test_command_runtime_discards_output_beyond_retention_limit(self) -> None:
        runtime = CommandRuntime(PassthroughSandbox(), max_output_chars=3)
        chunks: list[str] = []
        errors: list[Exception] = []

        runtime._drain_output(StringIO("x" * 100_000), chunks, errors)

        self.assertEqual("".join(chunks), "xxxx")
        self.assertEqual(errors, [])

    def test_command_runtime_limits_must_be_positive(self) -> None:
        constructors = (
            lambda: CommandRuntime(PassthroughSandbox(), timeout_seconds=0),
            lambda: CommandRuntime(PassthroughSandbox(), timeout_seconds=float("nan")),
            lambda: CommandRuntime(PassthroughSandbox(), max_output_chars=0),
        )
        for constructor in constructors:
            with self.subTest(constructor=constructor):
                with self.assertRaisesRegex(ValueError, "positive"):
                    constructor()

    def test_write_file_creates_new_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tool = WriteFileTool()

            result = json.loads(tool.run(root, {"path": "new.txt", "content": "hello"}))

            self.assertTrue(result["ok"])
            self.assertEqual(result["path"], "new.txt")
            self.assertEqual(result["written"], 5)
            self.assertEqual((root / "new.txt").read_text(encoding="utf-8"), "hello")

    def test_write_file_overwrites_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "existing.txt").write_text("old content", encoding="utf-8")
            tool = WriteFileTool()

            result = json.loads(tool.run(root, {"path": "existing.txt", "content": "fresh"}))

            self.assertTrue(result["ok"])
            self.assertEqual((root / "existing.txt").read_text(encoding="utf-8"), "fresh")

    def test_write_file_accepts_empty_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            result = json.loads(
                WriteFileTool().run(root, {"path": "empty.txt", "content": ""})
            )

            self.assertTrue(result["ok"])
            self.assertEqual((root / "empty.txt").read_text(encoding="utf-8"), "")

    def test_file_tool_limits_must_be_positive(self) -> None:
        constructors = (
            lambda: ReadFileTool(max_chars=0),
            lambda: SearchTextTool(max_results=0),
        )
        for constructor in constructors:
            with self.subTest(constructor=constructor):
                with self.assertRaisesRegex(ValueError, "positive"):
                    constructor()

    def test_write_file_rejects_directory_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "sub").mkdir()
            tool = WriteFileTool()

            result = json.loads(tool.run(root, {"path": "sub", "content": "x"}))

            self.assertFalse(result["ok"])
            self.assertIn("not a file", result["error"])

    def test_edit_file_replaces_matching_string(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "target.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
            tool = EditFileTool()

            result = json.loads(tool.run(root, {"path": "target.py", "old_string": "y = 2", "new_string": "y = 99"}))

            self.assertTrue(result["ok"])
            self.assertTrue(result["replaced"])
            self.assertEqual((root / "target.py").read_text(encoding="utf-8"), "x = 1\ny = 99\n")

    def test_edit_file_reports_old_string_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "target.py").write_text("x = 1\n", encoding="utf-8")
            tool = EditFileTool()

            result = json.loads(tool.run(root, {"path": "target.py", "old_string": "no_match", "new_string": "x"}))

            self.assertFalse(result["ok"])
            self.assertIn("not found", result["error"])

    def test_edit_file_reports_multiple_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "target.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
            tool = EditFileTool()

            result = json.loads(tool.run(root, {"path": "target.py", "old_string": "x = 1", "new_string": "x = 2"}))

            self.assertFalse(result["ok"])
            self.assertIn("2 times", result["error"])

    def test_direct_tool_classes_work(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").write_text("needle\n", encoding="utf-8")

            listed = json.loads(ListFilesTool().run(root, {}))
            read = json.loads(ReadFileTool().run(root, {"path": "README.md"}))
            searched = json.loads(SearchTextTool().run(root, {"query": "needle"}))

            self.assertEqual(listed["entries"], ["README.md"])
            self.assertEqual(read["content"], "needle\n")
            self.assertEqual(searched["results"][0]["path"], "README.md")


if __name__ == "__main__":
    unittest.main()
