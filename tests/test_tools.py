import json
import os
import tempfile
import unittest
from pathlib import Path

from simple_agent.tools.list_files import ListFilesTool
from simple_agent.tools.read_file import ReadFileTool
from simple_agent.tools.search_text import SearchTextTool
from simple_agent.tools.write_file import WriteFileTool
from simple_agent.tools.edit_file import EditFileTool
from simple_agent.tools.run_command import RunCommandTool
from simple_agent.tools.tools import Tools


class ToolsTest(unittest.TestCase):
    def test_read_file_returns_file_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "notes.txt").write_text("hello from a file", encoding="utf-8")
            tools = Tools(root=root)

            result = json.loads(tools.run("read_file", {"path": "notes.txt"}))

            self.assertTrue(result["ok"])
            self.assertEqual(result["path"], "notes.txt")
            self.assertEqual(result["content"], "hello from a file")
            self.assertFalse(result["truncated"])

    def test_read_file_reports_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tools = Tools(root=temp_dir)

            result = json.loads(tools.run("read_file", {"path": "missing.txt"}))

            self.assertFalse(result["ok"])
            self.assertIn("does not exist", result["error"])

    def test_rejects_paths_outside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tools = Tools(root=temp_dir)

            result = json.loads(tools.run("read_file", {"path": "../outside.txt"}))

            self.assertFalse(result["ok"])
            self.assertIn("outside workspace", result["error"])

    def test_absolute_paths_must_stay_inside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inside = root / "notes.txt"
            inside.write_text("inside", encoding="utf-8")
            tools = Tools(root=root)

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
            tools = Tools(root=root)

            try:
                result = json.loads(tools.run("read_file", {"path": "link.txt"}))
            finally:
                outside.unlink(missing_ok=True)

            self.assertFalse(result["ok"])
            self.assertIn("outside workspace", result["error"])

    def test_list_files_rejects_invalid_path_argument(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tools = Tools(root=temp_dir)

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
            tools = Tools(root=root)

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
            tools = Tools(root=root)

            result = json.loads(tools.run("search_text", {"query": "needle"}))

            self.assertTrue(result["ok"])
            self.assertEqual(
                result["results"],
                [{"path": "one.txt", "line": 2, "text": "needle here"}],
            )

    def test_list_files_returns_workspace_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "src").mkdir()
            (root / "README.md").write_text("# Test", encoding="utf-8")
            tools = Tools(root=root)

            result = json.loads(tools.run("list_files", {}))

            self.assertTrue(result["ok"])
            self.assertEqual(result["entries"], ["README.md", "src/"])

    def test_tools_expose_definitions(self) -> None:
        tools = Tools()

        definitions = tools.definitions()
        run_command_definition = definitions[-1]["function"]

        self.assertEqual(
            [definition["function"]["name"] for definition in definitions],
            ["list_files", "read_file", "search_text", "write_file", "edit_file", "run_command"],
        )
        self.assertIn("Allowed commands", run_command_definition["description"])
        self.assertIn("git status", run_command_definition["description"])

    def test_run_command_runs_allowed_command_in_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tools = Tools(root=root)

            result = json.loads(tools.run("run_command", {"command": "pwd"}))

            self.assertTrue(result["ok"])
            self.assertEqual(result["exit_code"], 0)
            self.assertEqual(result["stdout"].strip(), str(root))
            self.assertEqual(result["stderr"], "")
            self.assertFalse(result["timed_out"])

    def test_run_command_reports_nonzero_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tool = RunCommandTool()

            result = json.loads(tool.run(Path(temp_dir), {"command": "ls missing-file"}))

            self.assertFalse(result["ok"])
            self.assertNotEqual(result["exit_code"], 0)
            self.assertFalse(result["timed_out"])

    def test_run_command_rejects_invalid_command_argument(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tool = RunCommandTool()

            result = json.loads(tool.run(Path(temp_dir), {"command": ""}))

            self.assertFalse(result["ok"])
            self.assertIn("command is required", result["error"])

    def test_run_command_rejects_disallowed_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tool = RunCommandTool()

            python_result = json.loads(tool.run(Path(temp_dir), {"command": "python --version"}))
            rm_result = json.loads(tool.run(Path(temp_dir), {"command": "rm file.txt"}))
            touch_result = json.loads(tool.run(Path(temp_dir), {"command": "touch file.txt"}))

            self.assertFalse(python_result["ok"])
            self.assertIn("command is not allowed: python", python_result["error"])
            self.assertFalse(rm_result["ok"])
            self.assertIn("command is not allowed: rm", rm_result["error"])
            self.assertFalse(touch_result["ok"])
            self.assertIn("command is not allowed: touch", touch_result["error"])

    def test_run_command_rejects_shell_operators(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tool = RunCommandTool()

            pipe_result = json.loads(tool.run(Path(temp_dir), {"command": "ls | cat"}))
            redirect_result = json.loads(tool.run(Path(temp_dir), {"command": "ls > out.txt"}))
            chain_result = json.loads(tool.run(Path(temp_dir), {"command": "pwd && ls"}))

            self.assertFalse(pipe_result["ok"])
            self.assertIn("shell syntax is not allowed", pipe_result["error"])
            self.assertFalse(redirect_result["ok"])
            self.assertIn("shell syntax is not allowed", redirect_result["error"])
            self.assertFalse(chain_result["ok"])
            self.assertIn("shell syntax is not allowed", chain_result["error"])

    def test_run_command_rejects_mutating_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "victim.txt"
            target.write_text("keep", encoding="utf-8")
            tool = RunCommandTool()

            result = json.loads(tool.run(root, {"command": "find . -name victim.txt -delete"}))

            self.assertFalse(result["ok"])
            self.assertIn("command flag is not allowed: -delete", result["error"])
            self.assertTrue(target.exists())

    def test_run_command_rejects_path_arguments_outside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            outside = root.parent / "outside-command-target.txt"
            outside.write_text("secret", encoding="utf-8")
            (root / "link.txt").symlink_to(outside)
            tool = RunCommandTool()

            try:
                absolute_result = json.loads(tool.run(root, {"command": f"cat {outside}"}))
                symlink_result = json.loads(tool.run(root, {"command": "cat link.txt"}))
            finally:
                outside.unlink(missing_ok=True)

            self.assertFalse(absolute_result["ok"])
            self.assertIn("outside workspace", absolute_result["error"])
            self.assertFalse(symlink_result["ok"])
            self.assertIn("outside workspace", symlink_result["error"])

    def test_run_command_allows_read_only_git_subcommands(self) -> None:
        tool = RunCommandTool()

        status_result = json.loads(tool.run(Path.cwd(), {"command": "git status --short"}))
        commit_result = json.loads(tool.run(Path.cwd(), {"command": "git commit --allow-empty -m test"}))

        self.assertIn("exit_code", status_result)
        self.assertFalse(status_result["timed_out"])
        self.assertFalse(commit_result["ok"])
        self.assertIn("git subcommand is not allowed: commit", commit_result["error"])

    def test_run_command_times_out(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            fifo_path = Path(temp_dir) / "stream"
            os.mkfifo(fifo_path)
            tool = RunCommandTool(timeout_seconds=0.001)

            result = json.loads(tool.run(Path(temp_dir), {"command": "cat stream"}))

            self.assertFalse(result["ok"])
            self.assertIsNone(result["exit_code"])
            self.assertTrue(result["timed_out"])

    def test_run_command_truncates_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "long.txt").write_text("abcdef", encoding="utf-8")
            tool = RunCommandTool(max_output_chars=3)

            result = json.loads(tool.run(root, {"command": "cat long.txt"}))

            self.assertTrue(result["ok"])
            self.assertEqual(result["stdout"], "abc")
            self.assertTrue(result["truncated"])

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
