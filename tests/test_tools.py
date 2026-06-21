import json
import tempfile
import unittest
from pathlib import Path

from simple_agent.tools.list_files import ListFilesTool
from simple_agent.tools.read_file import ReadFileTool
from simple_agent.tools.search_text import SearchTextTool
from simple_agent.tools.write_file import WriteFileTool
from simple_agent.tools.edit_file import EditFileTool
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

        self.assertEqual(
            [definition["function"]["name"] for definition in definitions],
            ["list_files", "read_file", "search_text", "write_file", "edit_file"],
        )

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
