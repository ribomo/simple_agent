import json
import os
from pathlib import Path
import socket
import tempfile
import time
import unittest
from unittest.mock import patch

from plain_agent.sandbox import CommandRequest, SandboxMode
from plain_agent.sandbox.bubblewrap import BubblewrapSandbox, discover_linux_sandbox
from plain_agent.tools.permissions.controller import PermissionController
from plain_agent.tools.permissions.request import ApprovalDecision, CommandPermissionRequest
from plain_agent.tools.run_command import RunCommandTool


def approve_request(request: CommandPermissionRequest) -> ApprovalDecision:
    return ApprovalDecision.ALLOW_ONCE


class LinuxSandboxIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        discovery = discover_linux_sandbox()
        if discovery.backend is None:
            raise unittest.SkipTest(discovery.warning or "Bubblewrap is unavailable")
        cls.backend = discovery.backend

    def run_command(
        self,
        workspace: Path,
        argv: list[str],
        mode: SandboxMode = SandboxMode.READ_ONLY,
        timeout_seconds: float = 3,
        max_output_chars: int = 12_000,
    ) -> dict[str, object]:
        tool = RunCommandTool(
            self.backend,
            PermissionController(approve_request),
            timeout_seconds=timeout_seconds,
            max_output_chars=max_output_chars,
        )
        return json.loads(
            tool.run(
                workspace,
                {
                    "argv": argv,
                    "mode": mode.value,
                    "justification": "Exercise the sandbox integration",
                },
            )
        )

    def test_workspace_reads_succeed_and_read_only_writes_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            target = workspace / "notes.txt"
            target.write_text("hello sandbox", encoding="utf-8")

            read = self.run_command(workspace, ["cat", "notes.txt"])
            write = self.run_command(workspace, ["touch", "blocked.txt"])

            self.assertTrue(read["ok"])
            self.assertEqual(read["stdout"], "hello sandbox")
            self.assertFalse(write["ok"])
            self.assertFalse((workspace / "blocked.txt").exists())

    def test_workspace_write_persists_permitted_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)

            result = self.run_command(
                workspace,
                ["bash", "-lc", "printf persisted > result.txt"],
                SandboxMode.WORKSPACE_WRITE,
            )

            self.assertTrue(result["ok"])
            self.assertEqual((workspace / "result.txt").read_text(), "persisted")

    def test_sibling_home_secret_and_symlink_target_are_inaccessible(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.home()) as parent_dir:
            parent = Path(parent_dir)
            workspace = parent / "workspace"
            sibling = parent / "sibling"
            workspace.mkdir()
            sibling.mkdir()
            secret = sibling / "secret.txt"
            secret.write_text("not for the command", encoding="utf-8")
            (workspace / "escape").symlink_to(secret)

            absolute = self.run_command(workspace, ["cat", str(secret)])
            symlink = self.run_command(workspace, ["cat", "escape"])

            self.assertFalse(absolute["ok"])
            self.assertFalse(symlink["ok"])
            self.assertNotIn("not for the command", str(absolute["stdout"]))
            self.assertNotIn("not for the command", str(symlink["stdout"]))

    def test_explicit_extra_read_root_is_available_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as parent_dir:
            parent = Path(parent_dir)
            workspace = parent / "workspace"
            toolchain = parent / "toolchain"
            workspace.mkdir()
            toolchain.mkdir()
            target = toolchain / "version.txt"
            target.write_text("toolchain data", encoding="utf-8")
            backend = BubblewrapSandbox(self.backend.executable, (toolchain.resolve(),))
            tool = RunCommandTool(backend, PermissionController(approve_request))

            read = json.loads(
                tool.run(
                    workspace,
                    {
                        "argv": ["cat", str(target)],
                        "justification": "Read an explicitly permitted root",
                    },
                )
            )
            write = json.loads(
                tool.run(
                    workspace,
                    {
                        "argv": ["bash", "-lc", f"printf changed > {target}"],
                        "mode": "workspace-write",
                        "justification": "Verify that extra roots remain read-only",
                    },
                )
            )

            self.assertTrue(read["ok"])
            self.assertEqual(read["stdout"], "toolchain data")
            self.assertFalse(write["ok"])
            self.assertEqual(target.read_text(), "toolchain data")

    def test_parent_api_keys_are_not_inherited(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            script = "import os; print(os.environ.get('OPENAI_API_KEY', 'missing'))"

            with patch.dict(os.environ, {"OPENAI_API_KEY": "top-secret"}):
                result = self.run_command(workspace, ["python3", "-c", script])

            self.assertTrue(result["ok"])
            self.assertEqual(result["stdout"].strip(), "missing")

    def test_writes_to_read_only_system_roots_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)

            result = self.run_command(
                workspace,
                ["touch", "/usr/local/plain-agent-sandbox-write-test"],
                SandboxMode.WORKSPACE_WRITE,
            )

            self.assertFalse(result["ok"])

    def test_workspace_metadata_and_sensitive_files_are_protected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            for name in (".git", ".venv", ".agents", ".codex", ".sandbox"):
                (workspace / name).mkdir()
                (workspace / name / "value").write_text("hidden metadata", encoding="utf-8")
            (workspace / ".env").write_text("TOKEN=secret", encoding="utf-8")
            (workspace / "private.key").write_text("private key", encoding="utf-8")

            git_write = self.run_command(
                workspace,
                ["touch", ".git/changed"],
                SandboxMode.WORKSPACE_WRITE,
            )
            venv_write = self.run_command(
                workspace,
                ["touch", ".venv/changed"],
                SandboxMode.WORKSPACE_WRITE,
            )
            hidden = self.run_command(workspace, ["cat", ".agents/value"])
            env_file = self.run_command(workspace, ["cat", ".env"])
            key_file = self.run_command(workspace, ["cat", "private.key"])

            self.assertFalse(git_write["ok"])
            self.assertFalse(venv_write["ok"])
            self.assertFalse(hidden["ok"])
            self.assertFalse(env_file["ok"])
            self.assertEqual(env_file["stdout"], "")
            self.assertFalse(key_file["ok"])
            self.assertEqual(key_file["stdout"], "")

    def test_tcp_udp_and_host_loopback_are_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            tcp_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            tcp_listener.bind(("127.0.0.1", 0))
            tcp_listener.listen()
            udp_listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            udp_listener.bind(("127.0.0.1", 0))
            udp_listener.settimeout(0.2)
            script = (
                "import socket,sys; "
                "s=socket.socket(socket.AF_INET, int(sys.argv[1])); "
                "\ntry: s.connect(('127.0.0.1', int(sys.argv[2]))); s.send(b'x')"
                "\nexcept OSError: raise SystemExit(0)"
                "\nraise SystemExit(9)"
            )

            try:
                tcp = self.run_command(
                    workspace,
                    ["python3", "-c", script, str(socket.SOCK_STREAM), str(tcp_listener.getsockname()[1])],
                )
                udp = self.run_command(
                    workspace,
                    ["python3", "-c", script, str(socket.SOCK_DGRAM), str(udp_listener.getsockname()[1])],
                )
                with self.assertRaises(TimeoutError):
                    udp_listener.recvfrom(1)
            finally:
                tcp_listener.close()
                udp_listener.close()

            self.assertTrue(tcp["ok"])
            self.assertFalse(udp["ok"])

    def test_pathname_unix_socket_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            socket_path = workspace / "service.sock"
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(str(socket_path))
            listener.listen()
            listener.settimeout(0.2)
            script = (
                "import socket,sys; "
                "s=socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); "
                "\ntry: s.connect(sys.argv[1])"
                "\nexcept OSError: raise SystemExit(0)"
                "\ns.send(b'x'); raise SystemExit(9)"
            )

            try:
                result = self.run_command(
                    workspace,
                    ["python3", "-c", script, str(socket_path)],
                )
                with self.assertRaises(TimeoutError):
                    listener.accept()
            finally:
                listener.close()

            self.assertTrue(result["ok"])

    def test_explicit_shell_argv_works(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.run_command(
                Path(temp_dir), ["bash", "-lc", "printf explicit-shell"]
            )

            self.assertTrue(result["ok"])
            self.assertEqual(result["stdout"], "explicit-shell")

    def test_timeout_terminates_descendants(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            result = self.run_command(
                workspace,
                ["bash", "-lc", "(sleep 0.3; touch descendant-survived) & wait"],
                SandboxMode.WORKSPACE_WRITE,
                timeout_seconds=0.05,
            )
            time.sleep(0.4)

            self.assertFalse(result["ok"])
            self.assertTrue(result["timed_out"])
            self.assertFalse((workspace / "descendant-survived").exists())

    def test_output_is_truncated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.run_command(
                Path(temp_dir),
                ["bash", "-lc", "printf abcdef"],
                max_output_chars=3,
            )

            self.assertTrue(result["ok"])
            self.assertEqual(result["stdout"], "abc")
            self.assertTrue(result["truncated"])


if __name__ == "__main__":
    unittest.main()
