import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from plain_agent.sandbox import (
    BubblewrapSandbox,
    CommandRequest,
    SandboxConfigurationError,
    SandboxMode,
    SandboxUnavailableError,
)
from plain_agent.sandbox.bubblewrap import (
    SandboxDiscovery,
    discover_linux_sandbox,
    parse_read_roots,
)
from plain_agent.sandbox.bubblewrap.discovery import _find_bubblewrap
from plain_agent.sandbox.bubblewrap.workspace import masked_workspace_paths
from plain_agent.tools.tools import Tools


class SandboxTypesTest(unittest.TestCase):
    def test_command_request_validates_and_quotes_argv(self) -> None:
        request = CommandRequest.from_arguments(
            Path("."),
            {"argv": ["printf", "two words", "it's"], "mode": "workspace-write"},
        )

        self.assertEqual(request.argv, ("printf", "two words", "it's"))
        self.assertEqual(request.mode, SandboxMode.WORKSPACE_WRITE)
        self.assertEqual(request.display, "printf 'two words' 'it'\"'\"'s'")
        with self.assertRaises(AttributeError):
            request.argv = ("changed",)

    def test_command_request_defaults_to_read_only(self) -> None:
        request = CommandRequest.from_arguments(Path("."), {"argv": ["true"]})

        self.assertEqual(request.mode, SandboxMode.READ_ONLY)

    def test_command_request_display_escapes_terminal_controls(self) -> None:
        request = CommandRequest(
            (
                "printf",
                "line\nnext",
                "\x1b[2K",
                "left\rright",
                "\u202eabc",
                "zero\u200bwidth",
                r"\n",
                "café",
            ),
            SandboxMode.READ_ONLY,
            Path.cwd(),
        )

        self.assertTrue(request.display.isprintable())
        self.assertIn(r"line\nnext", request.display)
        self.assertIn(r"\x1b[2K", request.display)
        self.assertIn(r"left\rright", request.display)
        self.assertIn(r"\u202eabc", request.display)
        self.assertIn(r"zero\u200bwidth", request.display)
        self.assertIn(r"\\n", request.display)
        self.assertIn("café", request.display)
        self.assertNotEqual(
            CommandRequest(("\n",), SandboxMode.READ_ONLY, Path.cwd()).display,
            CommandRequest((r"\n",), SandboxMode.READ_ONLY, Path.cwd()).display,
        )

    def test_command_request_rejects_invalid_values(self) -> None:
        invalid_arguments = (
            {},
            {"argv": "true"},
            {"argv": []},
            {"argv": [""]},
            {"argv": ["true", 4]},
            {"argv": ["bad\x00value"]},
            {"argv": ["true"], "mode": "dangerous"},
        )

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(SandboxConfigurationError):
                    CommandRequest.from_arguments(Path("."), arguments)


class BubblewrapSandboxTest(unittest.TestCase):
    def test_bubblewrap_discovery_ignores_inherited_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trusted = root / "trusted" / "bwrap"
            untrusted = root / "untrusted" / "bwrap"
            trusted.parent.mkdir()
            untrusted.parent.mkdir()
            trusted.write_text("trusted", encoding="utf-8")
            untrusted.write_text("untrusted", encoding="utf-8")
            trusted.chmod(0o755)
            untrusted.chmod(0o755)

            with (
                patch(
                    "plain_agent.sandbox.bubblewrap.discovery.BUBBLEWRAP_CANDIDATE_PATHS",
                    (trusted,),
                ),
                patch.dict(os.environ, {"PATH": str(untrusted.parent)}),
            ):
                executable = _find_bubblewrap()

        self.assertEqual(executable, trusted.resolve())

    def test_parse_read_roots_requires_absolute_existing_paths_and_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            alias = root / "alias"
            target = root / "target"
            target.mkdir()
            alias.symlink_to(target, target_is_directory=True)

            parsed = parse_read_roots(os.pathsep.join((str(target), str(alias), str(target))))

            self.assertEqual(parsed, (target,))
            with self.assertRaises(SandboxConfigurationError):
                parse_read_roots("relative/path")
            with self.assertRaises(SandboxConfigurationError):
                parse_read_roots(str(root / "missing"))

    def test_build_command_constructs_namespaces_mounts_and_workspace_protections(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir).resolve()
            for name in (".git", ".venv", ".agents", ".codex", ".sandbox"):
                (workspace / name).mkdir()
            (workspace / ".venv" / "bin").mkdir()
            (workspace / ".env").write_text("SECRET=yes", encoding="utf-8")
            (workspace / "identity.pem").write_text("private", encoding="utf-8")
            extra_root = workspace.parent.resolve()
            backend = BubblewrapSandbox(Path("/usr/bin/bwrap"), (extra_root,))
            request = CommandRequest(
                argv=("bash", "-lc", "printf ok"),
                mode=SandboxMode.WORKSPACE_WRITE,
                workspace=workspace,
            )

            with patch.dict(
                os.environ,
                {
                    "PATH": f"/etc{os.pathsep}/unmounted/bin{os.pathsep}/usr/bin",
                    "TERM": "xterm-256color",
                    "LC_CTYPE": "C.UTF-8",
                    "LC_API_KEY": "must-not-leak-either",
                    "OPENAI_API_KEY": "must-not-leak",
                },
                clear=True,
            ):
                argv = backend.build_command(request)

            for flag in (
                "--die-with-parent",
                "--new-session",
                "--unshare-user",
                "--unshare-pid",
                "--unshare-ipc",
                "--unshare-net",
                "--unshare-uts",
                "--disable-userns",
                "--cap-drop",
                "--clearenv",
            ):
                self.assertIn(flag, argv)
            self.assertContainsSequence(argv, ["--bind", str(workspace), str(workspace)])
            self.assertContainsSequence(
                argv, ["--ro-bind", str(workspace / ".git"), str(workspace / ".git")]
            )
            self.assertContainsSequence(
                argv, ["--ro-bind", str(workspace / ".venv"), str(workspace / ".venv")]
            )
            self.assertContainsSequence(argv, ["--tmpfs", str(workspace / ".agents")])
            self.assertContainsSequence(argv, ["--tmpfs", str(workspace / ".codex")])
            self.assertContainsSequence(argv, ["--tmpfs", str(workspace / ".sandbox")])
            self.assertContainsSequence(
                argv, ["--ro-bind", "/dev/null", str(workspace / ".env")]
            )
            self.assertContainsSequence(
                argv, ["--ro-bind", "/dev/null", str(workspace / "identity.pem")]
            )
            self.assertContainsSequence(argv, ["--ro-bind", str(extra_root), str(extra_root)])
            self.assertEqual(argv[-4:], ["--", "bash", "-lc", "printf ok"])

            env = self._setenv_values(argv)
            self.assertEqual(env["HOME"], "/tmp/plain-agent-home")
            self.assertEqual(env["TMPDIR"], "/tmp")
            self.assertEqual(env["TERM"], "xterm-256color")
            self.assertEqual(env["LC_CTYPE"], "C.UTF-8")
            self.assertNotIn("LC_API_KEY", env)
            self.assertNotIn("OPENAI_API_KEY", env)
            self.assertNotIn("/etc", env["PATH"].split(os.pathsep))
            self.assertNotIn("/unmounted/bin", env["PATH"])
            self.assertIn(str(workspace / ".venv" / "bin"), env["PATH"])

    def test_read_only_mode_uses_read_only_workspace_mount(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir).resolve()
            backend = BubblewrapSandbox(Path("/usr/bin/bwrap"))

            argv = backend.build_command(
                CommandRequest(("true",), SandboxMode.READ_ONLY, workspace)
            )

            self.assertContainsSequence(argv, ["--ro-bind", str(workspace), str(workspace)])

    def test_protected_symlinks_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir).resolve()
            target = workspace / "ordinary.txt"
            target.write_text("secret", encoding="utf-8")
            (workspace / ".env").symlink_to(target)
            backend = BubblewrapSandbox(Path("/usr/bin/bwrap"))

            with self.assertRaises(SandboxConfigurationError):
                backend.build_command(
                    CommandRequest(("true",), SandboxMode.READ_ONLY, workspace)
                )

    @unittest.skipUnless(hasattr(socket, "AF_UNIX"), "Unix sockets are unavailable")
    def test_pathname_unix_sockets_are_masked(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir).resolve()
            socket_path = workspace / "service.sock"
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                listener.bind(str(socket_path))
                backend = BubblewrapSandbox(Path("/usr/bin/bwrap"))

                argv = backend.build_command(
                    CommandRequest(("true",), SandboxMode.READ_ONLY, workspace)
                )
            finally:
                listener.close()

        self.assertContainsSequence(
            argv,
            ["--ro-bind", "/dev/null", str(socket_path)],
        )

    def test_workspace_path_inspection_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir).resolve()
            (workspace / "ordinary.txt").write_text("content", encoding="utf-8")

            with patch.object(Path, "lstat", side_effect=OSError("inspection denied")):
                with self.assertRaisesRegex(
                    SandboxConfigurationError,
                    "could not inspect workspace protection path",
                ):
                    masked_workspace_paths(workspace)

    def test_unavailable_backend_omits_run_command_and_keeps_file_tools(self) -> None:
        discovery = SandboxDiscovery(None, "run_command is disabled: install Bubblewrap")
        with patch("plain_agent.tools.tools.discover_linux_sandbox", return_value=discovery):
            tools = Tools(".")

        names = [definition["function"]["name"] for definition in tools.definitions()]
        self.assertNotIn("run_command", names)
        self.assertIn("read_file", names)
        self.assertEqual(tools.startup_warnings, [discovery.warning])

    def test_discovery_fails_closed_when_bubblewrap_verification_fails(self) -> None:
        with (
            patch("plain_agent.sandbox.bubblewrap.discovery.sys.platform", "linux"),
            patch(
                "plain_agent.sandbox.bubblewrap.discovery._find_bubblewrap",
                return_value=Path("/usr/bin/bwrap"),
            ),
            patch.object(
                BubblewrapSandbox,
                "verify_usable",
                side_effect=SandboxUnavailableError("namespaces unavailable"),
            ),
        ):
            discovery = discover_linux_sandbox()

        self.assertIsNone(discovery.backend)
        self.assertIn("namespaces unavailable", discovery.warning or "")

    def test_discovery_reports_unsupported_platform(self) -> None:
        with patch("plain_agent.sandbox.bubblewrap.discovery.sys.platform", "darwin"):
            discovery = discover_linux_sandbox()

        self.assertIsNone(discovery.backend)
        self.assertIn("Linux only", discovery.warning or "")

    def assertContainsSequence(self, values: list[str], sequence: list[str]) -> None:
        width = len(sequence)
        self.assertTrue(
            any(values[index : index + width] == sequence for index in range(len(values) - width + 1)),
            f"{sequence!r} not found in argv",
        )

    def _setenv_values(self, argv: list[str]) -> dict[str, str]:
        output: dict[str, str] = {}
        for index, value in enumerate(argv):
            if value == "--setenv":
                output[argv[index + 1]] = argv[index + 2]
        return output


if __name__ == "__main__":
    unittest.main()
