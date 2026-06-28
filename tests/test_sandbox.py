import os
from pathlib import Path
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
    _find_bubblewrap,
    discover_linux_sandbox,
    parse_read_roots,
)
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
                    "plain_agent.sandbox.bubblewrap.BUBBLEWRAP_PATHS",
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
                    "PATH": f"/unmounted/bin{os.pathsep}/usr/bin",
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

    def test_unavailable_backend_omits_run_command_and_keeps_file_tools(self) -> None:
        discovery = SandboxDiscovery(None, "run_command is disabled: install Bubblewrap")
        with patch("plain_agent.tools.tools.discover_linux_sandbox", return_value=discovery):
            tools = Tools(".")

        names = [definition["function"]["name"] for definition in tools.definitions()]
        self.assertNotIn("run_command", names)
        self.assertIn("read_file", names)
        self.assertEqual(tools.startup_warnings, [discovery.warning])

    def test_discovery_fails_closed_when_bubblewrap_probe_fails(self) -> None:
        with (
            patch("plain_agent.sandbox.bubblewrap.sys.platform", "linux"),
            patch(
                "plain_agent.sandbox.bubblewrap._find_bubblewrap",
                return_value=Path("/usr/bin/bwrap"),
            ),
            patch.object(
                BubblewrapSandbox,
                "probe",
                side_effect=SandboxUnavailableError("namespaces unavailable"),
            ),
        ):
            discovery = discover_linux_sandbox()

        self.assertIsNone(discovery.backend)
        self.assertIn("namespaces unavailable", discovery.warning or "")

    def test_discovery_reports_unsupported_platform(self) -> None:
        with patch("plain_agent.sandbox.bubblewrap.sys.platform", "darwin"):
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
