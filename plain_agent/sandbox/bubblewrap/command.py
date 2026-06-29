"""Bubblewrap command construction and availability verification."""

import os
from pathlib import Path
import subprocess

from plain_agent.sandbox.base import (
    CommandRequest,
    SandboxConfigurationError,
    SandboxMode,
    SandboxUnavailableError,
)
from plain_agent.sandbox.bubblewrap.workspace import build_workspace_protection_arguments

SANDBOX_ADDITIONAL_READ_ROOTS_ENV = "PLAIN_AGENT_SANDBOX_ADDITIONAL_READ_ROOTS"

# Runtime and linker dependencies needed to execute normal host commands.
SYSTEM_ROOTS = (
    Path("/usr"),
    Path("/usr/local"),
    Path("/bin"),
    Path("/sbin"),
    Path("/lib"),
    Path("/lib64"),
)

# Host configuration required for identity lookup, DNS, TLS, and dynamic linking.
ETC_READ_PATHS = (
    Path("/etc/alternatives"),
    Path("/etc/ca-certificates"),
    Path("/etc/gitconfig"),
    Path("/etc/group"),
    Path("/etc/hosts"),
    Path("/etc/ld.so.cache"),
    Path("/etc/ld.so.conf"),
    Path("/etc/ld.so.conf.d"),
    Path("/etc/localtime"),
    Path("/etc/machine-id"),
    Path("/etc/nsswitch.conf"),
    Path("/etc/passwd"),
    Path("/etc/pki"),
    Path("/etc/protocols"),
    Path("/etc/resolv.conf"),
    Path("/etc/services"),
    Path("/etc/ssl"),
)

# Bubblewrap clears the environment; only terminal and locale presentation survive.
PASSTHROUGH_ENV_NAMES = (
    "CLICOLOR",
    "CLICOLOR_FORCE",
    "COLORTERM",
    "FORCE_COLOR",
    "LANG",
    "LANGUAGE",
    "LC_ADDRESS",
    "LC_ALL",
    "LC_COLLATE",
    "LC_CTYPE",
    "LC_IDENTIFICATION",
    "LC_MEASUREMENT",
    "LC_MESSAGES",
    "LC_MONETARY",
    "LC_NAME",
    "LC_NUMERIC",
    "LC_PAPER",
    "LC_TELEPHONE",
    "LC_TIME",
    "NO_COLOR",
    "PY_COLORS",
    "TERM",
)


class BubblewrapSandbox:
    """Construct a minimal, offline Bubblewrap environment for a command."""

    def __init__(self, executable: Path, extra_read_roots: tuple[Path, ...] = ()) -> None:
        self.executable = executable.resolve()
        self.extra_read_roots = canonical_read_roots(extra_read_roots)
        # System layouts are stable during a run, so discover optional paths once.
        self._available_system_roots = tuple(path for path in SYSTEM_ROOTS if path.exists())
        self._available_etc_read_paths = tuple(
            path for path in ETC_READ_PATHS if path.exists()
        )

    def build_command(self, request: CommandRequest) -> list[str]:
        workspace = request.workspace.resolve(strict=True)
        if not workspace.is_dir():
            raise SandboxConfigurationError("workspace must be an existing directory")

        args = self._base_arguments()
        args.extend(self._system_mounts())
        # User-configured host paths are additions to, never extensions of, write access.
        for root in self.extra_read_roots:
            args.extend(("--ro-bind", str(root), str(root)))

        mount_flag = "--bind" if request.mode is SandboxMode.WORKSPACE_WRITE else "--ro-bind"
        args.extend((mount_flag, str(workspace), str(workspace)))
        args.extend(build_workspace_protection_arguments(workspace, request.mode))
        args.extend(self._environment_args(workspace))
        args.extend(("--chdir", str(workspace), "--"))
        args.extend(request.argv)
        return args

    def verify_usable(self, timeout_seconds: float = 2) -> None:
        """Raise unless the host can launch the complete Bubblewrap policy."""
        # Finding the Bubblewrap executable proves only that the file exists. Running a
        # trusted no-op exercises the namespaces, mounts, capabilities, and environment
        # restrictions that every real command will use.
        true_path = _first_existing(Path("/usr/bin/true"), Path("/bin/true"))
        if true_path is None:
            raise SandboxUnavailableError(
                "Bubblewrap verification could not find the 'true' executable"
            )
        request = CommandRequest(
            argv=(str(true_path),),
            mode=SandboxMode.READ_ONLY,
            workspace=Path.cwd().resolve(),
        )
        try:
            # build_command encodes the environment allowlist with --setenv, so the
            # Bubblewrap launcher itself does not need any ambient host variables.
            completed = subprocess.run(
                self.build_command(request),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                shell=False,
                check=False,
                env={},
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SandboxUnavailableError(f"Bubblewrap verification failed: {exc}") from exc
        if completed.returncode != 0:
            # Unsupported kernel features and invalid mounts appear as startup failures.
            detail = completed.stderr.strip() or f"exit code {completed.returncode}"
            raise SandboxUnavailableError(f"Bubblewrap verification failed: {detail}")

    def _base_arguments(self) -> list[str]:
        return [
            str(self.executable),
            # At startup, place the command in a separate POSIX session. While it runs,
            # kill it with SIGKILL if Bubblewrap or Plain Agent exits, preventing an
            # orphaned sandbox process from continuing in the background.
            "--die-with-parent",
            "--new-session",
            # Isolate identity and prevent commands from creating nested user namespaces.
            "--unshare-user",
            "--disable-userns",
            # Isolate process IDs, IPC objects, networking, hostname, and the cgroup view.
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-net",
            "--unshare-uts",
            "--unshare-cgroup-try",
            # Start without Linux capabilities or inherited environment variables.
            "--cap-drop",
            "ALL",
            "--clearenv",
            # Supply only minimal virtual system filesystems.
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            # Runtime files and the synthetic home directory are private and ephemeral.
            "--tmpfs",
            "/tmp",
            "--tmpfs",
            "/run",
            "--dir",
            "/tmp/plain-agent-home",
        ]

    def _system_mounts(self) -> list[str]:
        args: list[str] = []
        for root in self._available_system_roots:
            if root.is_symlink():
                # Preserve merged-/usr layouts such as /bin -> usr/bin.
                args.extend(("--symlink", os.readlink(root), str(root)))
            else:
                args.extend(("--ro-bind", str(root), str(root)))

        for path in self._available_etc_read_paths:
            args.extend(("--ro-bind", str(path), str(path)))
        return args

    def _environment_args(self, workspace: Path) -> list[str]:
        """Build the complete environment exposed to the sandboxed command."""
        args = [
            # Keep programs away from the host home and place temporary files on
            # the private tmpfs created by _base_arguments.
            "--setenv",
            "HOME",
            "/tmp/plain-agent-home",
            "--setenv",
            "TMPDIR",
            "/tmp",
            # Include executable locations only when their targets are mounted.
            "--setenv",
            "PATH",
            self._build_executable_search_path(workspace),
        ]
        # Preserve terminal and locale behavior without inheriting credentials or
        # application-specific configuration from the host.
        for name in PASSTHROUGH_ENV_NAMES:
            value = os.environ.get(name)
            if value is not None:
                args.extend(("--setenv", name, value))
        return args

    def _build_executable_search_path(self, workspace: Path) -> str:
        mounted_roots = (
            tuple(path.resolve() for path in self._available_system_roots)
            + self.extra_read_roots
        )
        candidates: list[Path] = []
        venv_bin = workspace / ".venv" / "bin"
        if venv_bin.is_dir():
            candidates.append(venv_bin)
        for raw_path in os.environ.get("PATH", "").split(os.pathsep):
            if not raw_path:
                continue
            path = Path(raw_path)
            if _is_path_available_in_sandbox(path, mounted_roots):
                candidates.append(path)
        candidates.extend((Path("/usr/local/bin"), Path("/usr/bin"), Path("/bin")))

        output: list[str] = []
        seen: set[str] = set()
        for path in candidates:
            text = str(path)
            if path.exists() and text not in seen:
                output.append(text)
                seen.add(text)
        return os.pathsep.join(output)


def canonical_read_roots(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    """Validate, resolve, and deduplicate extra read-only roots."""
    roots: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if not path.is_absolute():
            raise SandboxConfigurationError(
                f"{SANDBOX_ADDITIONAL_READ_ROOTS_ENV} entries must be absolute: {path}"
            )
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise SandboxConfigurationError(
                f"{SANDBOX_ADDITIONAL_READ_ROOTS_ENV} entry does not exist: {path}"
            ) from exc
        if resolved not in seen:
            roots.append(resolved)
            seen.add(resolved)
    return tuple(roots)


def _contains(root: Path, path: Path) -> bool:
    return path == root or root in path.parents


def _is_path_available_in_sandbox(path: Path, mounted_roots: tuple[Path, ...]) -> bool:
    """Return whether an existing absolute path is exposed by a sandbox mount."""
    if not path.is_absolute() or not path.exists():
        return False
    resolved = path.resolve()
    return any(_contains(root, resolved) for root in mounted_roots)


def _first_existing(*paths: Path) -> Path | None:
    return next((path for path in paths if path.exists()), None)
