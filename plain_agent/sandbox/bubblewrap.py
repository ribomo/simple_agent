"""Linux Bubblewrap command sandbox."""

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys

from plain_agent.sandbox.base import (
    CommandRequest,
    SandboxConfigurationError,
    SandboxMode,
    SandboxUnavailableError,
)
from plain_agent.tools.permissions.file_permission import (
    SENSITIVE_FILE_NAMES,
    SENSITIVE_FILE_SUFFIXES,
)

SANDBOX_READ_ROOTS_ENV = "PLAIN_AGENT_SANDBOX_READ_ROOTS"
BUBBLEWRAP_PATHS = (
    Path("/usr/bin/bwrap"),
    Path("/bin/bwrap"),
)
SYSTEM_ROOTS = (
    Path("/usr"),
    Path("/usr/local"),
    Path("/bin"),
    Path("/sbin"),
    Path("/lib"),
    Path("/lib64"),
)
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
HIDDEN_WORKSPACE_DIRS = (".agents", ".codex", ".sandbox")
READ_ONLY_WORKSPACE_PATHS = (".git", ".venv")


@dataclass(frozen=True)
class SandboxDiscovery:
    backend: "BubblewrapSandbox | None"
    warning: str | None


class BubblewrapSandbox:
    """Construct a minimal, offline Bubblewrap environment for a command."""

    def __init__(self, executable: Path, extra_read_roots: tuple[Path, ...] = ()) -> None:
        self.executable = executable.resolve()
        self.extra_read_roots = _canonical_read_roots(extra_read_roots)

    def build_command(self, request: CommandRequest) -> list[str]:
        workspace = request.workspace.resolve(strict=True)
        if not workspace.is_dir():
            raise SandboxConfigurationError("workspace must be an existing directory")

        args = [
            str(self.executable),
            "--die-with-parent",
            "--new-session",
            "--unshare-user",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-net",
            "--unshare-uts",
            "--unshare-cgroup-try",
            "--cap-drop",
            "ALL",
            "--clearenv",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--tmpfs",
            "/run",
            "--dir",
            "/tmp/plain-agent-home",
        ]

        args.extend(self._system_mounts())
        for root in self.extra_read_roots:
            args.extend(("--ro-bind", str(root), str(root)))

        mount_flag = "--bind" if request.mode is SandboxMode.WORKSPACE_WRITE else "--ro-bind"
        args.extend((mount_flag, str(workspace), str(workspace)))
        args.extend(self._workspace_protections(workspace, request.mode))
        args.extend(self._environment_args(workspace))
        args.extend(("--chdir", str(workspace), "--"))
        args.extend(request.argv)
        return args

    def probe(self, timeout_seconds: float = 2) -> None:
        true_path = _first_existing(Path("/usr/bin/true"), Path("/bin/true"))
        if true_path is None:
            raise SandboxUnavailableError("Bubblewrap probe could not find the 'true' executable")
        request = CommandRequest(
            argv=(str(true_path),),
            mode=SandboxMode.READ_ONLY,
            workspace=Path.cwd().resolve(),
        )
        try:
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
            raise SandboxUnavailableError(f"Bubblewrap probe failed: {exc}") from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"exit code {completed.returncode}"
            raise SandboxUnavailableError(f"Bubblewrap probe failed: {detail}")

    def _system_mounts(self) -> list[str]:
        args: list[str] = []
        seen: set[Path] = set()
        for root in SYSTEM_ROOTS:
            if root in seen or not root.exists():
                continue
            if root.is_symlink():
                args.extend(("--symlink", os.readlink(root), str(root)))
            else:
                args.extend(("--ro-bind", str(root), str(root)))
            seen.add(root)

        for path in ETC_READ_PATHS:
            if path.exists():
                args.extend(("--ro-bind", str(path), str(path)))
        return args

    def _environment_args(self, workspace: Path) -> list[str]:
        args = [
            "--setenv",
            "HOME",
            "/tmp/plain-agent-home",
            "--setenv",
            "TMPDIR",
            "/tmp",
            "--setenv",
            "PATH",
            self._filtered_path(workspace),
        ]
        for name in PASSTHROUGH_ENV_NAMES:
            value = os.environ.get(name)
            if value is not None:
                args.extend(("--setenv", name, value))
        return args

    def _filtered_path(self, workspace: Path) -> str:
        allowed_roots = (
            tuple(path.resolve() for path in SYSTEM_ROOTS if path.exists())
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
            if not path.is_absolute() or not path.exists():
                continue
            resolved = path.resolve()
            if any(_contains(root, resolved) for root in allowed_roots):
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

    def _workspace_protections(self, workspace: Path, mode: SandboxMode) -> list[str]:
        args: list[str] = []
        for name in HIDDEN_WORKSPACE_DIRS:
            path = workspace / name
            _reject_protected_symlink(path)
            if path.is_dir() and not path.is_symlink():
                args.extend(("--tmpfs", str(path)))
            elif path.exists():
                args.extend(("--ro-bind", "/dev/null", str(path)))

        if mode is SandboxMode.WORKSPACE_WRITE:
            for name in READ_ONLY_WORKSPACE_PATHS:
                path = workspace / name
                _reject_protected_symlink(path)
                if path.exists():
                    args.extend(("--ro-bind", str(path), str(path)))

        for path in _sensitive_workspace_files(workspace):
            args.extend(("--ro-bind", "/dev/null", str(path)))
        return args


def discover_linux_sandbox() -> SandboxDiscovery:
    """Find and verify Bubblewrap without ever falling back to direct execution."""
    if not sys.platform.startswith("linux"):
        return SandboxDiscovery(
            backend=None,
            warning="run_command is disabled: the sandbox currently supports Linux only.",
        )

    executable = _find_bubblewrap()
    if executable is None:
        return SandboxDiscovery(
            backend=None,
            warning="run_command is disabled: install Bubblewrap ('bwrap') to enable sandboxed commands.",
        )
    try:
        read_roots = parse_read_roots(os.environ.get(SANDBOX_READ_ROOTS_ENV))
        backend = BubblewrapSandbox(executable, read_roots)
        backend.probe()
    except (SandboxConfigurationError, SandboxUnavailableError, OSError) as exc:
        return SandboxDiscovery(backend=None, warning=f"run_command is disabled: {exc}")
    return SandboxDiscovery(backend=backend, warning=None)


def _find_bubblewrap() -> Path | None:
    """Find Bubblewrap only in system locations outside the inherited PATH."""
    for path in BUBBLEWRAP_PATHS:
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved
    return None


def parse_read_roots(value: str | None) -> tuple[Path, ...]:
    """Parse explicit, read-only host paths exposed inside the sandbox."""
    if value is None or not value.strip():
        return ()
    roots: list[Path] = []
    for raw_path in value.split(os.pathsep):
        if not raw_path:
            continue
        roots.append(Path(raw_path))
    return _canonical_read_roots(tuple(roots))


def _sensitive_workspace_files(workspace: Path) -> list[Path]:
    sensitive: list[Path] = []
    skipped_dirs = set(HIDDEN_WORKSPACE_DIRS)
    for root, dirs, files in os.walk(
        workspace,
        followlinks=False,
        onerror=_raise_walk_error,
    ):
        dirs[:] = [name for name in dirs if name not in skipped_dirs]
        root_path = Path(root)
        for name in files:
            lower_name = name.lower()
            path = root_path / name
            if lower_name in SENSITIVE_FILE_NAMES or path.suffix.lower() in SENSITIVE_FILE_SUFFIXES:
                _reject_protected_symlink(path)
                sensitive.append(path)
    return sorted(sensitive)


def _contains(root: Path, path: Path) -> bool:
    return path == root or root in path.parents


def _first_existing(*paths: Path) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def _canonical_read_roots(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    roots: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if not path.is_absolute():
            raise SandboxConfigurationError(
                f"{SANDBOX_READ_ROOTS_ENV} entries must be absolute: {path}"
            )
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise SandboxConfigurationError(
                f"{SANDBOX_READ_ROOTS_ENV} entry does not exist: {path}"
            ) from exc
        if resolved not in seen:
            roots.append(resolved)
            seen.add(resolved)
    return tuple(roots)


def _reject_protected_symlink(path: Path) -> None:
    if path.is_symlink():
        raise SandboxConfigurationError(f"protected sandbox path must not be a symlink: {path}")


def _raise_walk_error(error: OSError) -> None:
    raise SandboxConfigurationError(f"could not inspect workspace protections: {error}") from error
