"""Linux Bubblewrap discovery and configuration."""

from dataclasses import dataclass
import os
from pathlib import Path
import sys

from plain_agent.sandbox.bubblewrap.command import (
    BubblewrapSandbox,
    SANDBOX_ADDITIONAL_READ_ROOTS_ENV,
    canonical_read_roots,
)
from plain_agent.sandbox.base import (
    SandboxConfigurationError,
    SandboxUnavailableError,
)

BUBBLEWRAP_CANDIDATE_PATHS = (
    Path("/usr/bin/bwrap"),
    Path("/bin/bwrap"),
)


@dataclass(frozen=True)
class SandboxDiscovery:
    backend: BubblewrapSandbox | None
    warning: str | None


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
            warning=(
                "run_command is disabled: install Bubblewrap ('bwrap') to enable "
                "sandboxed commands."
            ),
        )
    try:
        additional_read_roots = parse_read_roots(
            os.environ.get(SANDBOX_ADDITIONAL_READ_ROOTS_ENV)
        )
        backend = BubblewrapSandbox(executable, additional_read_roots)
        backend.verify_usable()
    except (SandboxConfigurationError, SandboxUnavailableError, OSError) as exc:
        return SandboxDiscovery(backend=None, warning=f"run_command is disabled: {exc}")
    return SandboxDiscovery(backend=backend, warning=None)


def _find_bubblewrap() -> Path | None:
    """Find Bubblewrap only in system locations outside the inherited PATH."""
    for path in BUBBLEWRAP_CANDIDATE_PATHS:
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
    paths = tuple(Path(raw_path) for raw_path in value.split(os.pathsep) if raw_path)
    return canonical_read_roots(paths)
