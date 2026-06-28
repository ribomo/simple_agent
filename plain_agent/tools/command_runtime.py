"""Runtime for OS-sandboxed workspace commands."""

from dataclasses import asdict, dataclass
import subprocess
import threading
from typing import TextIO

from plain_agent.sandbox import CommandRequest, SandboxBackend, SandboxConfigurationError


class CommandRuntimeError(ValueError):
    """Raised when a sandboxed command cannot be run."""


@dataclass
class CommandResult:
    ok: bool
    command: str
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    truncated: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class CommandRuntime:
    """Run commands exclusively through an OS sandbox backend."""

    def __init__(
        self,
        sandbox: SandboxBackend,
        timeout_seconds: float = 30,
        max_output_chars: int = 12_000,
    ) -> None:
        self.sandbox = sandbox
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars

    def run(self, request: CommandRequest) -> CommandResult:
        try:
            sandboxed_argv = self.sandbox.build_command(request)
        except (OSError, SandboxConfigurationError) as exc:
            raise CommandRuntimeError(f"could not configure sandbox: {exc}") from exc

        try:
            process = subprocess.Popen(
                sandboxed_argv,
                cwd=request.workspace,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                shell=False,
                env={},
            )
        except OSError as exc:
            raise CommandRuntimeError(f"could not run sandboxed command: {exc}") from exc

        if process.stdout is None or process.stderr is None:
            process.kill()
            process.wait()
            raise CommandRuntimeError("could not capture sandboxed command output")

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        read_errors: list[Exception] = []
        readers = (
            threading.Thread(
                target=self._drain_output,
                args=(process.stdout, stdout_chunks, read_errors),
                daemon=True,
            ),
            threading.Thread(
                target=self._drain_output,
                args=(process.stderr, stderr_chunks, read_errors),
                daemon=True,
            ),
        )
        for reader in readers:
            reader.start()

        timed_out = False
        try:
            process.wait(timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            process.wait()

        for reader in readers:
            reader.join()
        process.stdout.close()
        process.stderr.close()

        if read_errors:
            detail = str(read_errors[0]).strip()
            raise CommandRuntimeError(
                f"could not read sandboxed command output: {detail or type(read_errors[0]).__name__}"
            )

        stdout, stderr, truncated = self._truncate_outputs(
            "".join(stdout_chunks),
            "".join(stderr_chunks),
        )
        return CommandResult(
            ok=not timed_out and process.returncode == 0,
            command=request.display,
            exit_code=None if timed_out else process.returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            truncated=truncated,
        )

    def _drain_output(
        self,
        stream: TextIO,
        chunks: list[str],
        errors: list[Exception],
    ) -> None:
        retained = 0
        retain_limit = self.max_output_chars + 1
        try:
            while chunk := stream.read(4096):
                remaining = retain_limit - retained
                if remaining > 0:
                    kept = chunk[:remaining]
                    chunks.append(kept)
                    retained += len(kept)
        except Exception as exc:
            errors.append(exc)

    def _truncate_outputs(self, stdout: str, stderr: str) -> tuple[str, str, bool]:
        if len(stdout) + len(stderr) <= self.max_output_chars:
            return stdout, stderr, False

        stdout = stdout[:self.max_output_chars]
        stderr = stderr[:self.max_output_chars - len(stdout)]
        return stdout, stderr, True
