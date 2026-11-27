"""Running SLURM's commands: one injectable runner, typed errors.

Everything in euler-slurm reaches SLURM through a :class:`Runner`: an object
with ``run(args, timeout) -> (returncode, stdout, stderr)``. The default,
:class:`SubprocessRunner`, executes the command on this machine (no shell).
Tests, and tools that reach a cluster some other way, pass their own.

A command that is not installed is a :class:`SlurmUnavailable` error, never a
crash: euler-slurm is meant to be importable (and its capability report
callable) on machines without SLURM.
"""

from __future__ import annotations

import os
import subprocess
from typing import Mapping, Optional, Protocol, Sequence, Tuple

#: Seconds a single SLURM command may take before it is given up.
DEFAULT_TIMEOUT = 30.0

#: Environment variables that change what SLURM's commands print
#: (``SQUEUE_FORMAT``, ``SQUEUE_SORT``, ``SACCT_FORMAT``, ``SQUEUE_USERS`` …).
#: They are removed so the output has the shape the parsers read.
OUTPUT_VARIABLE_PREFIXES = ("SQUEUE_", "SACCT_", "SINFO_", "SCONTROL_")

#: Set for every command: timestamps as ``YYYY-MM-DDTHH:MM:SS`` whatever the
#: user configured (``SLURM_TIME_FORMAT=relative`` would print "Tomorr 14:30").
CONTROLLED_VARIABLES = {"SLURM_TIME_FORMAT": "standard"}


class SlurmError(Exception):
    """Base class of every error euler-slurm raises."""


class SlurmUnavailable(SlurmError):
    """A SLURM command is not installed, or cannot be executed, here."""

    def __init__(self, command: str, reason: str) -> None:
        self.command = command
        self.reason = reason
        super().__init__(reason)


class SlurmCommandError(SlurmError):
    """A SLURM command ran but failed (non-zero exit status)."""

    def __init__(self, args: Sequence[str], returncode: int, stderr: str) -> None:
        self.args_list = list(args)
        self.returncode = returncode
        self.stderr = stderr
        detail = first_line(stderr) or "no error message"
        super().__init__(
            f"`{' '.join(self.args_list)}` failed ({returncode}): {detail}"
        )


class SlurmTimeout(SlurmCommandError):
    """A SLURM command did not answer in time (the controller may be busy)."""

    def __init__(self, args: Sequence[str], timeout: float) -> None:
        self.timeout = timeout
        super().__init__(args, -1, f"no answer within {timeout:g} s")


class Runner(Protocol):
    """Runs one command. Raises :class:`SlurmUnavailable` if it is missing."""

    def run(self, args: Sequence[str], timeout: float) -> Tuple[int, str, str]:
        """Run ``args`` and return ``(returncode, stdout, stderr)``."""
        ...


def first_line(text: Optional[str]) -> str:
    """The first non-blank line of ``text`` (for error messages)."""
    for line in (text or "").splitlines():
        if line.strip():
            return line.strip()
    return ""


def command_environment(base: Optional[Mapping[str, str]] = None) -> dict:
    """The environment SLURM's commands run with (see the module constants)."""
    env = dict(os.environ if base is None else base)
    for key in list(env):
        if key.startswith(OUTPUT_VARIABLE_PREFIXES):
            del env[key]
    env.update(CONTROLLED_VARIABLES)
    return env


class SubprocessRunner:
    """Runs commands on this machine with :func:`subprocess.run` (no shell)."""

    def __init__(self, env: Optional[Mapping[str, str]] = None) -> None:
        self._env = dict(env) if env is not None else None

    def run(self, args: Sequence[str], timeout: float) -> Tuple[int, str, str]:
        argv = [str(arg) for arg in args]
        if not argv:
            raise ValueError("no command given")
        command = argv[0]
        try:
            completed = subprocess.run(
                argv,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
                env=self._env if self._env is not None else command_environment(),
                check=False,
            )
        except FileNotFoundError:
            raise SlurmUnavailable(
                command,
                f"`{command}` was not found on PATH (SLURM is not installed here, "
                "or its module is not loaded)",
            ) from None
        except PermissionError as exc:
            raise SlurmUnavailable(
                command, f"`{command}` cannot be executed: {exc.strerror or exc}"
            ) from None
        except subprocess.TimeoutExpired:
            raise SlurmTimeout(argv, timeout) from None
        except OSError as exc:
            raise SlurmUnavailable(
                command, f"`{command}` could not be started: {exc.strerror or exc}"
            ) from None
        return completed.returncode, completed.stdout or "", completed.stderr or ""


def default_runner() -> Runner:
    """The runner used when none is given: commands on this machine."""
    return SubprocessRunner()


def run_checked(
    runner: Runner, args: Sequence[str], timeout: float = DEFAULT_TIMEOUT
) -> str:
    """Run ``args``; its stdout, or :class:`SlurmCommandError` when it fails."""
    returncode, stdout, stderr = runner.run(list(args), timeout)
    if returncode != 0:
        raise SlurmCommandError(args, returncode, stderr)
    return stdout


__all__ = [
    "CONTROLLED_VARIABLES",
    "DEFAULT_TIMEOUT",
    "OUTPUT_VARIABLE_PREFIXES",
    "Runner",
    "SlurmCommandError",
    "SlurmError",
    "SlurmTimeout",
    "SlurmUnavailable",
    "SubprocessRunner",
    "command_environment",
    "default_runner",
    "first_line",
    "run_checked",
]
