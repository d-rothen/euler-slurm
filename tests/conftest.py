"""Shared test helpers: a fake runner and the recorded SLURM outputs."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

import pytest

from euler_slurm.runner import SlurmUnavailable

FIXTURES = Path(__file__).parent / "fixtures"
LOGS = FIXTURES / "logs"

Response = Union[
    Tuple[int, str, str], BaseException, Callable[[List[str]], Tuple[int, str, str]]
]


def fixture_text(name: str) -> str:
    return (FIXTURES / name).read_text()


def epoch(text: str) -> float:
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp()


#: "Now" in the fixtures: 2026-11-20T15:51:00Z.
NOW = epoch("2026-11-20T15:51:00")

UNRECOGNIZED_JSON = (
    1,
    "",
    "squeue: unrecognized option '--json'\nTry \"squeue --help\"\n",
)


class FakeRunner:
    """Answers commands from a list of ``(tokens, response)`` rules.

    A rule matches when its first token is the command and every other token
    is one of the arguments. The first matching rule answers. Commands in
    ``missing`` raise :class:`SlurmUnavailable`; unmatched commands fail
    like an unknown option would.
    """

    def __init__(
        self,
        rules: Sequence[Tuple[Sequence[str], Response]] = (),
        *,
        missing: Sequence[str] = (),
    ) -> None:
        self.rules = [(tuple(tokens), response) for tokens, response in rules]
        self.missing = set(missing)
        self.calls: List[List[str]] = []

    def add(self, tokens: Sequence[str], response: Response) -> "FakeRunner":
        self.rules.append((tuple(tokens), response))
        return self

    def run(self, args: Sequence[str], timeout: float) -> Tuple[int, str, str]:
        argv = [str(arg) for arg in args]
        self.calls.append(argv)
        if argv[0] in self.missing:
            raise SlurmUnavailable(argv[0], f"`{argv[0]}` was not found on PATH")
        for tokens, response in self.rules:
            if tokens[0] == argv[0] and all(token in argv[1:] for token in tokens[1:]):
                if isinstance(response, BaseException):
                    raise response
                if callable(response):
                    return response(argv)
                return response
        return 1, "", f"{argv[0]}: unrecognized option (fake: no rule for {argv})"

    def called(self, *tokens: str) -> bool:
        return any(
            call[0] == tokens[0] and all(token in call[1:] for token in tokens[1:])
            for call in self.calls
        )


@pytest.fixture(autouse=True)
def utc(monkeypatch: pytest.MonkeyPatch):
    """Every test runs in UTC, as alice, without SLURM's output variables."""
    monkeypatch.setenv("TZ", "UTC")
    monkeypatch.setenv("USER", "alice")
    monkeypatch.delenv("SLURM_JOB_USER", raising=False)
    if hasattr(time, "tzset"):
        time.tzset()
    yield
    monkeypatch.undo()
    if hasattr(time, "tzset"):
        time.tzset()


def json_cluster(
    extra: Optional[Sequence[Tuple[Sequence[str], Response]]] = None,
) -> FakeRunner:
    """A Slurm 23.11 cluster: every command answers --json."""
    runner = FakeRunner(list(extra or ()))
    runner.add(("squeue", "--json"), (0, fixture_text("squeue_v0040.json"), ""))
    runner.add(("sacct", "--json"), (0, fixture_text("sacct_v0040.json"), ""))
    runner.add(("scontrol", "--json"), (0, fixture_text("scontrol_v0040.json"), ""))
    return runner


def text_cluster(
    extra: Optional[Sequence[Tuple[Sequence[str], Response]]] = None,
) -> FakeRunner:
    """A Slurm 20.11 cluster: no --json anywhere, delimiter-separated text."""
    runner = FakeRunner(list(extra or ()))
    runner.add(("squeue", "--json"), UNRECOGNIZED_JSON)
    runner.add(("sacct", "--json"), (1, "", "sacct: unrecognized option '--json'\n"))
    runner.add(
        ("scontrol", "--json"), (1, "", "scontrol: unrecognized option '--json'\n")
    )
    runner.add(("squeue", "--Format"), (0, fixture_text("squeue_tres.txt"), ""))
    runner.add(("squeue", "--format"), (0, fixture_text("squeue_format.txt"), ""))
    runner.add(("sacct", "--parsable2"), (0, fixture_text("sacct_parsable2.txt"), ""))
    return runner
