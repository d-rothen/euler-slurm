import os
import stat
import sys

import pytest

from euler_slurm.runner import (
    SlurmCommandError,
    SlurmError,
    SlurmTimeout,
    SlurmUnavailable,
    SubprocessRunner,
    command_environment,
    default_runner,
    first_line,
    run_checked,
)


def test_missing_binary_is_a_typed_error_not_a_crash():
    runner = SubprocessRunner()
    with pytest.raises(SlurmUnavailable) as caught:
        runner.run(["euler-slurm-no-such-squeue", "--version"], 5)
    assert caught.value.command == "euler-slurm-no-such-squeue"
    assert "not found on PATH" in caught.value.reason
    assert isinstance(caught.value, SlurmError)


def test_non_executable_file_is_unavailable(tmp_path):
    script = tmp_path / "squeue"
    script.write_text("#!/bin/sh\necho hi\n")
    script.chmod(stat.S_IRUSR | stat.S_IWUSR)
    if os.access(script, os.X_OK):  # running as root may still execute it
        pytest.skip("cannot make a file non-executable here")
    with pytest.raises(SlurmUnavailable):
        SubprocessRunner().run([str(script)], 5)


def test_runs_a_command_and_returns_its_output():
    code, out, err = SubprocessRunner().run(
        [
            sys.executable,
            "-c",
            "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)",
        ],
        10,
    )
    assert (code, out.strip(), err.strip()) == (3, "out", "err")


def test_timeout_is_typed():
    with pytest.raises(SlurmTimeout) as caught:
        SubprocessRunner().run(
            [sys.executable, "-c", "import time; time.sleep(5)"], 0.2
        )
    assert isinstance(caught.value, SlurmCommandError)
    assert "no answer within" in str(caught.value)


def test_output_variables_are_controlled(monkeypatch):
    monkeypatch.setenv("SQUEUE_FORMAT", "%i")
    monkeypatch.setenv("SACCT_FORMAT", "jobid")
    monkeypatch.setenv("SLURM_TIME_FORMAT", "relative")
    monkeypatch.setenv("SLURM_CONF", "/etc/slurm/slurm.conf")
    env = command_environment()
    assert "SQUEUE_FORMAT" not in env and "SACCT_FORMAT" not in env
    assert env["SLURM_TIME_FORMAT"] == "standard"
    assert env["SLURM_CONF"] == "/etc/slurm/slurm.conf"
    code, out, _ = SubprocessRunner().run(
        [
            sys.executable,
            "-c",
            "import os; print(os.environ.get('SQUEUE_FORMAT'), os.environ['SLURM_TIME_FORMAT'])",
        ],
        10,
    )
    assert out.split() == ["None", "standard"]


def test_explicit_environment_is_used_as_given():
    runner = SubprocessRunner(env={"PATH": os.environ.get("PATH", ""), "ONLY": "me"})
    _, out, _ = runner.run(
        [sys.executable, "-c", "import os; print(os.environ.get('ONLY'))"], 10
    )
    assert out.strip() == "me"


def test_run_checked_raises_with_the_first_stderr_line():
    with pytest.raises(SlurmCommandError) as caught:
        run_checked(
            default_runner(),
            [
                sys.executable,
                "-c",
                "import sys; sys.stderr.write('\\nboom happened\\nmore\\n'); sys.exit(2)",
            ],
        )
    assert caught.value.returncode == 2
    assert "boom happened" in str(caught.value)
    assert (
        run_checked(default_runner(), [sys.executable, "-c", "print(42)"]).strip()
        == "42"
    )


def test_empty_command_is_refused():
    with pytest.raises(ValueError):
        SubprocessRunner().run([], 1)


def test_first_line():
    assert first_line("\n  \nslurm_load_jobs error: Invalid job id specified\nx") == (
        "slurm_load_jobs error: Invalid job id specified"
    )
    assert first_line(None) == ""
