import json

import pytest

from euler_slurm.diagnose import (
    RULES,
    Diagnosis,
    clean_line,
    diagnose,
    diagnose_job,
    diagnose_logs,
    read_logs,
    read_tail,
    scan_lines,
)
from euler_slurm.jobs import Job, JobNotFound, SlurmClient
from tests.conftest import LOGS, NOW, FakeRunner, fixture_text, text_cluster


def _job(state="FAILED", log=None, **kwargs):
    path = str(LOGS / log) if log else None
    defaults = dict(
        job_id="90999001", name="train", state=state, stdout_path=path, stderr_path=path
    )
    defaults.update(kwargs)
    return Job(**defaults)


def _codes(diagnosis: Diagnosis):
    return [finding.code for finding in diagnosis.findings]


def _finding(diagnosis, code):
    return next(f for f in diagnosis.findings if f.code == code)


def test_cuda_oom_is_found_and_quoted():
    diagnosis = diagnose_job(_job(log="cuda_oom.log", exit_code=1))
    assert _codes(diagnosis)[:2] == ["state.failed", "log.cuda_oom"]
    finding = _finding(diagnosis, "log.cuda_oom")
    assert finding.severity == "error" and "GPU memory" in finding.message
    assert "batch size" in finding.hint
    assert finding.evidence["line"].startswith(
        "torch.OutOfMemoryError: CUDA out of memory"
    )
    assert finding.evidence["path"].endswith("cuda_oom.log")
    assert "log.python_exception" not in _codes(diagnosis)  # the specific cause wins
    assert "log.host_oom" not in _codes(diagnosis)  # GPU memory is not RAM


def test_tqdm_carriage_returns_do_not_hide_the_error():
    lines = read_tail(str(LOGS / "cuda_oom_tqdm.log"), 200)
    assert lines[0].startswith("Epoch 1:  12%")  # only the last \r segment
    assert len(lines) == 9
    assert "log.cuda_oom" in _codes(diagnose_job(_job(log="cuda_oom_tqdm.log")))


def test_nccl_timeout_with_rank_prefixes():
    diagnosis = diagnose_job(_job(log="nccl_timeout.log", exit_code=1))
    finding = _finding(diagnosis, "log.distributed_timeout")
    assert "first error of any rank" in finding.hint
    assert "Watchdog caught collective operation timeout" in finding.evidence["line"]


def test_import_error_points_to_euler_suite_doctor():
    diagnosis = diagnose_job(_job(log="import_error.log", exit_code=1))
    finding = _finding(diagnosis, "log.import_error")
    assert "`euler_loading`" in finding.message
    assert "euler-suite doctor" in finding.hint and "job's environment" in finding.hint


def test_disk_quota():
    finding = _finding(
        diagnose_job(_job(log="disk_quota.log", exit_code=1)), "log.disk_quota"
    )
    assert "quota" in finding.message and "du -sh" in finding.hint


def test_time_limit_merges_into_the_timeout_state():
    diagnosis = diagnose_job(
        _job("TIMEOUT", log="time_limit.log", signal=15, job_id="90999003")
    )
    assert _codes(diagnosis) == ["state.timeout"]
    primary = diagnosis.findings[0]
    assert "DUE TO TIME LIMIT" in primary.evidence["line"]
    assert "--time" in primary.hint
    # The same log under another state is its own finding.
    other = diagnose_job(_job("FAILED", log="time_limit.log", exit_code=0, signal=15))
    assert "log.time_limit" in _codes(other)


def test_preemption_and_node_failure():
    preempted = diagnose_job(_job("PREEMPTED", log="preempted.log"))
    assert _codes(preempted) == ["state.preempted"]
    assert preempted.findings[0].severity == "warning"
    assert "--requeue" in preempted.findings[0].hint
    node = diagnose_job(_job("FAILED", log="node_failure.log", exit_code=1))
    assert "log.node_failure" in _codes(node)
    assert "--exclude" in _finding(node, "log.node_failure").hint


def test_host_oom_merges_and_killed_line_yields():
    diagnosis = diagnose_job(
        _job("OUT_OF_MEMORY", log="host_oom.log", exit_code=0, signal=125)
    )
    assert _codes(diagnosis) == ["state.out_of_memory"]
    assert "oom_kill" in diagnosis.findings[0].evidence["line"]
    assert "--mem" in diagnosis.findings[0].hint
    as_failed = diagnose_job(_job("FAILED", log="host_oom.log", exit_code=137))
    assert "log.host_oom" in _codes(as_failed) and "log.killed" not in _codes(as_failed)


def test_a_bare_killed_line_is_hedged():
    diagnosis = diagnose_job(_job("FAILED", log="killed.log", exit_code=137))
    finding = _finding(diagnosis, "log.killed")
    assert finding.severity == "warning" and "most often" in finding.message


def test_cancelled_log_line():
    assert _codes(diagnose_job(_job("CANCELLED", log="cancelled.log"))) == [
        "state.cancelled"
    ]
    running = diagnose_job(_job("FAILED", log="cancelled.log", exit_code=0, signal=15))
    assert "log.cancelled" in _codes(running)


def test_last_python_exception_when_nothing_specific():
    diagnosis = diagnose_job(_job(log="python_error.log", exit_code=1))
    finding = _finding(diagnosis, "log.python_exception")
    assert finding.message.endswith("KeyError: 'abs_rel_sparse'")


def test_clean_log_has_only_the_state():
    diagnosis = diagnose_job(_job("COMPLETED", log="clean.log", exit_code=0))
    assert _codes(diagnosis) == ["state.completed"]
    assert diagnosis.summary() == {
        "counts": {"error": 0, "warning": 0, "info": 1},
        "worst": "info",
    }


def test_every_rule_has_a_log_fixture_that_triggers_it():
    triggered = set()
    for log in LOGS.glob("*.log"):
        lines = read_tail(str(log), 200)
        triggered |= set(scan_lines(lines, str(log), "job 1"))
    assert {rule.topic for rule in RULES} | {"python_exception"} <= triggered


def test_unreadable_and_unknown_logs():
    missing = diagnose_job(
        _job(
            stdout_path="/nonexistent/x.out",
            stderr_path="/nonexistent/x.err",
            exit_code=1,
        )
    )
    unreadable = [f for f in missing.findings if f.code == "log.unreadable"]
    assert len(unreadable) == 2 and all(f.severity == "info" for f in unreadable)
    assert "does not exist" in unreadable[0].message
    unknown = diagnose_job(_job(exit_code=1))
    (finding,) = [f for f in unknown.findings if f.code == "log.unreadable"]
    assert (
        "did not report the path" in finding.message
        and "scontrol show job" in finding.hint
    )
    discarded = diagnose_job(
        _job(stdout_path="/dev/null", stderr_path="/dev/null", exit_code=1)
    )
    assert "log.unreadable" not in _codes(discarded)
    pattern = read_logs(_job(stdout_path="/w/%N.out", stderr_path=None), 10)
    assert "pattern" in pattern[0].error


def test_separate_stderr_is_read_too(tmp_path):
    out = tmp_path / "job.out"
    err = tmp_path / "job.err"
    out.write_text("step 1\nstep 2\n")
    err.write_text(fixture_text("logs/import_error.log"))
    diagnosis = diagnose_job(
        _job(stdout_path=str(out), stderr_path=str(err), exit_code=1)
    )
    assert [t.stream for t in diagnosis.logs] == ["stdout", "stderr"]
    assert _finding(diagnosis, "log.import_error").evidence["path"] == str(err)


def test_pending_job_reads_no_logs_and_explains_the_reason():
    job = Job(
        job_id="91000015",
        state="PENDING",
        state_reason="JobHeldUser",
        stdout_path="/nonexistent/never.out",
    )
    diagnosis = diagnose_job(job)
    assert diagnosis.logs == [] and _codes(diagnosis) == ["pending.jobhelduser"]
    finding = diagnosis.findings[0]
    assert finding.severity == "error" and "scontrol release 91000015" in finding.hint
    detail = diagnose_job(
        Job(
            job_id="1",
            state="PENDING",
            state_reason="ReqNodeNotAvail, UnavailableNodes:eu-g1-[001-004]",
        )
    )
    assert "SLURM adds: UnavailableNodes:eu-g1-[001-004]" in detail.findings[0].message
    assert detail.findings[0].code == "pending.reqnodenotavail"


def test_near_time_limit_warning():
    job = Job(
        job_id="3",
        state="RUNNING",
        time_used_s=3500,
        time_limit_s=3600,
        stdout_path="/nonexistent/a.out",
        stderr_path="/nonexistent/a.out",
    )
    diagnosis = diagnose_job(job)
    assert diagnosis.findings[0].code == "job.near_time_limit"
    assert "97%" in diagnosis.findings[0].message


def test_read_tail_limits(tmp_path):
    big = tmp_path / "big.log"
    big.write_text("".join(f"line {i}\n" for i in range(50000)))
    lines = read_tail(str(big), 5)
    assert lines == [f"line {i}" for i in range(49995, 50000)]
    assert read_tail(str(big), 0) == []
    huge_line = tmp_path / "huge.log"
    huge_line.write_text("x" * (6 * 1024 * 1024) + "\nlast line\n")
    assert read_tail(str(huge_line), 3)[-1] == "last line"


def test_clean_line():
    assert clean_line("\x1b[31mERROR\x1b[0m boom") == "ERROR boom"
    assert clean_line("[rank3]: RuntimeError: x") == "RuntimeError: x"
    assert clean_line("0: ImportError: y") == "ImportError: y"
    assert clean_line("a\rb\r  \r") == "b"


def test_finding_dict_has_the_euler_suite_shape():
    data = diagnose_job(_job(log="import_error.log", exit_code=1)).findings[1].to_dict()
    assert {"code", "severity", "subject", "message", "hint"} <= set(data)
    assert data["subject"] == "job 90999001"
    json.dumps(data)


def test_diagnose_end_to_end_with_a_runner(tmp_path):
    (tmp_path / "train-90999001.out").write_text("epoch 1\n")
    (tmp_path / "train-90999001.err").write_text(fixture_text("logs/cuda_oom.log"))
    scontrol = fixture_text("scontrol_failed.txt").replace("{WORKDIR}", str(tmp_path))
    runner = text_cluster().add(("scontrol", "--oneliner"), (0, scontrol, ""))
    diagnosis = diagnose("90999001", runner=runner)
    assert diagnosis.job.state == "FAILED" and diagnosis.job.exit_code == 1
    assert _codes(diagnosis)[:2] == ["state.failed", "log.cuda_oom"]
    assert diagnosis.summary()["worst"] == "error"
    client = SlurmClient(runner, clock=lambda: NOW)
    assert diagnose("90999001", 50, client=client).logs[1].readable


def _forgotten():
    """A cluster that no longer knows any job (aged out, no accounting record)."""
    runner = FakeRunner().add(("scontrol",), (1, "", "Invalid job id specified"))
    runner.add(("sacct",), (0, '{"jobs": []}', ""))
    return runner


def test_diagnose_unknown_job_raises():
    with pytest.raises(JobNotFound):
        diagnose("5", runner=_forgotten())


# ---------------------------------------------------------------------------
# Log files a tool names (--log)
# ---------------------------------------------------------------------------


def test_a_named_log_is_read_and_scanned_too():
    diagnosis = diagnose_job(
        _job(log="clean.log", exit_code=1), logs=[str(LOGS / "cuda_oom.log")]
    )
    assert [(t.stream, t.path) for t in diagnosis.logs] == [
        ("stdout+stderr", str(LOGS / "clean.log")),
        ("log", str(LOGS / "cuda_oom.log")),
    ]
    assert all(t.readable for t in diagnosis.logs)
    finding = _finding(diagnosis, "log.cuda_oom")
    assert finding.evidence["path"] == str(LOGS / "cuda_oom.log")
    assert diagnosis.verdict.code == "log.cuda_oom"
    assert diagnosis.logs[1].to_dict()["stream"] == "log"


def test_a_named_log_equal_to_stdout_is_read_once(tmp_path):
    same = str(LOGS) + "/../logs/./import_error.log"  # same file, other spelling
    diagnosis = diagnose_job(_job(log="import_error.log", exit_code=1), logs=[same])
    assert [t.stream for t in diagnosis.logs] == ["stdout+stderr"]
    out, err = tmp_path / "job.out", tmp_path / "job.err"
    out.write_text("ok\n")
    err.write_text("ok\n")
    extra = str(LOGS / "disk_quota.log")
    tails = read_logs(
        _job(stdout_path=str(out), stderr_path=str(err)),
        10,
        [str(err), extra, extra + "/", str(tmp_path / "." / "job.out")],
    )
    assert [(t.stream, t.path) for t in tails] == [
        ("stdout", str(out)),
        ("stderr", str(err)),
        ("log", extra),
    ]


def test_named_logs_are_taken_as_they_are(tmp_path):
    odd = tmp_path / "run%x.log"  # not a SLURM pattern here
    odd.write_text(fixture_text("logs/disk_quota.log"))
    diagnosis = diagnose_job(_job(exit_code=1), logs=[str(odd), "/nonexistent/a.log"])
    assert diagnosis.logs[1].readable and "log.disk_quota" in _codes(diagnosis)
    (unreadable,) = [
        f
        for f in diagnosis.findings
        if f.code == "log.unreadable" and "/nonexistent/a.log" in f.message
    ]
    assert "does not exist" in unreadable.message


def test_a_pending_job_reads_no_named_logs_either():
    job = Job(job_id="7", state="PENDING", state_reason="Priority")
    diagnosis = diagnose_job(job, logs=[str(LOGS / "cuda_oom.log")])
    assert diagnosis.logs == [] and _codes(diagnosis) == ["pending.priority"]


def test_a_rule_yields_across_files(tmp_path):
    out, err = tmp_path / "job.out", tmp_path / "job.err"
    out.write_text(fixture_text("logs/killed.log"))
    err.write_text(fixture_text("logs/host_oom.log"))
    diagnosis = diagnose_job(
        _job(stdout_path=str(out), stderr_path=str(err), exit_code=137)
    )
    assert "log.host_oom" in _codes(diagnosis)
    assert "log.killed" not in _codes(diagnosis)  # the oom-kill explains it


# ---------------------------------------------------------------------------
# When SLURM has forgotten the job: logs only
# ---------------------------------------------------------------------------


def test_diagnose_reads_only_the_logs_of_a_forgotten_job():
    diagnosis = diagnose(
        "5", logs=[str(LOGS / "import_error.log")], runner=_forgotten()
    )
    assert diagnosis.job is None and diagnosis.explanation is None
    assert diagnosis.job_id == "5"
    first = diagnosis.findings[0]
    assert first.code == "job.not_found" and first.severity == "info"
    assert first.message == (
        "SLURM no longer has a record of job 5 (it ended too long ago, or "
        "accounting is off), so this diagnosis reads only its logs."
    )
    assert "`sacct -j 5`" in first.hint
    assert _codes(diagnosis) == ["job.not_found", "log.import_error"]
    assert [t.stream for t in diagnosis.logs] == ["log"]
    verdict = diagnosis.verdict
    assert verdict.summary == "No SLURM record"
    assert verdict.code == "log.import_error" and verdict.severity == "error"
    assert "`euler_loading`" in verdict.cause and "euler-suite doctor" in verdict.fix
    assert verdict.evidence["path"] == str(LOGS / "import_error.log")
    assert diagnosis.summary()["worst"] == "error"


def test_logs_only_without_anything_found():
    diagnosis = diagnose_logs("5", ["/nonexistent/x.log", str(LOGS / "clean.log")])
    assert _codes(diagnosis) == ["job.not_found", "log.unreadable"]
    assert diagnosis.verdict.code == "job.not_found"
    assert diagnosis.verdict.fix and diagnosis.verdict.evidence is None
    with pytest.raises(ValueError):
        diagnose_logs("5;rm", [])


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------


def test_verdict_specific_log_finding_beats_python_exception(tmp_path):
    out, err = tmp_path / "job.out", tmp_path / "job.err"
    out.write_text(fixture_text("logs/python_error.log"))
    err.write_text(fixture_text("logs/import_error.log"))
    diagnosis = diagnose_job(
        _job(stdout_path=str(out), stderr_path=str(err), exit_code=1)
    )
    codes = _codes(diagnosis)
    # Both are listed (the exception first: stdout is read first) …
    assert codes.index("log.python_exception") < codes.index("log.import_error")
    # … but the verdict is the specific one.
    assert diagnosis.verdict.code == "log.import_error"
    assert diagnosis.verdict.evidence["path"] == str(err)


def test_verdict_python_exception_beats_the_state():
    diagnosis = diagnose_job(_job(log="python_error.log", exit_code=1))
    verdict = diagnosis.verdict
    assert verdict.summary == "Failed (exit code 1)"
    assert verdict.code == "log.python_exception" and verdict.severity == "error"
    assert (
        verdict.cause == "The last Python error in the log: KeyError: 'abs_rel_sparse'"
    )
    assert verdict.fix == "The traceback above it in the log shows where it was raised."
    assert verdict.evidence == {
        "path": str(LOGS / "python_error.log"),
        "line": "KeyError: 'abs_rel_sparse'",
    }


def test_verdict_falls_back_to_the_state():
    failed = diagnose_job(_job(log="clean.log", exit_code=1))
    verdict = failed.verdict
    assert verdict.code == "state.failed" and verdict.severity == "error"
    assert verdict.cause == failed.findings[0].message
    assert "euler-slurm diagnose 90999001" in verdict.fix  # <job_id> filled in
    assert verdict.evidence is None
    completed = diagnose_job(_job("COMPLETED", log="clean.log", exit_code=0))
    assert completed.verdict.summary == "Completed"
    assert completed.verdict.code == "state.completed"
    assert completed.verdict.severity == "info"
    # An unreadable log is never the verdict.
    missing = diagnose_job(
        _job(stdout_path="/nonexistent/x.out", stderr_path="/nonexistent/x.err")
    )
    assert missing.verdict.code == "state.failed"


def test_verdict_merged_state_keeps_the_log_line():
    diagnosis = diagnose_job(
        _job("TIMEOUT", log="time_limit.log", signal=15, job_id="90999003")
    )
    verdict = diagnosis.verdict
    assert verdict.summary == "Hit its time limit" and verdict.code == "state.timeout"
    assert "DUE TO TIME LIMIT" in verdict.evidence["line"]
    assert "--time" in verdict.fix


def test_verdict_state_beats_a_traceback_when_slurm_stopped_the_job(tmp_path):
    log = tmp_path / "job.log"
    log.write_text(
        fixture_text("logs/time_limit.log")
        + "Traceback (most recent call last):\n"
        + '  File "train.py", line 9, in <module>\n'
        + "torch.distributed.elastic.multiprocessing.api.SignalException: "
        + "Process 4711 got signal: 15\n"
    )
    job = _job("TIMEOUT", stdout_path=str(log), stderr_path=str(log), signal=15)
    diagnosis = diagnose_job(job)
    assert "log.python_exception" in _codes(diagnosis)  # still listed
    assert diagnosis.verdict.code == "state.timeout"  # but not the cause
    killed = diagnose_job(_job("OUT_OF_MEMORY", log="killed.log", signal=125))
    assert killed.verdict.code == "state.out_of_memory"
    # A specific cause in the log still wins over the state.
    nccl = diagnose_job(_job("TIMEOUT", log="nccl_timeout.log", signal=15))
    assert nccl.verdict.code == "log.distributed_timeout"


def test_verdict_warning_log_finding_beats_the_state():
    diagnosis = diagnose_job(_job("FAILED", log="killed.log", exit_code=137))
    assert diagnosis.verdict.code == "log.killed"
    assert diagnosis.verdict.severity == "warning"
    assert "sacct -j 90999001" in diagnosis.verdict.fix


def test_verdict_of_a_pending_job_is_its_reason():
    job = Job(job_id="91000015", state="PENDING", state_reason="JobHeldUser")
    verdict = diagnose_job(job).verdict
    assert verdict.summary == "Held by you" and verdict.code == "pending.jobhelduser"
    assert verdict.fix == "Release it: `scontrol release 91000015`."
    assert verdict.severity == "error" and verdict.evidence is None
    waiting = diagnose_job(Job(job_id="9", state="PENDING", state_reason="Priority"))
    assert waiting.verdict.summary == "Waiting for its turn"
    assert waiting.verdict.to_dict() == {
        "summary": "Waiting for its turn",
        "cause": waiting.findings[0].message,
        "fix": waiting.findings[0].hint,
        "code": "pending.priority",
        "severity": "info",
        "evidence": None,
    }
