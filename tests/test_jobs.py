import json

import pytest

from euler_slurm import Gpus, JobNotFound, SlurmClient, get_job, list_jobs
from euler_slurm._parse import parse_scontrol_records
from euler_slurm.jobs import (
    Job,
    expand_output_path,
    job_from_sacct_json,
    job_from_sacct_line,
    job_from_scontrol,
    job_from_squeue_json,
    job_from_squeue_line,
    parse_squeue_tres,
    validate_job_id,
)
from euler_slurm.runner import SlurmCommandError, SlurmUnavailable
from tests.conftest import (
    NOW,
    FakeRunner,
    fixture_text,
    json_cluster,
    text_cluster,
)


def _by_id(jobs):
    return {job.job_id: job for job in jobs}


def _client(runner, **kwargs):
    return SlurmClient(runner, clock=lambda: NOW, **kwargs)


# ---------------------------------------------------------------------------
# squeue --json (Slurm 23.11, data_parser v0.0.40)
# ---------------------------------------------------------------------------


def test_squeue_json_v0040_fields():
    data = json.loads(fixture_text("squeue_v0040.json"))
    jobs = _by_id(job_from_squeue_json(raw, now=NOW) for raw in data["jobs"])
    running = jobs["91000003"]
    assert running.raw_job_id == 91000003
    assert running.array_job_id is None and running.array_task_id is None
    assert running.name == "ev-rtx-1120-1430" and running.user == "alice"
    assert running.state == "RUNNING" and running.state_reason is None
    assert running.partition == "gpu.4h" and running.nodes == "eu-g6-023"
    assert running.node_count == 1 and running.cpus == 8
    assert running.gpus == Gpus(1, "rtx_4090")
    assert running.time_limit_s == 4 * 3600 and not running.time_limit_unlimited
    assert running.time_used_s == 4790  # 14:31:10 → 15:51:00
    assert running.submit_time == "2026-11-20T14:30:02+00:00"
    assert running.start_time == "2026-11-20T14:31:10+00:00"
    assert running.exit_code is None and running.signal is None  # not ended
    assert running.stdout_path == "/cluster/home/alice/proj/logs/slurm-91000003.out"
    assert running.work_dir == "/cluster/home/alice/proj"
    assert running.source == "squeue-json"

    pending = jobs["91000009"]
    assert pending.state == "PENDING" and pending.state_reason == "Priority"
    assert pending.pending and not pending.terminal
    assert pending.time_used_s == 0 and pending.time_limit_s == 2 * 86400
    assert pending.gpus == Gpus(1, None)
    assert pending.nodes is None


def test_squeue_json_arrays_flags_and_limits():
    data = json.loads(fixture_text("squeue_v0040.json"))
    jobs = _by_id(job_from_squeue_json(raw, now=NOW) for raw in data["jobs"])
    task = jobs["91000020_1"]
    assert (task.raw_job_id, task.array_job_id, task.array_task_id) == (
        91000021,
        91000020,
        1,
    )
    assert task.gpus == Gpus(1, "a100_80gb")  # type from gres_detail
    meta = jobs["91000020_[2-4%2]"]
    assert meta.array_task_id is None and meta.array_task_range == "2-4%2"
    assert meta.state_reason == "JobArrayTaskLimit"
    assert meta.stdout_path == "/cluster/home/alice/sweep/sweep-91000020_%a.out"
    multinode = jobs["91000014"]
    assert multinode.state_reason == "ReqNodeNotAvail, UnavailableNodes:eu-g1-[001-004]"
    assert multinode.gpus == Gpus(8, "rtx_4090")  # total 8, typed per node
    completing = jobs["91000004"]
    assert completing.state == "COMPLETING" and completing.state_flags == ("COMPLETED",)
    unlimited = jobs["91000012"]
    assert unlimited.time_limit_s is None and unlimited.time_limit_unlimited
    assert unlimited.name == "debug|shell" and unlimited.gpus == Gpus(2, None)


def test_squeue_json_v0037_plain_numbers():
    data = json.loads(fixture_text("squeue_v0037.json"))
    jobs = _by_id(job_from_squeue_json(raw, now=NOW) for raw in data["jobs"])
    running = jobs["81000001"]
    assert running.state == "RUNNING" and running.time_limit_s == 14400
    assert running.gpus == Gpus(1, "rtx_4090") and running.time_used_s == 4790
    pending = jobs["81000002"]
    assert pending.state_reason == "Resources"
    assert pending.time_limit_unlimited and pending.time_limit_s is None
    task = jobs["81000010_4"]
    assert task.array_task_id == 4 and task.gpus == Gpus(1, "rtx_3090")


def test_raw_wait_status_exit_codes_of_old_json():
    base = {"job_id": 5, "job_state": "FAILED", "start_time": 100, "end_time": 200}
    assert job_from_squeue_json({**base, "exit_code": 256}, now=NOW).exit_code == 1
    killed = job_from_squeue_json({**base, "exit_code": 9}, now=NOW)
    assert (killed.exit_code, killed.signal, killed.shell_exit_code) == (0, 9, 137)
    assert killed.time_used_s == 100


# ---------------------------------------------------------------------------
# sacct --json
# ---------------------------------------------------------------------------


def test_sacct_json_terminal_states_and_exit_codes():
    data = json.loads(fixture_text("sacct_v0040.json"))
    jobs = _by_id(job_from_sacct_json(raw) for raw in data["jobs"])
    failed = jobs["90999001"]
    assert failed.state == "FAILED" and failed.terminal
    assert (failed.exit_code, failed.signal) == (1, None)
    assert failed.state_reason == "NonZeroExitCode"
    assert failed.time_used_s == 2520 and failed.time_limit_s == 14400
    assert failed.end_time == "2026-11-20T06:42:10+00:00"
    assert failed.stdout_path == "/cluster/home/alice/proj/logs/train-90999001.out"
    assert failed.stderr_path == "/cluster/home/alice/proj/logs/train-90999001.err"
    assert failed.gpus == Gpus(1, "rtx_4090") and failed.cpus == 8
    assert failed.source == "sacct-json"
    oom = jobs["90999002"]
    assert oom.state == "OUT_OF_MEMORY" and (oom.exit_code, oom.signal) == (0, 125)
    timeout = jobs["90999003"]
    assert (
        timeout.state == "TIMEOUT"
        and timeout.signal == 15
        and timeout.shell_exit_code == 143
    )
    assert jobs["90999004"].state == "CANCELLED"
    assert jobs["90999005"].exit_code == 0 and jobs["90999005"].stdout_path.endswith(
        "ok.out"
    )
    assert (
        jobs["90999006"].state == "NODE_FAIL" and jobs["90999006"].nodes == "eu-g6-017"
    )
    assert jobs["90999007"].state == "PREEMPTED"
    task = jobs["90999010_3"]
    assert (task.raw_job_id, task.array_job_id, task.array_task_id) == (
        90999013,
        90999010,
        3,
    )
    assert task.stdout_path == "/cluster/home/alice/proj/sweep-90999010_003.out"
    still_running = jobs["91000003"]
    assert still_running.exit_code is None and not still_running.terminal


# ---------------------------------------------------------------------------
# Text paths
# ---------------------------------------------------------------------------


def test_squeue_text_lines():
    tres = parse_squeue_tres(fixture_text("squeue_tres.txt"))
    jobs = _by_id(
        job_from_squeue_line(line, tres)
        for line in fixture_text("squeue_format.txt").splitlines()
    )
    running = jobs["91000003"]
    assert running.gpus == Gpus(1, "rtx_4090")
    assert running.time_used_s == 4790 and running.time_limit_s == 14400
    assert (
        running.start_time == "2026-11-20T14:31:10+00:00" and running.source == "squeue"
    )
    task = jobs["91000020_1"]
    assert (task.raw_job_id, task.array_job_id, task.array_task_id) == (
        91000021,
        91000020,
        1,
    )
    assert task.gpus == Gpus(1, "a100_80gb")
    meta = jobs["91000020_[2-4%2]"]
    assert meta.array_task_range == "2-4%2" and meta.start_time is None
    multinode = jobs["91000014"]
    assert multinode.state_reason.startswith("ReqNodeNotAvail, UnavailableNodes:")
    assert multinode.gpus == Gpus(8, "rtx_4090") and multinode.node_count == 2
    assert jobs["91000004"].state == "COMPLETING"
    shell = jobs["91000012"]
    assert shell.name == "debug|shell"  # the delimiter stays in the name
    assert shell.time_limit_unlimited and shell.end_time is None
    assert jobs["91000015"].state_reason == "JobHeldUser"


def test_squeue_text_without_tres_uses_gres_per_node():
    line = fixture_text("squeue_format.txt").splitlines()[4]
    assert job_from_squeue_line(line).gpus == Gpus(8, "rtx_4090")
    assert job_from_squeue_line("garbage|line") is None
    assert job_from_squeue_line("|".join(["x"] * 19)) is None


def test_sacct_parsable2_lines():
    lines = fixture_text("sacct_parsable2.txt").splitlines()
    jobs = [job_from_sacct_line(line) for line in lines]
    assert jobs[1] is None  # the 90999001.batch step
    jobs = _by_id(job for job in jobs if job is not None)
    assert jobs["90999001"].exit_code == 1 and jobs["90999001"].source == "sacct"
    assert (jobs["90999002"].state, jobs["90999002"].signal) == ("OUT_OF_MEMORY", 125)
    cancelled = jobs["90999004"]
    assert cancelled.state == "CANCELLED" and cancelled.cancelled_by == "4290001"
    task = jobs["90999010_3"]
    assert (task.raw_job_id, task.array_job_id, task.array_task_id) == (
        90999013,
        90999010,
        3,
    )
    truncated = jobs["90999008"]
    assert truncated.state == "CANCELLED" and truncated.nodes is None
    assert truncated.start_time is None and truncated.time_limit_unlimited
    assert truncated.name == "never|started"
    pending = jobs["90999030_[1-3]"]
    assert pending.array_task_range == "1-3" and pending.time_limit_s is None
    assert pending.exit_code is None and pending.gpus == Gpus(1, None)


def test_scontrol_records():
    (running,) = [
        job_from_scontrol(r, now=NOW)
        for r in parse_scontrol_records(fixture_text("scontrol_running.txt"))
    ]
    assert running.work_dir == "/cluster/home/alice/my proj"
    assert running.stderr_path == "/cluster/home/alice/my proj/logs/slurm-91000003.err"
    assert running.user == "alice" and running.time_used_s == 4790
    assert running.gpus == Gpus(1, "rtx_4090") and running.source == "scontrol"
    (pending,) = [
        job_from_scontrol(r)
        for r in parse_scontrol_records(fixture_text("scontrol_pending.txt"))
    ]
    assert pending.state_reason == "ReqNodeNotAvail, UnavailableNodes:eu-g1-[001-004]"
    assert pending.node_count == 2 and pending.gpus == Gpus(8, "rtx_4090")
    assert pending.start_time is None and pending.nodes is None
    task, meta = [
        job_from_scontrol(r, now=NOW)
        for r in parse_scontrol_records(fixture_text("scontrol_array.txt"))
    ]
    assert task.job_id == "91000020_1" and task.gpus == Gpus(1, "a100_80gb")
    assert meta.job_id == "91000020_[2-4]" and meta.array_task_range == "2-4"
    (multi,) = [
        job_from_scontrol(r, now=NOW)
        for r in parse_scontrol_records(fixture_text("scontrol_multiline.txt"))
    ]
    assert multi.stdout_path == "/cluster/home/alice/proj/logs/slurm-91000003.out"


def test_json_and_text_paths_agree():
    """The same cluster read both ways gives the same facts."""
    json_jobs = _by_id(_client(json_cluster()).list_jobs(include_finished=False))
    text_jobs = _by_id(_client(text_cluster()).list_jobs(include_finished=False))
    assert set(json_jobs) == set(text_jobs)
    fields = (
        "raw_job_id",
        "array_job_id",
        "array_task_id",
        "array_task_range",
        "name",
        "user",
        "state",
        "state_reason",
        "partition",
        "nodes",
        "node_count",
        "cpus",
        "gpus",
        "time_limit_s",
        "time_limit_unlimited",
        "submit_time",
        "work_dir",
    )
    for job_id, job in json_jobs.items():
        other = text_jobs[job_id]
        for name in fields:
            assert getattr(job, name) == getattr(other, name), (job_id, name)
        assert job.time_used_s == other.time_used_s, job_id  # running and ending
    assert json_jobs["91000004"].time_used_s == 4 * 3600  # COMPLETING: end - start


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------


def test_list_jobs_json_merges_active_and_finished():
    runner = json_cluster()
    listing = _client(runner).list_jobs()
    assert listing.sources == {"squeue": "json", "sacct": "json"}
    assert listing.user == "alice" and listing.since_hours == 24
    ids = [job.job_id for job in listing]
    assert "91000099" not in ids  # bob's job: JSON before 23.02 ignored --user
    assert ids.count("91000003") == 1  # in squeue and sacct: once, from squeue
    assert _by_id(listing)["91000003"].source == "squeue-json"
    finished = [job for job in listing if job.terminal]
    assert [job.end_time for job in finished] == sorted(
        (job.end_time for job in finished), reverse=True
    )
    assert {job.job_id for job in finished} >= {
        "90999001",
        "90999002",
        "90999003",
        "90999010_3",
    }
    assert runner.called("squeue", "--user", "alice", "--json")
    assert runner.called(
        "sacct", "--allocations", "--user", "alice", "--starttime=now-24hours", "--json"
    )
    assert len(listing) == len(listing.jobs) and listing[0] is listing.jobs[0]


def test_list_jobs_text_fallback():
    runner = text_cluster()
    listing = _client(runner).list_jobs(since_hours=6)
    assert listing.sources == {"squeue": "text", "sacct": "text"}
    jobs = _by_id(listing)
    assert jobs["91000003"].source == "squeue"
    assert "90999001" in jobs and "90999030_[1-3]" not in jobs  # sacct: ended only
    assert runner.called("sacct", "--starttime=now-6hours", "--parsable2", "--noheader")
    assert runner.called("squeue", "--noheader", "--Format")


def test_list_jobs_by_id_does_not_filter_by_user():
    runner = json_cluster()
    listing = _client(runner).list_jobs(job_ids=["91000099", 90999001])
    assert {job.job_id for job in listing} == {"91000099", "90999001"}
    assert listing.user is None and listing.since_hours is None
    assert runner.called("squeue", "--jobs=91000099,90999001", "--json")
    assert runner.called("sacct", "--allusers", "--jobs=91000099,90999001")


def test_list_jobs_array_parent_id_matches_its_tasks():
    listing = _client(json_cluster()).list_jobs(
        job_ids=["91000020"], include_finished=False
    )
    assert {job.job_id for job in listing} == {"91000020_1", "91000020_[2-4%2]"}


def test_list_jobs_without_accounting_records_a_problem():
    runner = json_cluster()
    runner.rules = [r for r in runner.rules if r[0][0] != "sacct"]
    runner.add(
        ("sacct",), (1, "", "sacct: error: Slurm accounting storage is disabled\n")
    )
    listing = _client(runner).list_jobs()
    assert listing.sources["sacct"] == "unavailable"
    assert listing.problems[0]["code"] == "sacct_failed"
    assert "accounting storage is disabled" in listing.problems[0]["message"]
    assert all(not job.terminal for job in listing)

    missing = FakeRunner(missing=["sacct"]).add(
        ("squeue", "--json"), (0, fixture_text("squeue_v0040.json"), "")
    )
    listing = _client(missing).list_jobs()
    assert listing.problems[0]["code"] == "sacct_unavailable"


def test_list_jobs_active_only_skips_sacct():
    runner = json_cluster()
    listing = _client(runner).list_jobs("alice", include_finished=False)
    assert listing.sources["sacct"] == "skipped" and not runner.called("sacct")


def test_list_jobs_unknown_job_id_in_text_mode_is_empty():
    runner = FakeRunner().add(
        ("squeue", "--format"),
        (1, "", "slurm_load_jobs error: Invalid job id specified\n"),
    )
    runner.add(("sacct", "--parsable2"), (0, "", ""))
    assert len(_client(runner, use_json=False).list_jobs(job_ids=["123"])) == 0


def test_list_jobs_squeue_failure_raises():
    runner = FakeRunner().add(
        ("squeue", "--format"),
        (1, "", "squeue: error: Unable to contact slurm controller\n"),
    )
    with pytest.raises(SlurmCommandError, match="Unable to contact"):
        _client(runner, use_json=False).list_jobs()
    with pytest.raises(SlurmUnavailable):
        _client(FakeRunner(missing=["squeue"])).list_jobs()


def test_use_json_modes():
    text_only = text_cluster()
    _client(text_only, use_json=False).list_jobs()
    assert not any("--json" in call for call in text_only.calls)
    with pytest.raises(SlurmCommandError):
        _client(text_cluster(), use_json=True).list_jobs()


def test_get_job_scontrol_json_then_text():
    job = _client(json_cluster()).get_job("91000003")
    assert job.source == "scontrol-json" and job.stderr_path.endswith(
        "slurm-91000003.err"
    )
    assert job.time_used_s == 4790
    runner = text_cluster().add(
        ("scontrol", "--oneliner"), (0, fixture_text("scontrol_running.txt"), "")
    )
    job = _client(runner).get_job(91000003)
    assert job.source == "scontrol" and job.work_dir == "/cluster/home/alice/my proj"


def test_get_job_array_task_from_many_records():
    runner = text_cluster().add(
        ("scontrol", "--oneliner"), (0, fixture_text("scontrol_array.txt"), "")
    )
    assert (
        _client(runner).get_job("91000020").job_id == "91000020_[2-4]"
    )  # raw id 91000020
    assert _client(runner).get_job("91000020_1").raw_job_id == 91000021


def test_get_job_falls_back_to_sacct_once_scontrol_forgot_it():
    runner = text_cluster().add(
        ("scontrol", "--oneliner"),
        (1, "", "slurm_load_jobs error: Invalid job id specified\n"),
    )
    job = _client(runner).get_job("90999002")
    assert job.state == "OUT_OF_MEMORY" and job.source == "sacct"
    assert runner.called("sacct", "--allusers", "--jobs=90999002")


def test_get_job_unknown_and_failures():
    runner = FakeRunner().add(
        ("scontrol",), (1, "", "slurm_load_jobs error: Invalid job id specified\n")
    )
    runner.add(("sacct",), (0, json.dumps({"jobs": []}), ""))
    with pytest.raises(JobNotFound, match="does not know job 4242"):
        _client(runner).get_job("4242")
    # The controller is down and accounting has nothing: the real error, not "not found".
    down = FakeRunner().add(
        ("scontrol",),
        (1, "", "slurm_load_jobs error: Unable to contact slurm controller\n"),
    )
    down.add(("sacct",), (0, json.dumps({"jobs": []}), ""))
    with pytest.raises(SlurmCommandError, match="Unable to contact"):
        _client(down).get_job("4242")
    # No scontrol here, but sacct knows it.
    only_sacct = FakeRunner(missing=["scontrol"]).add(
        ("sacct", "--json"), (0, fixture_text("sacct_v0040.json"), "")
    )
    assert _client(only_sacct).get_job("90999005").state == "COMPLETED"
    # Neither is installed.
    with pytest.raises(SlurmUnavailable):
        _client(FakeRunner(missing=["scontrol", "sacct"])).get_job("1")
    # Accounting cannot be asked: say so.
    no_db = FakeRunner(missing=["sacct"]).add(
        ("scontrol",), (1, "", "Invalid job id specified")
    )
    with pytest.raises(JobNotFound, match="accounting database could not be asked"):
        _client(no_db).get_job("7")


@pytest.mark.parametrize("bad", ["abc", "12;rm -rf /", "", "12.batch", "-5"])
def test_job_ids_are_validated(bad):
    with pytest.raises(ValueError):
        validate_job_id(bad)


@pytest.mark.parametrize(
    "good", ["123", "123_4", "123_[1-3]", "123_[1-9:2%4]", "123+1", " 77 "]
)
def test_valid_job_ids(good):
    assert validate_job_id(good) == good.strip()


def test_module_functions_take_a_runner():
    listing = list_jobs(runner=json_cluster(), include_finished=False)
    assert listing.sources["squeue"] == "json"
    assert get_job("91000003", runner=json_cluster()).job_id == "91000003"


def test_expand_output_path():
    job = Job(
        job_id="7_2",
        raw_job_id=9,
        array_job_id=7,
        array_task_id=2,
        name="sweep",
        user="alice",
        work_dir="/w",
    )
    assert expand_output_path("%x-%A_%a.out", job) == "/w/sweep-7_2.out"
    assert expand_output_path("/abs/%j.%u.log", job) == "/abs/9.alice.log"
    assert expand_output_path("o-%4a-%%.txt", job) == "/w/o-0002-%.txt"
    assert expand_output_path("%N.out", job) == "/w/%N.out"  # cannot know the node
    assert expand_output_path("", job) is None


def test_job_to_dict_is_json_ready():
    job = _client(json_cluster()).get_job("91000003")
    data = job.to_dict()
    json.dumps(data)
    assert data["gpus"] == {"count": 1, "type": "rtx_4090"}
    assert data["terminal"] is False and data["state_flags"] == []


def test_json_errors_are_not_an_empty_queue():
    down = {
        "jobs": [],
        "errors": [
            {
                "error": "Unable to contact slurm controller (connect failure)",
                "error_number": 1007,
                "source": "slurm_load_jobs()",
            }
        ],
        "warnings": [],
    }
    runner = FakeRunner().add(("squeue", "--json"), (1, json.dumps(down), ""))
    with pytest.raises(SlurmCommandError, match="Unable to contact slurm controller"):
        _client(runner).list_jobs()
    unknown = {"jobs": [], "errors": [{"error": "Invalid job id specified"}]}
    runner = FakeRunner().add(("squeue", "--json"), (1, json.dumps(unknown), ""))
    runner.add(("sacct", "--json"), (0, json.dumps({"jobs": []}), ""))
    assert len(_client(runner).list_jobs(job_ids=["5"])) == 0
    scontrol_down = FakeRunner().add(("scontrol", "--json"), (1, json.dumps(down), ""))
    scontrol_down.add(("sacct", "--json"), (0, json.dumps({"jobs": []}), ""))
    with pytest.raises(SlurmCommandError, match="Unable to contact"):
        _client(scontrol_down).get_job("5")
    scontrol_unknown = FakeRunner().add(
        ("scontrol", "--json"), (1, json.dumps(unknown), "")
    )
    scontrol_unknown.add(("sacct", "--json"), (0, json.dumps({"jobs": []}), ""))
    with pytest.raises(JobNotFound):
        _client(scontrol_unknown).get_job("5")
