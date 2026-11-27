import json
import subprocess
import sys

import pytest

from euler_slurm import __version__, catalog
from euler_slurm.cli import EXIT_FAILED, EXIT_OK, EXIT_UNAVAILABLE, EXIT_USAGE, main
from tests.conftest import LOGS, FakeRunner, fixture_text, json_cluster, text_cluster


def _run(capsys, argv, runner=None):
    code = main(argv, runner=runner)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_jobs_human_table(capsys):
    code, out, _ = _run(capsys, ["jobs"], json_cluster())
    assert code == EXIT_OK
    lines = out.splitlines()
    assert lines[0].split()[:3] == ["JOB", "ID", "NAME"]
    row = next(line for line in lines if line.startswith("91000009"))
    assert "Waiting for its turn (Priority)" in row and "1 GPU" in row
    running = next(line for line in lines if line.startswith("91000003 "))
    assert "1x rtx_4090" in running and "eu-g6-023" in running
    oom = next(line for line in lines if line.startswith("90999002"))
    assert oom.rstrip().endswith("Ran out of memory")
    held = next(line for line in lines if line.startswith("91000015"))
    assert "Held by you (JobHeldUser)" in held
    assert "euler-slurm diagnose <id>" in out


def test_jobs_json_envelope(capsys):
    code, out, _ = _run(
        capsys, ["jobs", "--json", "--since-hours", "6"], text_cluster()
    )
    document = json.loads(out)
    assert code == EXIT_OK and document["contract"] == {
        "kind": "slurm_jobs",
        "version": "1.0",
    }
    assert (
        document["query"]["since_hours"] == 6
        and document["sources"]["squeue"] == "text"
    )


def test_jobs_empty_and_problems(capsys):
    runner = FakeRunner(missing=["sacct"]).add(
        ("squeue", "--json"), (0, '{"jobs": []}', "")
    )
    code, out, _ = _run(capsys, ["jobs", "--user", "carol"], runner)
    assert code == EXIT_OK
    assert "No jobs for carol in the queue" in out
    assert "note: " in out and "finished jobs are not listed" in out


def test_job_human_and_json(capsys):
    runner = text_cluster().add(
        ("scontrol", "--oneliner"), (0, fixture_text("scontrol_pending.txt"), "")
    )
    code, out, _ = _run(capsys, ["job", "91000014"], runner)
    assert code == EXIT_OK
    assert out.startswith("Job 91000014  multinode  (user alice)")
    assert "PENDING (ReqNodeNotAvail, UnavailableNodes:eu-g1-[001-004])" in out
    assert "SLURM adds:" in out and "8x rtx_4090" in out
    assert "What to do:" in out and "scontrol update" in out
    code, out, _ = _run(capsys, ["job", "91000014", "--json"], runner)
    document = json.loads(out)
    assert document["contract"]["kind"] == "slurm_job"
    assert document["job"]["explanation"]["code"] == "ReqNodeNotAvail"


def test_job_not_found(capsys):
    runner = FakeRunner().add(("scontrol",), (1, "", "Invalid job id specified"))
    runner.add(("sacct",), (0, '{"jobs": []}', ""))
    code, _, err = _run(capsys, ["job", "4242"], runner)
    assert code == EXIT_FAILED and "does not know job 4242" in err
    code, out, _ = _run(capsys, ["job", "4242", "--json"], runner)
    document = json.loads(out)
    assert code == EXIT_FAILED and document["error"]["code"] == "job_not_found"


def test_without_slurm(capsys):
    runner = FakeRunner(missing=["squeue", "sacct", "scontrol", "sinfo"])
    code, _, err = _run(capsys, ["jobs"], runner)
    assert code == EXIT_UNAVAILABLE and "not found on PATH" in err
    code, out, _ = _run(capsys, ["diagnose", "5", "--json"], runner)
    assert (
        code == EXIT_UNAVAILABLE
        and json.loads(out)["error"]["code"] == "slurm_unavailable"
    )
    code, out, _ = _run(capsys, ["capabilities"], runner)
    assert code == EXIT_OK and json.loads(out)["available"] is False


def test_invalid_job_id_is_a_usage_error(capsys):
    code, _, err = _run(capsys, ["job", "12;rm"], FakeRunner())
    assert code == EXIT_USAGE and "not a SLURM job id" in err


def test_diagnose_human_and_json(capsys, tmp_path):
    (tmp_path / "train-90999001.out").write_text("ok\n")
    (tmp_path / "train-90999001.err").write_text(
        (LOGS / "import_error.log").read_text()
    )
    scontrol = fixture_text("scontrol_failed.txt").replace("{WORKDIR}", str(tmp_path))
    runner = text_cluster().add(("scontrol", "--oneliner"), (0, scontrol, ""))
    code, out, _ = _run(capsys, ["diagnose", "90999001"], runner)
    assert code == EXIT_OK
    lines = out.splitlines()
    assert lines[0] == "Job 90999001 (train) - FAILED, exit code 1"
    # The verdict comes first: summary, cause, fix.
    assert lines[2] == "  Summary:    Failed (exit code 1)"
    assert lines[3].startswith("  Cause:      Python could not import `euler_loading`")
    assert any(
        line.startswith("  Fix:        Run `euler-suite doctor`") for line in lines
    )
    assert "Also found:" in out and "[error]     Failed (exit code 1)" in out
    assert out.count("Python could not import") == 1  # not repeated below
    assert "ModuleNotFoundError: No module named 'euler_loading'" in out
    code, out, _ = _run(
        capsys, ["diagnose", "90999001", "--json", "--lines", "20"], runner
    )
    document = json.loads(out)
    assert document["contract"] == {"kind": "slurm_diagnosis", "version": "1.1"}
    assert "log.import_error" in [f["code"] for f in document["findings"]]
    assert document["verdict"]["code"] == "log.import_error"
    assert document["verdict"]["summary"] == "Failed (exit code 1)"


def test_diagnose_with_named_logs(capsys, tmp_path):
    (tmp_path / "train-90999001.out").write_text("ok\n")
    (tmp_path / "train-90999001.err").write_text("ok\n")
    scontrol = fixture_text("scontrol_failed.txt").replace("{WORKDIR}", str(tmp_path))
    runner = text_cluster().add(("scontrol", "--oneliner"), (0, scontrol, ""))
    argv = [
        "diagnose",
        "90999001",
        "--log",
        str(LOGS / "cuda_oom.log"),
        "--log",
        str(tmp_path / "train-90999001.err"),  # SLURM's stderr: read once
        "--json",
    ]
    code, out, _ = _run(capsys, argv, runner)
    document = json.loads(out)
    assert code == EXIT_OK
    assert [(t["stream"], t["readable"]) for t in document["logs"]] == [
        ("stdout", True),
        ("stderr", True),
        ("log", True),
    ]
    assert document["verdict"]["code"] == "log.cuda_oom"
    assert document["verdict"]["evidence"]["path"] == str(LOGS / "cuda_oom.log")
    code, out, _ = _run(capsys, argv[:-1], runner)
    assert code == EXIT_OK and "Cause:      The job ran out of GPU memory" in out


def test_diagnose_a_forgotten_job_from_its_logs(capsys):
    runner = FakeRunner().add(("scontrol",), (1, "", "Invalid job id specified"))
    runner.add(("sacct",), (0, '{"jobs": []}', ""))
    log = str(LOGS / "nccl_timeout.log")
    code, out, _ = _run(capsys, ["diagnose", "4242", "--log", log, "--json"], runner)
    document = json.loads(out)
    assert code == EXIT_OK and document["job"] is None
    assert "error" not in document
    assert document["findings"][0]["code"] == "job.not_found"
    assert document["verdict"]["summary"] == "No SLURM record"
    assert document["verdict"]["code"] == "log.distributed_timeout"
    code, out, _ = _run(capsys, ["diagnose", "4242", "--log", log], runner)
    assert code == EXIT_OK
    assert out.splitlines()[0] == "Job 4242 - no SLURM record; read from its logs only"
    assert "Summary:    No SLURM record" in out
    # Without --log it stays an error.
    code, out, _ = _run(capsys, ["diagnose", "4242", "--json"], runner)
    assert code == EXIT_FAILED
    assert json.loads(out)["error"]["code"] == "job_not_found"


def test_explain(capsys):
    code, out, _ = _run(capsys, ["explain", "QOSMaxGRESPerUser"])
    assert code == EXIT_OK
    assert out.startswith("QOSMaxGRESPerUser: At your GPU limit (pending reason)")
    assert "Meaning:" in out and "What to do:" in out
    _, out, _ = _run(capsys, ["explain", "launch", "failed", "requeued", "held"])
    assert "Start failed; put back and held" in out and "will not start unless" in out
    _, out, _ = _run(capsys, ["explain", "137"])
    assert out.startswith("137: Killed (SIGKILL) (exit code)")
    _, out, _ = _run(capsys, ["explain", "TO"])
    assert out.startswith("TIMEOUT: Hit its time limit (job state)")
    _, out, _ = _run(capsys, ["explain", "NotARealReason"])
    assert "does not guess" in out


def test_explain_all_json_is_the_catalog(capsys):
    code, out, _ = _run(capsys, ["explain", "--all", "--json"])
    document = json.loads(out)
    assert code == EXIT_OK
    assert document["contract"] == {"kind": "slurm_explanations", "version": "1.0"}
    assert document["package"]["name"] == "euler-slurm"
    assert document["catalog"] == catalog() and "explanations" not in document


def test_explain_codes_json(capsys):
    code, out, _ = _run(
        capsys, ["explain", "launch", "failed", "requeued", "held", "--json"]
    )
    document = json.loads(out)
    assert code == EXIT_OK
    assert document["contract"] == {"kind": "slurm_explanations", "version": "1.0"}
    (explanation,) = document["explanations"]
    assert explanation["code"] == "launch failed requeued held"
    assert explanation["category"] == "hold" and explanation["blocked"] is True
    assert "<job_id>" in explanation["fix"]
    _, out, _ = _run(capsys, ["explain", "0:9", "--json"])
    (explanation,) = json.loads(out)["explanations"]
    assert explanation["code"] == "137" and explanation["raw"] == "0:9"


def test_explain_all_human_listing(capsys):
    code, out, _ = _run(capsys, ["explain", "--all"])
    assert code == EXIT_OK
    lines = out.splitlines()
    assert lines[0] == "Pending reasons (86):"
    assert "Job states (24):" in lines and "Exit codes (15):" in lines
    priority = next(line for line in lines if line.split()[:1] == ["Priority"])
    assert priority.endswith("  Waiting for its turn")
    assert any(line.split()[:2] == ["0:125", "Killed"] for line in lines)
    assert any(line.split()[:2] == ["PENDING", "Waiting"] for line in lines)


def test_explain_usage_errors(capsys):
    code, _, err = _run(capsys, ["explain"])
    assert code == EXIT_USAGE and "--all" in err
    code, _, _ = _run(capsys, ["explain", "--json"])
    assert code == EXIT_USAGE
    code, _, err = _run(capsys, ["explain", "--all", "Priority"])
    assert code == EXIT_USAGE and "not both" in err


def test_no_slurm_json_flag(capsys):
    runner = text_cluster()
    code, _, _ = _run(capsys, ["--no-slurm-json", "jobs", "--active"], runner)
    assert code == EXIT_OK and not any("--json" in call for call in runner.calls)
    assert not runner.called("sacct")


def test_usage_errors_exit_2(capsys):
    with pytest.raises(SystemExit) as caught:
        main([])
    assert caught.value.code == 2
    with pytest.raises(SystemExit) as caught:
        main(["--version"])
    assert caught.value.code == 0
    assert __version__ == "0.2.0"
    assert f"euler-slurm {__version__}" in capsys.readouterr().out


def test_console_script_and_module_run():
    result = subprocess.run(
        [sys.executable, "-m", "euler_slurm", "explain", "Priority"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0 and "Waiting for its turn" in result.stdout
