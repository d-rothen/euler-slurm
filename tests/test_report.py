import importlib.metadata
import json

import pytest

from euler_slurm import capability_report
from euler_slurm.diagnose import diagnose_job, diagnose_logs
from euler_slurm.explain import catalog, explain
from euler_slurm.jobs import JobNotFound, SlurmClient
from euler_slurm.report import (
    CAPABILITY_KIND,
    ENVELOPES,
    catalog_envelope,
    diagnosis_envelope,
    error_envelope,
    explanations_envelope,
    job_envelope,
    jobs_envelope,
    parse_sinfo_partitions,
    parse_slurm_version,
)
from euler_slurm.runner import SlurmCommandError, SlurmUnavailable
from tests.conftest import LOGS, NOW, FakeRunner, fixture_text, json_cluster


def _healthy_cluster(version="slurm 23.11.4\n"):
    return FakeRunner(
        [
            (("sinfo", "--version"), (0, version, "")),
            (("squeue", "--help"), (0, fixture_text("squeue_help.txt"), "")),
            (("sinfo", "--format"), (0, fixture_text("sinfo_partitions.txt"), "")),
        ]
    )


def _assert_envelope(document, kind):
    assert document["contract"] == {"kind": kind, "version": ENVELOPES[kind]}
    assert document["package"]["name"] == "euler-slurm"
    assert document["package"]["version"]
    json.dumps(document)  # JSON-ready


def test_envelope_versions():
    assert ENVELOPES == {
        "slurm_jobs": "1.0",
        "slurm_job": "1.0",
        "slurm_diagnosis": "1.1",
        "slurm_explanations": "1.0",
        "slurm_capability_report": "1.1",
    }


def test_jobs_envelope():
    listing = SlurmClient(json_cluster(), clock=lambda: NOW).list_jobs()
    document = jobs_envelope(listing)
    _assert_envelope(document, "slurm_jobs")
    assert document["query"] == {"user": "alice", "job_ids": None, "since_hours": 24}
    assert document["sources"] == {"squeue": "json", "sacct": "json"}
    by_id = {job["job_id"]: job for job in document["jobs"]}
    assert by_id["91000009"]["explanation"]["code"] == "Priority"
    assert by_id["91000009"]["explanation"]["fix"].count("91000009") == 2
    assert by_id["90999003"]["explanation"]["code"] == "TIMEOUT"
    assert by_id["91000003"]["gpus"] == {"count": 1, "type": "rtx_4090"}
    assert document["problems"] == [] and document["generated_at"].endswith("+00:00")


def test_job_and_diagnosis_envelopes():
    job = SlurmClient(json_cluster(), clock=lambda: NOW).get_job("91000003")
    document = job_envelope(job)
    _assert_envelope(document, "slurm_job")
    assert document["job"]["job_id"] == "91000003"
    assert document["job"]["explanation"]["kind"] == "state"

    from dataclasses import replace

    failed = replace(
        job,
        state="FAILED",
        exit_code=1,
        stdout_path=str(LOGS / "cuda_oom.log"),
        stderr_path=str(LOGS / "cuda_oom.log"),
    )
    document = diagnosis_envelope(diagnose_job(failed))
    _assert_envelope(document, "slurm_diagnosis")
    assert document["contract"]["version"] == "1.1"
    assert list(document)[3:] == ["job", "summary", "verdict", "findings", "logs"]
    assert document["summary"]["worst"] == "error"
    assert [f["code"] for f in document["findings"]][:2] == [
        "state.failed",
        "log.cuda_oom",
    ]
    assert document["verdict"] == {
        "summary": "Failed (exit code 1)",
        "cause": "The job ran out of GPU memory (CUDA out of memory).",
        "fix": document["findings"][1]["hint"],
        "code": "log.cuda_oom",
        "severity": "error",
        "evidence": document["findings"][1]["evidence"],
    }
    assert document["logs"][0] == {
        "stream": "stdout+stderr",
        "path": str(LOGS / "cuda_oom.log"),
        "readable": True,
        "lines_read": 8,
        "error": None,
    }


def test_logs_only_diagnosis_envelope():
    document = diagnosis_envelope(diagnose_logs("4242", [str(LOGS / "clean.log")]))
    _assert_envelope(document, "slurm_diagnosis")
    assert document["job"] is None
    assert document["findings"][0]["code"] == "job.not_found"
    assert document["verdict"]["summary"] == "No SLURM record"
    assert document["verdict"]["code"] == "job.not_found"
    assert document["verdict"]["evidence"] is None
    assert document["logs"][0]["stream"] == "log"


def test_explanations_envelopes():
    document = catalog_envelope()
    _assert_envelope(document, "slurm_explanations")
    assert document["contract"]["version"] == "1.0"
    assert document["catalog"] == catalog() and "explanations" not in document
    document = explanations_envelope([explain("0:9")])
    _assert_envelope(document, "slurm_explanations")
    assert document["explanations"] == [explain("0:9").to_dict()]
    assert "catalog" not in document


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (
            SlurmUnavailable("squeue", "`squeue` was not found on PATH"),
            "slurm_unavailable",
        ),
        (JobNotFound("5"), "job_not_found"),
        (SlurmCommandError(["squeue"], 1, "boom"), "slurm_command_failed"),
        (ValueError("bad id"), "invalid_argument"),
        (RuntimeError("other"), "error"),
    ],
)
def test_error_envelope(error, code):
    document = error_envelope("slurm_job", error)
    _assert_envelope(document, "slurm_job")
    assert document["error"]["code"] == code and document["error"]["message"]


def test_capability_report_on_a_cluster():
    runner = _healthy_cluster()
    report = capability_report(runner)
    _assert_envelope(report, CAPABILITY_KIND)
    assert report["available"] is True and report["reason"] is None
    assert report["slurm"]["version"] == "23.11.4"
    assert report["slurm"]["json"] == {
        "supported": True,
        "basis": "squeue --help lists --json",
    }
    partitions = {p["name"]: p for p in report["partitions"]}
    gpu4h = partitions["gpu.4h"]
    assert gpu4h["default"] is True and gpu4h["available"] == "up"
    assert gpu4h["time_limit_s"] == 14400 and gpu4h["nodes"] == 32
    assert gpu4h["gpus"] == [
        {"type": "rtx_4090", "per_node": 8, "nodes": 24, "total": 192},
        {"type": "rtx_3090", "per_node": 8, "nodes": 8, "total": 64},
    ]
    assert gpu4h["cpus_per_node"] == [64, 128]
    assert partitions["gpu.120h"]["gpus"] == [
        {"type": "a100_80gb", "per_node": 4, "nodes": 6, "total": 24}
    ]
    assert partitions["gpu.120h"]["time_limit_s"] == 5 * 86400
    assert partitions["normal.4h"]["gpus"] == [] and partitions["normal.4h"][
        "cpus_per_node"
    ] == [128]
    assert (
        partitions["debug"]["time_limit_unlimited"]
        and partitions["debug"]["available"] == "down"
    )
    assert partitions["debug"]["gpus"] == [
        {"type": None, "per_node": 2, "nodes": 2, "total": 4}
    ]
    assert [c["name"] for c in report["commands"]] == [
        "jobs",
        "job",
        "diagnose",
        "explain",
        "capabilities",
    ]
    assert {e["kind"] for e in report["envelopes"]} == set(ENVELOPES)
    commands = {c["name"]: c for c in report["commands"]}
    assert commands["explain"]["envelope"] == "slurm_explanations"
    assert "--all" in commands["explain"]["usage"]
    assert "--log PATH" in commands["diagnose"]["usage"]
    assert "QOSMaxGRESPerUser" in report["explanations"]["pending_reasons"]
    assert report["catalog"] == catalog()
    assert report["user"] == "alice" and report["problems"] == []


def test_capability_report_without_slurm_never_raises():
    report = capability_report(FakeRunner(missing=["sinfo", "squeue"]))
    _assert_envelope(report, CAPABILITY_KIND)
    assert report["available"] is False
    assert "sinfo" in report["reason"] and "not found" in report["reason"]
    assert report["problems"] == []  # no SLURM here is not a problem
    assert report["commands"] and report["explanations"]["states"]
    # The catalog does not need SLURM: tools read it from any report.
    assert report["contract"]["version"] == "1.1"
    assert report["catalog"] == catalog()
    assert len(report["catalog"]["pending_reasons"]) == 86


def test_capability_report_real_machine_never_raises():
    report = capability_report()  # whatever this machine has
    assert report["contract"]["kind"] == CAPABILITY_KIND
    assert isinstance(report["available"], bool)
    json.dumps(report)


@pytest.mark.parametrize(
    "runner",
    [
        FakeRunner([(("sinfo",), RuntimeError("the runner itself broke"))]),
        FakeRunner([(("sinfo",), MemoryError("no memory left"))]),
        FakeRunner([(("sinfo", "--version"), lambda args: None)]),  # not a tuple
    ],
)
def test_capability_report_survives_broken_runners(runner):
    report = capability_report(runner)
    assert report["available"] is False and report["reason"].startswith(
        "internal error"
    )
    assert report["catalog"]["states"]


def test_capability_report_controller_down_is_a_problem():
    runner = _healthy_cluster()
    runner.rules[2] = (
        ("sinfo", "--format"),
        (1, "", "sinfo: error: Unable to contact slurm controller (connect failure)\n"),
    )
    report = capability_report(runner)
    assert report["available"] is False
    assert report["problems"][0]["code"] == "slurm_unreachable"
    assert "Unable to contact slurm controller" in report["problems"][0]["message"]
    assert report["slurm"]["version"] == "23.11.4"


def test_capability_report_old_slurm_has_no_json():
    runner = _healthy_cluster("slurm 20.11.9\n")
    report = capability_report(runner)
    assert report["available"] is True
    assert report["slurm"]["json"] == {
        "supported": False,
        "basis": "SLURM 20.11.9 predates --json (21.08)",
    }
    assert not runner.called("squeue", "--help")
    no_json_help = _healthy_cluster()
    no_json_help.rules[1] = (("squeue", "--help"), (0, "Usage: squeue [-u user]\n", ""))
    assert capability_report(no_json_help)["slurm"]["json"]["supported"] is False


def test_version_and_sinfo_parsing():
    assert parse_slurm_version("slurm-wlm 21.08.5") == ("21.08.5", (21, 8, 5))
    assert parse_slurm_version("slurm 24.05") == ("24.05", (24, 5))
    assert parse_slurm_version("nothing") is None
    assert parse_sinfo_partitions("garbage\n\n|x|\n") == []


def test_entry_point_is_registered():
    entry_points = importlib.metadata.entry_points()
    if hasattr(entry_points, "select"):
        group = entry_points.select(group="euler.capabilities")
    else:  # Python 3.9
        group = entry_points.get("euler.capabilities", [])
    (entry,) = [ep for ep in group if ep.name == "euler-slurm"]
    assert entry.value == "euler_slurm.report:capability_report"
    assert entry.load() is capability_report
