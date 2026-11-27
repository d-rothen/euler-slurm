"""Versioned JSON envelopes, and the ``euler.capabilities`` report.

Every JSON document euler-slurm prints carries ``contract: {kind, version}``
(the suite's envelope rule) and the producing ``package``:

==========================  =======  ==========================================
kind                        version  printed by
==========================  =======  ==========================================
``slurm_jobs``              1.0      ``euler-slurm jobs --json``
``slurm_job``               1.0      ``euler-slurm job <id> --json``
``slurm_diagnosis``         1.1      ``euler-slurm diagnose <id> --json``
``slurm_explanations``      1.0      ``euler-slurm explain --all --json`` (the
                                     catalog), ``explain CODE… --json``
``slurm_capability_report`` 1.1      ``euler-slurm capabilities``, and the
                                     ``euler.capabilities`` entry point
==========================  =======  ==========================================

Minor versions only add fields: ``slurm_diagnosis`` 1.1 added ``verdict`` (and
``job: null`` when only logs could be read), ``slurm_capability_report`` 1.1
the explanation ``catalog``.

Readers should check ``kind`` and the major version and ignore unknown fields.
A failed command prints the same kind with ``error: {code, message}``.

:func:`capability_report` never raises: on a machine without SLURM it returns
``available: false`` and the reason.
"""

from __future__ import annotations

import importlib.metadata
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import __version__
from ._parse import is_unlimited, parse_duration, parse_gpus, text_or_none
from .diagnose import Diagnosis
from .explain import (
    Explanation,
    catalog,
    explain_job,
    known_exit_codes,
    known_reasons,
    known_states,
)
from .jobs import Job, JobListing, JobNotFound, current_user
from .runner import (
    Runner,
    SlurmCommandError,
    SlurmError,
    SlurmUnavailable,
    default_runner,
    first_line,
)

PACKAGE_NAME = "euler-slurm"

JOBS_KIND = "slurm_jobs"
JOB_KIND = "slurm_job"
DIAGNOSIS_KIND = "slurm_diagnosis"
EXPLANATIONS_KIND = "slurm_explanations"
CAPABILITY_KIND = "slurm_capability_report"

#: Every envelope euler-slurm writes: kind → version.
ENVELOPES: Dict[str, str] = {
    JOBS_KIND: "1.0",
    JOB_KIND: "1.0",
    DIAGNOSIS_KIND: "1.1",
    EXPLANATIONS_KIND: "1.0",
    CAPABILITY_KIND: "1.1",
}

#: The CLI commands, as the capability report lists them.
COMMANDS: Tuple[Dict[str, Any], ...] = (
    {
        "name": "jobs",
        "usage": "euler-slurm jobs [--user USER] [--since-hours N] [--active] [--json]",
        "summary": "Your active jobs and the ones that ended recently.",
        "envelope": JOBS_KIND,
    },
    {
        "name": "job",
        "usage": "euler-slurm job JOB_ID [--json]",
        "summary": "One job in detail, with its state or pending reason explained.",
        "envelope": JOB_KIND,
    },
    {
        "name": "diagnose",
        "usage": "euler-slurm diagnose JOB_ID [--log PATH]... [--lines N] [--json]",
        "summary": "Why a job failed or waits: its record and the end of its logs.",
        "envelope": DIAGNOSIS_KIND,
    },
    {
        "name": "explain",
        "usage": "euler-slurm explain (REASON|STATE|EXIT_CODE... | --all) [--json]",
        "summary": (
            "A pending reason, job state or exit code in plain words; --all "
            "lists every one (--all --json: the whole catalog)."
        ),
        "envelope": EXPLANATIONS_KIND,
    },
    {
        "name": "capabilities",
        "usage": "euler-slurm capabilities",
        "summary": "What SLURM offers here (this report).",
        "envelope": CAPABILITY_KIND,
    },
)

#: Slurm's first release with ``--json`` on squeue/sacct/scontrol.
JSON_MIN_VERSION = (21, 8)

#: Seconds each probe of the capability report may take.
PROBE_TIMEOUT = 10.0


def package_info() -> Dict[str, str]:
    try:
        version = importlib.metadata.version(PACKAGE_NAME)
    except importlib.metadata.PackageNotFoundError:
        version = __version__
    return {"name": PACKAGE_NAME, "version": version}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def envelope(kind: str) -> Dict[str, Any]:
    """The common head of every document: contract, package, generation time."""
    return {
        "contract": {"kind": kind, "version": ENVELOPES[kind]},
        "package": package_info(),
        "generated_at": _now(),
    }


def job_dict(job: Job) -> Dict[str, Any]:
    """A job as the envelopes carry it: its fields plus the plain-words explanation."""
    data = job.to_dict()
    data["explanation"] = explain_job(job).to_dict()
    return data


def jobs_envelope(listing: JobListing) -> Dict[str, Any]:
    """``slurm_jobs`` 1.0."""
    document = envelope(JOBS_KIND)
    document.update(
        query={
            "user": listing.user,
            "job_ids": listing.job_ids,
            "since_hours": listing.since_hours,
        },
        sources=dict(listing.sources),
        jobs=[job_dict(job) for job in listing.jobs],
        problems=[dict(problem) for problem in listing.problems],
    )
    return document


def job_envelope(job: Job) -> Dict[str, Any]:
    """``slurm_job`` 1.0."""
    document = envelope(JOB_KIND)
    document["job"] = job_dict(job)
    return document


def diagnosis_envelope(diagnosis: Diagnosis) -> Dict[str, Any]:
    """``slurm_diagnosis`` 1.1: job (``null`` when only logs were read),
    summary, verdict, findings and logs."""
    document = envelope(DIAGNOSIS_KIND)
    document.update(
        job=job_dict(diagnosis.job) if diagnosis.job is not None else None,
        summary=diagnosis.summary(),
        verdict=diagnosis.verdict.to_dict(),
        findings=[finding.to_dict() for finding in diagnosis.findings],
        logs=[tail.to_dict() for tail in diagnosis.logs],
    )
    return document


def catalog_envelope() -> Dict[str, Any]:
    """``slurm_explanations`` 1.0 with the whole ``catalog`` (``explain --all``)."""
    document = envelope(EXPLANATIONS_KIND)
    document["catalog"] = catalog()
    return document


def explanations_envelope(explanations: Sequence[Explanation]) -> Dict[str, Any]:
    """``slurm_explanations`` 1.0 with some ``explanations`` (``explain CODE``)."""
    document = envelope(EXPLANATIONS_KIND)
    document["explanations"] = [explanation.to_dict() for explanation in explanations]
    return document


def error_code(error: BaseException) -> str:
    if isinstance(error, SlurmUnavailable):
        return "slurm_unavailable"
    if isinstance(error, JobNotFound):
        return "job_not_found"
    if isinstance(error, SlurmCommandError):
        return "slurm_command_failed"
    if isinstance(error, ValueError):
        return "invalid_argument"
    return "error"


def error_envelope(kind: str, error: BaseException) -> Dict[str, Any]:
    """The document of ``kind`` for a command that failed."""
    document = envelope(kind)
    document["error"] = {"code": error_code(error), "message": str(error)}
    return document


# ---------------------------------------------------------------------------
# Capability report
# ---------------------------------------------------------------------------

#: ``sinfo --format`` fields for the partitions: name (``*`` = default),
#: availability, time limit, nodes, CPUs and memory (MB) per node, GRES (last).
SINFO_FORMAT = "%P|%a|%l|%D|%c|%m|%G"


def parse_slurm_version(text: str) -> Optional[Tuple[str, Tuple[int, ...]]]:
    """``slurm 23.02.4`` / ``slurm-wlm 21.08.5`` → ("23.02.4", (23, 2, 4))."""
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text or "")
    if match is None:
        return None
    parts = tuple(int(part) for part in match.groups() if part is not None)
    return match.group(0), parts


def parse_sinfo_partitions(output: str) -> List[Dict[str, Any]]:
    """``sinfo --noheader --format SINFO_FORMAT`` → one entry per partition.

    sinfo prints one line per partition *and node configuration*; lines of
    one partition are merged (node counts summed, GPU types listed).
    """
    partitions: Dict[str, Dict[str, Any]] = {}
    for line in output.splitlines():
        fields = line.strip().split("|")
        if len(fields) < 7 or not fields[0].strip():
            continue
        name_text, avail, limit, nodes, cpus, memory = (f.strip() for f in fields[:6])
        gres = "|".join(fields[6:]).strip()
        default = name_text.endswith("*")
        name = name_text.rstrip("*")
        node_count = int(nodes) if nodes.isdigit() else 0
        entry = partitions.setdefault(
            name,
            {
                "name": name,
                "default": default,
                "available": text_or_none(avail),
                "time_limit_s": parse_duration(limit),
                "time_limit_unlimited": is_unlimited(limit),
                "nodes": 0,
                "cpus_per_node": [],
                "memory_mb_per_node": [],
                "gpus": [],
            },
        )
        entry["default"] = entry["default"] or default
        entry["nodes"] += node_count
        cpu_count = cpus.rstrip("+")
        if cpu_count.isdigit() and int(cpu_count) not in entry["cpus_per_node"]:
            entry["cpus_per_node"].append(int(cpu_count))
        memory_count = memory.rstrip("+")
        if (
            memory_count.isdigit()
            and int(memory_count) not in entry["memory_mb_per_node"]
        ):
            entry["memory_mb_per_node"].append(int(memory_count))
        for spec in _gpu_specs(gres):
            gpus = parse_gpus(spec)
            if gpus is None:
                continue
            entry["gpus"].append(
                {
                    "type": gpus.type,
                    "per_node": gpus.count,
                    "nodes": node_count,
                    "total": gpus.count * node_count,
                }
            )
    for entry in partitions.values():
        entry["cpus_per_node"].sort()
        entry["memory_mb_per_node"].sort()
    return list(partitions.values())


def _gpu_specs(gres: str) -> List[str]:
    """The GPU entries of a node's GRES, one per type (``gpu:a100:4(S:0-1)``)."""
    specs, depth, current = [], 0, []
    for char in gres:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char == "," and depth == 0:
            specs.append("".join(current))
            current = []
        else:
            current.append(char)
    specs.append("".join(current))
    return [spec.strip() for spec in specs if spec.strip().lower().startswith("gpu")]


def _probe(runner: Runner, args: List[str]) -> Tuple[int, str, str]:
    return runner.run(args, PROBE_TIMEOUT)


def _explanations() -> Dict[str, Any]:
    return {
        "pending_reasons": known_reasons(),
        "states": known_states(),
        "exit_codes": known_exit_codes(),
    }


def _base_report() -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "contract": {"kind": CAPABILITY_KIND, "version": ENVELOPES[CAPABILITY_KIND]},
        "package": package_info(),
        "available": False,
        "reason": None,
        "slurm": {"version": None, "json": {"supported": False, "basis": None}},
        "user": None,
        "partitions": [],
        "commands": [dict(command) for command in COMMANDS],
        "envelopes": [
            {"kind": kind, "version": version} for kind, version in ENVELOPES.items()
        ],
        "explanations": {},
        "catalog": {},
        "problems": [],
    }
    try:
        report["explanations"] = _explanations()
        report["catalog"] = catalog()
    except Exception:  # noqa: BLE001 - never raise from the report
        pass
    try:
        report["user"] = current_user()
    except Exception:  # noqa: BLE001 - never raise from the report
        pass
    return report


def capability_report(runner: Optional[Runner] = None) -> Dict[str, Any]:
    """``slurm_capability_report`` 1.1: what SLURM offers from this machine.

    The ``euler.capabilities`` entry of euler-slurm. Never raises: without
    SLURM, ``available`` is ``false`` and ``reason`` says why. A SLURM that is
    installed but does not answer is also reported under ``problems``. The
    explanation ``catalog`` (:func:`~euler_slurm.explain.catalog`) is there
    either way: it does not depend on SLURM.
    """
    try:
        report = _base_report()
    except Exception as exc:  # noqa: BLE001 - never raise from the report
        return {
            "contract": {
                "kind": CAPABILITY_KIND,
                "version": ENVELOPES[CAPABILITY_KIND],
            },
            "package": {"name": PACKAGE_NAME, "version": __version__},
            "available": False,
            "reason": f"internal error: {type(exc).__name__}: {exc}",
            "problems": [],
        }
    try:
        runner = runner if runner is not None else default_runner()
        _fill(report, runner)
    except SlurmUnavailable as exc:
        report["available"] = False
        report["reason"] = exc.reason
    except SlurmError as exc:
        report["available"] = False
        report["reason"] = str(exc)
        report["problems"].append({"code": "slurm_unreachable", "message": str(exc)})
    except Exception as exc:  # noqa: BLE001 - never raise from the report
        report["available"] = False
        report["reason"] = f"internal error: {type(exc).__name__}: {exc}"
    return report


def _fill(report: Dict[str, Any], runner: Runner) -> None:
    returncode, stdout, stderr = _probe(runner, ["sinfo", "--version"])
    if returncode != 0:
        raise SlurmCommandError(["sinfo", "--version"], returncode, stderr)
    version = parse_slurm_version(stdout)
    report["slurm"]["version"] = version[0] if version else None
    report["slurm"]["version_text"] = stdout.strip() or None

    json_info: Dict[str, Any] = {"supported": False, "basis": None}
    if version and version[1][:2] < JSON_MIN_VERSION:
        json_info.update(basis=f"SLURM {version[0]} predates --json (21.08)")
    else:
        try:
            _, help_out, help_err = _probe(runner, ["squeue", "--help"])
        except SlurmUnavailable as exc:
            json_info.update(basis=exc.reason)
        else:
            advertised = "--json" in (help_out + help_err)
            json_info.update(
                supported=advertised,
                basis="squeue --help lists --json"
                if advertised
                else "squeue --help has no --json",
            )
    report["slurm"]["json"] = json_info

    returncode, stdout, stderr = _probe(
        runner, ["sinfo", "--noheader", "--format", SINFO_FORMAT]
    )
    if returncode != 0:
        detail = first_line(stderr) or f"exit status {returncode}"
        raise SlurmCommandError(
            ["sinfo"], returncode, f"the controller did not answer: {detail}"
        )
    report["partitions"] = parse_sinfo_partitions(stdout)
    report["available"] = True


__all__ = [
    "CAPABILITY_KIND",
    "COMMANDS",
    "DIAGNOSIS_KIND",
    "ENVELOPES",
    "EXPLANATIONS_KIND",
    "JOBS_KIND",
    "JOB_KIND",
    "PACKAGE_NAME",
    "capability_report",
    "catalog_envelope",
    "diagnosis_envelope",
    "envelope",
    "error_envelope",
    "explanations_envelope",
    "job_dict",
    "job_envelope",
    "jobs_envelope",
    "package_info",
    "parse_sinfo_partitions",
    "parse_slurm_version",
]
