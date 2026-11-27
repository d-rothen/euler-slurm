"""``euler-slurm``: SLURM jobs for people, and as JSON for tools.

Exit status: 0 success, 1 the command failed (unknown job, SLURM error),
2 wrong usage, 3 SLURM is not installed here.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Sequence

from ._parse import format_duration
from .diagnose import Diagnosis, diagnose
from .explain import (
    SLURM_OOM_SIGNAL,
    Explanation,
    catalog,
    explain,
    explain_job,
    signal_name,
)
from .jobs import Job, JobListing, SlurmClient
from .report import (
    DIAGNOSIS_KIND,
    JOB_KIND,
    JOBS_KIND,
    capability_report,
    catalog_envelope,
    diagnosis_envelope,
    error_envelope,
    explanations_envelope,
    job_envelope,
    jobs_envelope,
    package_info,
)
from .runner import DEFAULT_TIMEOUT, Runner, SlurmError, SlurmUnavailable

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_UNAVAILABLE = 3


def _print_json(document: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(document, indent=2, sort_keys=False) + "\n")


def _clip(text: Optional[str], width: int) -> str:
    text = text or "-"
    return text if len(text) <= width else text[: width - 1] + "…"


def _ended_how(job: Job) -> str:
    if job.signal == SLURM_OOM_SIGNAL:
        return "out of memory (0:125)"
    if job.signal:
        name = signal_name(job.signal) or f"signal {job.signal}"
        return f"killed by {name}"
    if job.exit_code:
        return f"exit code {job.exit_code}"
    return ""


def _where_or_why(job: Job, explanation: Explanation) -> str:
    if job.pending:
        reason = f" ({job.state_reason})" if job.state_reason else ""
        return f"{explanation.label}{reason}"
    if job.terminal:
        return explanation.label  # FAILED's label carries the exit code
    return job.nodes or "-"


def _time(job: Job) -> str:
    used = format_duration(job.time_used_s)
    limit = (
        "unlimited" if job.time_limit_unlimited else format_duration(job.time_limit_s)
    )
    return f"{used}/{limit}"


def render_jobs(listing: JobListing) -> str:
    """The human table of ``euler-slurm jobs``."""
    lines: List[str] = []
    if not listing.jobs:
        who = f" for {listing.user}" if listing.user else ""
        window = (
            f" or ended in the last {listing.since_hours} h"
            if listing.since_hours and listing.sources.get("sacct") not in ("skipped",)
            else ""
        )
        lines.append(f"No jobs{who} in the queue{window}.")
    else:
        rows = [
            (
                job.job_id,
                _clip(job.name, 24),
                job.state,
                _time(job),
                str(job.gpus) if job.gpus else "-",
                _where_or_why(job, explain_job(job)),
            )
            for job in listing.jobs
        ]
        header = ("JOB ID", "NAME", "STATE", "TIME/LIMIT", "GPUS", "WHERE / WHY")
        widths = [max(len(row[i]) for row in [header, *rows]) for i in range(5)]
        for row in [header, *rows]:
            cells = [cell.ljust(width) for cell, width in zip(row, widths)]
            lines.append("  ".join([*cells, row[5]]).rstrip())
    for problem in listing.problems:
        lines.append(f"note: {problem['message']}")
    if any(
        job.pending or job.terminal and job.state != "COMPLETED" for job in listing.jobs
    ):
        lines.append("")
        lines.append(
            "`euler-slurm job <id>` explains a job; `euler-slurm diagnose <id>` reads its logs."
        )
    return "\n".join(lines)


def _wrap(label: str, text: str, width: int = 88) -> List[str]:
    import textwrap

    prefix = f"  {label:<12}"
    return textwrap.wrap(
        text,
        width=width,
        initial_indent=prefix,
        subsequent_indent=" " * len(prefix),
        break_on_hyphens=False,
        break_long_words=False,  # paths and commands stay whole
    ) or [prefix.rstrip()]


def render_explanation(explanation: Explanation, *, title: Optional[str] = None) -> str:
    kind = {
        "pending_reason": "pending reason",
        "state": "job state",
        "exit_code": "exit code",
    }
    head = (
        title
        or f"{explanation.code}: {explanation.label} ({kind.get(explanation.kind, explanation.kind)})"
    )
    lines = [head]
    lines += _wrap("Meaning:", explanation.meaning)
    if explanation.detail:
        lines += _wrap("SLURM adds:", explanation.detail)
    lines += _wrap("What to do:", explanation.fix)
    if explanation.blocked:
        lines += _wrap("Note:", "It will not start unless this is fixed.")
    return "\n".join(lines)


def render_job(job: Job) -> str:
    """The human view of ``euler-slurm job``."""
    explanation = explain_job(job)
    title = f"Job {job.job_id}  {job.name or ''}".rstrip()
    if job.user:
        title += f"  (user {job.user})"
    lines = [title]
    state = job.state + (
        f" ({job.state_reason})" if job.pending and job.state_reason else ""
    )
    lines += _wrap("State:", f"{state}: {explanation.label}")
    lines += _wrap("Meaning:", explanation.meaning)
    if explanation.detail:
        lines += _wrap("SLURM adds:", explanation.detail)
    lines += _wrap("What to do:", explanation.fix)
    resources = [
        f"partition {job.partition or '-'}",
        f"{job.node_count or '-'} node(s) {job.nodes or ''}".rstrip(),
        f"{job.cpus or '-'} CPUs",
        f"GPUs {job.gpus or '-'}",
    ]
    lines += _wrap("Resources:", ", ".join(resources))
    lines += _wrap("Time:", _time(job).replace("/", " of ", 1))
    times = [f"submitted {job.submit_time or '-'}"]
    if job.start_time:
        times.append(
            ("started " if not job.pending else "expected start ") + job.start_time
        )
    if job.end_time:
        times.append(("ended " if job.terminal else "expected end ") + job.end_time)
    lines += _wrap("When:", ", ".join(times))
    if job.terminal and (job.exit_code is not None or job.signal):
        lines += _wrap(
            "Exit:",
            f"{job.exit_code}:{job.signal or 0}  {_ended_how(job) or 'success'}",
        )
    lines += _wrap("Work dir:", job.work_dir or "-")
    lines += _wrap("Stdout:", job.stdout_path or "- (unknown)")
    if job.stderr_path != job.stdout_path:
        lines += _wrap("Stderr:", job.stderr_path or "- (unknown)")
    return "\n".join(lines)


def render_catalog() -> str:
    """The human listing of ``euler-slurm explain --all``: code and label, by kind."""
    data = catalog()
    groups = (
        ("Pending reasons", data["pending_reasons"]),
        ("Job states", data["states"]),
        ("Exit codes", data["exit_codes"]),
    )
    lines: List[str] = []
    for title, entries in groups:
        if lines:
            lines.append("")
        lines.append(f"{title} ({len(entries)}):")
        width = min(max(len(entry["code"]) for entry in entries), 32)
        for entry in entries:
            lines.append(f"  {entry['code'].ljust(width)}  {entry['label']}".rstrip())
    lines.append("")
    lines.append("`euler-slurm explain <code>` explains one, with what to do.")
    return "\n".join(lines)


def render_diagnosis(diagnosis: Diagnosis) -> str:
    """The human view of ``euler-slurm diagnose``: the verdict, then the rest."""
    job = diagnosis.job
    if job is None:
        head = f"Job {diagnosis.job_id} - no SLURM record; read from its logs only"
    else:
        how = _ended_how(job) if job.terminal else ""
        name = f"({job.name}) " if job.name else ""
        head = f"Job {job.job_id} {name}- {job.state}" + (f", {how}" if how else "")
    lines = [head, ""]
    verdict = diagnosis.verdict
    lines += _wrap("Summary:", verdict.summary)
    lines += _wrap("Cause:", verdict.cause)
    if verdict.fix:
        lines += _wrap("Fix:", verdict.fix)
    if verdict.evidence:
        lines += _wrap("Log line:", verdict.evidence.get("line", ""))
        if verdict.evidence.get("path"):
            lines += _wrap("In:", verdict.evidence["path"])
    others = [f for f in diagnosis.findings if f.code != verdict.code]
    if others:
        lines.append("")
        lines.append("Also found:")
    for finding in others:
        lines.append("")
        lines += _wrap(f"[{finding.severity}]", finding.message)
        if finding.hint:
            lines += _wrap("Fix:", finding.hint)
        if finding.evidence:
            lines += _wrap("Log line:", finding.evidence.get("line", ""))
            if finding.evidence.get("path"):
                lines += _wrap("In:", finding.evidence["path"])
    read = [tail for tail in diagnosis.logs if tail.readable]
    if read:
        lines.append("")
        lines.append(
            "Read the last lines of: "
            + ", ".join(f"{t.path} ({len(t.lines)})" for t in read)
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _client(args: argparse.Namespace, runner: Optional[Runner]) -> SlurmClient:
    return SlurmClient(
        runner,
        use_json=False if args.no_slurm_json else None,
        timeout=args.timeout,
    )


def _fail(args: argparse.Namespace, kind: Optional[str], error: BaseException) -> int:
    if kind and getattr(args, "json", False):
        _print_json(error_envelope(kind, error))
    else:
        sys.stderr.write(f"euler-slurm: {error}\n")
    if isinstance(error, SlurmUnavailable):
        return EXIT_UNAVAILABLE
    if isinstance(error, ValueError):
        return EXIT_USAGE
    return EXIT_FAILED


def _cmd_jobs(args: argparse.Namespace, runner: Optional[Runner]) -> int:
    try:
        listing = _client(args, runner).list_jobs(
            args.user,
            include_finished=not args.active,
            since_hours=args.since_hours,
        )
    except (SlurmError, ValueError) as error:
        return _fail(args, JOBS_KIND, error)
    if args.json:
        _print_json(jobs_envelope(listing))
    else:
        print(render_jobs(listing))
    return EXIT_OK


def _cmd_job(args: argparse.Namespace, runner: Optional[Runner]) -> int:
    try:
        job = _client(args, runner).get_job(args.job_id)
    except (SlurmError, ValueError) as error:
        return _fail(args, JOB_KIND, error)
    if args.json:
        _print_json(job_envelope(job))
    else:
        print(render_job(job))
    return EXIT_OK


def _cmd_diagnose(args: argparse.Namespace, runner: Optional[Runner]) -> int:
    try:
        diagnosis = diagnose(
            args.job_id,
            args.lines,
            logs=args.log or (),
            client=_client(args, runner),
        )
    except (SlurmError, ValueError) as error:
        return _fail(args, DIAGNOSIS_KIND, error)
    if args.json:
        _print_json(diagnosis_envelope(diagnosis))
    else:
        print(render_diagnosis(diagnosis))
    return EXIT_OK


def _cmd_explain(args: argparse.Namespace, runner: Optional[Runner]) -> int:
    text = " ".join(args.text).strip()
    if args.all:
        if text:
            sys.stderr.write("euler-slurm: give codes or --all, not both\n")
            return EXIT_USAGE
        if args.json:
            _print_json(catalog_envelope())
        else:
            print(render_catalog())
        return EXIT_OK
    if not text:
        sys.stderr.write(
            "euler-slurm: give a pending reason, job state or exit code (or --all)\n"
        )
        return EXIT_USAGE
    explanation = explain(text)
    if args.json:
        _print_json(explanations_envelope([explanation]))
    else:
        print(render_explanation(explanation))
    return EXIT_OK


def _cmd_capabilities(args: argparse.Namespace, runner: Optional[Runner]) -> int:
    _print_json(capability_report(runner))
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="euler-slurm",
        description=(
            "SLURM jobs in plain words, and as versioned JSON (--json) for tools "
            "such as euler-view."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"euler-slurm {package_info()['version']}",
    )
    parser.add_argument(
        "--no-slurm-json",
        action="store_true",
        help="read SLURM's text output only (do not ask squeue/sacct/scontrol for --json)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"seconds per SLURM command (default {DEFAULT_TIMEOUT:g})",
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    commands.required = True

    jobs = commands.add_parser(
        "jobs", help="your active jobs and the ones that ended recently"
    )
    jobs.add_argument("--user", "-u", help="whose jobs (default: you)")
    jobs.add_argument(
        "--since-hours",
        type=int,
        default=24,
        help="ended jobs of the last N hours (default 24)",
    )
    jobs.add_argument(
        "--active", action="store_true", help="only jobs still in the queue"
    )
    jobs.add_argument(
        "--json", action="store_true", help="print the slurm_jobs envelope"
    )
    jobs.set_defaults(handler=_cmd_jobs)

    job = commands.add_parser(
        "job", help="one job in detail, its state or reason explained"
    )
    job.add_argument("job_id", help="e.g. 123, 123_4")
    job.add_argument("--json", action="store_true", help="print the slurm_job envelope")
    job.set_defaults(handler=_cmd_job)

    diag = commands.add_parser(
        "diagnose", help="why a job failed or waits (reads its logs)"
    )
    diag.add_argument("job_id", help="e.g. 123, 123_4")
    diag.add_argument(
        "--log",
        action="append",
        metavar="PATH",
        help=(
            "another log file to read, like its stdout/stderr (repeatable); "
            "enough on its own when SLURM no longer knows the job"
        ),
    )
    diag.add_argument(
        "--lines", type=int, default=200, help="log lines to read (default 200)"
    )
    diag.add_argument(
        "--json", action="store_true", help="print the slurm_diagnosis envelope"
    )
    diag.set_defaults(handler=_cmd_diagnose)

    expl = commands.add_parser(
        "explain", help="a pending reason, job state or exit code in plain words"
    )
    expl.add_argument(
        "text", nargs="*", help="e.g. Priority, QOSMaxGRESPerUser, TIMEOUT, PD, 137"
    )
    expl.add_argument(
        "--all", action="store_true", help="every code euler-slurm can explain"
    )
    expl.add_argument(
        "--json",
        action="store_true",
        help="print the slurm_explanations envelope (with --all: the whole catalog)",
    )
    expl.set_defaults(handler=_cmd_explain)

    caps = commands.add_parser("capabilities", help="what SLURM offers here (JSON)")
    caps.set_defaults(handler=_cmd_capabilities)
    return parser


def main(argv: Optional[Sequence[str]] = None, runner: Optional[Runner] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.handler(args, runner)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
