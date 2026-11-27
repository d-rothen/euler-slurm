"""Why did my job fail (or why does it not start)? :func:`diagnose`.

It combines the job's record (state, pending reason, exit code, time used)
with the end of its stdout/stderr files, when they can be read here, into
**findings**: ``{code, severity, subject, message, hint}``, the same shape as
the findings of ``euler-suite doctor`` (euler-dataset-contract).

The log heuristics look for causes a newcomer would not recognise from the
state alone: GPU memory (``CUDA out of memory``), host memory (oom-kill),
NCCL / distributed timeouts, missing Python packages, full disks and quotas,
SLURM's own "cancelled due to time limit / preemption / node failure" lines,
and the last Python exception. Each finding quotes the log line it is based on.

Tools that know more log files than SLURM (a launcher's own log, say) pass
them as ``logs``; they are read and scanned like stdout/stderr. When SLURM
has forgotten the job, :func:`diagnose` with ``logs`` reads only those
(:func:`diagnose_logs`).

Of all findings, the **verdict** is the one thing to tell a newcomer: the job
in a few words, the most specific cause and its fix (:class:`Verdict`). Tools
show it as it is instead of ranking the findings themselves.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Pattern, Sequence, Tuple

from .explain import SEVERITIES, Explanation, explain_job
from .jobs import Job, JobNotFound, SlurmClient, validate_job_id
from .runner import Runner

#: Bytes read from the end of a log at most (a tqdm line can be huge).
MAX_TAIL_BYTES = 4 * 1024 * 1024
#: Characters of a log line quoted in a finding.
EVIDENCE_CHARS = 300

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_RANK_PREFIX = re.compile(r"^(?:\[(?:rank|default)\d+\]:\s?|\d+:\s)")


@dataclass(frozen=True)
class Finding:
    """One thing :func:`diagnose` found, with what to do about it."""

    code: str
    severity: str
    subject: str
    message: str
    hint: Optional[str] = None
    #: The log line it is based on: ``{path, line}``.
    evidence: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "code": self.code,
            "severity": self.severity,
            "subject": self.subject,
            "message": self.message,
        }
        if self.hint:
            result["hint"] = self.hint
        if self.evidence:
            result["evidence"] = dict(self.evidence)
        return result


@dataclass(frozen=True)
class Verdict:
    """The one thing to tell a newcomer about a job, chosen from the findings."""

    #: The job in a few words: "Failed (exit code 1)", "Waiting for its turn".
    summary: str
    #: Why, in plain words: the message of the most specific finding.
    cause: str
    #: What to do about it (that finding's hint).
    fix: Optional[str]
    #: The code of that finding (``log.cuda_oom``, ``state.timeout`` …).
    code: str
    severity: str
    #: The log line it is based on: ``{path, line}``.
    evidence: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "cause": self.cause,
            "fix": self.fix,
            "code": self.code,
            "severity": self.severity,
            "evidence": dict(self.evidence) if self.evidence else None,
        }


@dataclass
class LogTail:
    """What could be read of one log file."""

    #: ``stdout``, ``stderr``, ``stdout+stderr`` (one file for both), or
    #: ``log`` (a file the caller named).
    stream: str
    path: Optional[str]
    readable: bool = False
    lines: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stream": self.stream,
            "path": self.path,
            "readable": self.readable,
            "lines_read": len(self.lines),
            "error": self.error,
        }


@dataclass
class Diagnosis:
    """The job, the explanation of its state or reason, the findings and logs."""

    #: ``None`` when SLURM no longer knows the job and only logs were read.
    job: Optional[Job]
    #: Of the job's state or pending reason (``None`` without a job record).
    explanation: Optional[Explanation]
    findings: List[Finding]
    logs: List[LogTail]
    verdict: Verdict
    job_id: str

    def summary(self) -> Dict[str, Any]:
        counts = {severity: 0 for severity in SEVERITIES}
        for finding in self.findings:
            counts[finding.severity] = counts.get(finding.severity, 0) + 1
        worst = next((s for s in SEVERITIES if counts.get(s)), None)
        return {"counts": counts, "worst": worst}


# ---------------------------------------------------------------------------
# Reading logs
# ---------------------------------------------------------------------------


def clean_line(text: str) -> str:
    """A log line as a person sees it: last ``\\r`` segment, no ANSI, no rank tag."""
    if "\r" in text:
        segments = [segment for segment in text.split("\r") if segment.strip()]
        text = segments[-1] if segments else ""
    if "\x1b" in text:
        text = _ANSI.sub("", text)
    return _RANK_PREFIX.sub("", text.rstrip())


def read_tail(path: str, lines: int, *, max_bytes: int = MAX_TAIL_BYTES) -> List[str]:
    """The last ``lines`` lines of ``path`` (cleaned); raises ``OSError``."""
    if lines <= 0:
        return []
    with open(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        block = 64 * 1024
        data = b""
        position = size
        while position > 0 and data.count(b"\n") <= lines and len(data) < max_bytes:
            step = min(block, position)
            position -= step
            handle.seek(position)
            data = handle.read(step) + data
    text = data.decode("utf-8", errors="replace")
    # Split on "\n" only: str.splitlines() would also split tqdm's "\r" updates.
    raw_lines = text.split("\n")
    if raw_lines and raw_lines[-1] == "":
        raw_lines.pop()
    if position > 0 and raw_lines:
        raw_lines = raw_lines[1:]  # the first one is cut off
    cleaned = [clean_line(line) for line in raw_lines]
    return [line for line in cleaned if line.strip()][-lines:]


def _log_targets(
    job: Optional[Job], logs: Sequence[str] = ()
) -> List[Tuple[str, Optional[str]]]:
    """``(stream, path)`` to read: the job's stdout/stderr, then ``logs``.

    A file named twice (by ``os.path.normpath``) is read once.
    """
    targets: List[Tuple[str, Optional[str]]] = []
    if job is not None:
        out, err = job.stdout_path, job.stderr_path
        if (out and err and os.path.normpath(out) == os.path.normpath(err)) or (
            out is None and err is None
        ):
            targets.append(("stdout+stderr", out))
        else:
            targets += [("stdout", out), ("stderr", err)]
    seen = {os.path.normpath(path) for _, path in targets if path}
    for path in logs:
        if not path or os.path.normpath(path) in seen:
            continue
        seen.add(os.path.normpath(path))
        targets.append(("log", path))
    return targets


def _read(tail: LogTail, lines: int) -> LogTail:
    path = tail.path
    if not path:
        tail.error = "SLURM did not report the path of this file"
    elif path == "/dev/null":
        tail.error = "discarded (/dev/null)"
    elif tail.stream != "log" and re.search(r"%\d*[A-Za-z]", path):
        # SLURM's own path patterns (%x, %N …); a named log is taken as it is.
        tail.error = "the path contains a pattern euler-slurm cannot fill in"
    else:
        try:
            tail.lines = read_tail(path, lines)
            tail.readable = True
        except FileNotFoundError:
            tail.error = "the file does not exist (yet), or not on this machine"
        except PermissionError:
            tail.error = "permission denied"
        except IsADirectoryError:
            tail.error = "the path is a directory"
        except OSError as exc:
            tail.error = exc.strerror or str(exc)
    return tail


def read_logs(
    job: Optional[Job], lines: int, logs: Sequence[str] = ()
) -> List[LogTail]:
    """The tails of the job's stdout and stderr, then of ``logs``.

    Each says why it could not be read when it could not. ``job`` may be
    ``None`` (only ``logs`` are read); a file named twice is read once.
    """
    return [
        _read(LogTail(stream=stream, path=path), lines)
        for stream, path in _log_targets(job, logs)
    ]


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Rule:
    #: Finding code, ``log.<topic>``.
    topic: str
    severity: str
    #: Alternatives, most specific first: the quoted line is the last one
    #: matching the first alternative that matches at all.
    patterns: Tuple[Pattern[str], ...]
    message: str
    hint: str
    #: A job state whose explanation already says this (the finding merges into it).
    covered_by: Optional[str] = None
    #: Rules that make this one redundant when they match too.
    yields_to: Tuple[str, ...] = ()


def _p(*alternatives: str, flags: int = 0) -> Tuple[Pattern[str], ...]:
    return tuple(re.compile(alt, flags) for alt in alternatives)


RULES: Tuple[_Rule, ...] = (
    _Rule(
        "cuda_oom",
        "error",
        _p(
            r"CUDA out of memory",
            r"torch\.(?:cuda\.)?OutOfMemoryError",
            r"CUDA error: out of memory",
            r"CUBLAS_STATUS_ALLOC_FAILED",
            r"cudaErrorMemoryAllocation",
            r"RESOURCE_EXHAUSTED: Out of memory",
            r"HIP out of memory",
        ),
        "The job ran out of GPU memory (CUDA out of memory).",
        "Use less GPU memory: a smaller batch size (with gradient accumulation to "
        "keep the effective batch), mixed precision (bf16/fp16), gradient "
        "checkpointing, or a GPU with more memory. Check that nothing else runs on "
        "the same GPU (`nvidia-smi` on the node).",
    ),
    _Rule(
        "host_oom",
        "error",
        # Case-sensitive on purpose: "CUDA out of memory" is the GPU rule.
        _p(
            r"Detected \d+ oom[-_ ]kill event",
            r"\boom[-_]kill\b",
            r"out-of-memory handler",
            r"Out Of Memory",
            r"memory cgroup out of memory",
            r"Exceeded job memory limit",
            r"\bMemoryError\b",
            r"Unable to allocate [\d.]+ [KMGTP]i?B",
        ),
        "The job ran out of CPU memory (RAM) and a process was killed.",
        "Ask for more memory (`#SBATCH --mem=64G` or `--mem-per-cpu=`), or use "
        "less: fewer DataLoader workers, a smaller prefetch, memory-mapped data. "
        "`sacct -j <job_id> -o JobID,ReqMem,MaxRSS` shows what it asked for and used.",
        covered_by="OUT_OF_MEMORY",
    ),
    _Rule(
        "distributed_timeout",
        "error",
        _p(
            r"Watchdog caught collective operation timeout",
            r"NCCL.*(?:time[d ]?\s?out|Timeout)",
            r"ProcessGroupNCCL.*(?:time[d ]?\s?out|Timeout)",
            r"DistBackendError",
            r"DistStoreError",
            r"Timed out initializing process group",
            r"RendezvousTimeoutError",
            r"\bnccl(?:InternalError|SystemError|UnhandledCudaError|RemoteError)\b",
            r"NCCL error",
            flags=re.IGNORECASE,
        ),
        "A distributed (multi-GPU / multi-node) operation failed or timed out (NCCL).",
        "Usually one rank crashed or got stuck first and the others waited for it: "
        "look for the first error of any rank above this line. Make sure every "
        "rank runs the same number of steps (uneven data splits hang collectives) "
        "and that MASTER_ADDR/MASTER_PORT are reachable; `NCCL_DEBUG=INFO` shows "
        "more.",
    ),
    _Rule(
        "import_error",
        "error",
        _p(r"^(?:ModuleNotFoundError|ImportError): .+"),
        "Python could not import a module: the job's environment is missing a "
        "package, or has a version that does not fit.",
        "Run `euler-suite doctor` in the job's environment (the same `module load` "
        "/ `conda activate` / `source .venv/bin/activate` lines the job script runs) "
        "to see which packages are missing or disagree, then install them there.",
    ),
    _Rule(
        "disk_quota",
        "error",
        _p(
            r"Disk quota exceeded",
            r"\[Errno 122\]",
            r"No space left on device",
            r"\[Errno 28\]",
            r"\bEDQUOT\b",
            r"\bENOSPC\b",
        ),
        "A disk or quota is full: the job could not write a file.",
        "Free space or write elsewhere: `du -sh <dir>/* | sort -h` finds what is big "
        "(old checkpoints, caches such as ~/.cache), `quota -s` or your cluster's "
        "quota tool shows the limit; put large outputs on the scratch file system.",
    ),
    _Rule(
        "time_limit",
        "error",
        _p(r"DUE TO TIME LIMIT", r"due to time limit", r"Job step aborted.*time limit"),
        "SLURM stopped the job because it reached its time limit.",
        "Ask for more time (`#SBATCH --time=…`) or save checkpoints regularly and "
        "resume; `#SBATCH --signal=B:USR1@300` warns the script 5 minutes before "
        "the limit.",
        covered_by="TIMEOUT",
    ),
    _Rule(
        "preempted",
        "warning",
        _p(r"DUE TO PREEMPTION", r"due to preemption"),
        "SLURM stopped the job to make room for a higher-priority job (preemption).",
        "Resubmit, or use a partition/QOS that is not preemptible; checkpoint often "
        "and add `#SBATCH --requeue` so SLURM restarts it.",
        covered_by="PREEMPTED",
    ),
    _Rule(
        "node_failure",
        "error",
        _p(r"DUE TO NODE FAILURE", r"Node failure on \S+", r"NODE_FAIL"),
        "A node of the job failed (not caused by your code).",
        "Resubmit; if the same node fails again, add `--exclude=<node>`.",
        covered_by="NODE_FAIL",
    ),
    _Rule(
        "cancelled",
        "info",
        _p(r"\*\*\* (?:JOB|STEP) \S+ (?:ON \S+ )?CANCELLED AT \S+ \*\*\*"),
        "The job was cancelled (scancel) while it ran.",
        "If you did not cancel it, `sacct -j <job_id> -o JobID,State%30` names who did.",
        covered_by="CANCELLED",
        yields_to=("time_limit", "preempted", "node_failure"),
    ),
    _Rule(
        "killed",
        "warning",
        _p(r"^(?:.*: line \d+: +\d+ )?Killed\b"),
        "A process was killed with SIGKILL. On SLURM that is most often the memory "
        "limit (out-of-memory killer), or a cancellation or time limit.",
        "`sacct -j <job_id> -o JobID,State,ReqMem,MaxRSS` tells which; for memory, "
        "ask for more `--mem`.",
        yields_to=("host_oom", "time_limit", "cancelled", "preempted"),
    ),
)

_TRACEBACK = re.compile(r"Traceback \(most recent call last\):\s*$")
_EXCEPTION = re.compile(
    r"^([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Failure)|KeyboardInterrupt)"
    r"(?::\s?(.*))?$"
)
_MODULE = re.compile(r"No module named '([^']+)'")


def _clip(text: str) -> str:
    text = text.strip()
    return text if len(text) <= EVIDENCE_CHARS else text[: EVIDENCE_CHARS - 1] + "…"


def _last_exception(lines: Sequence[str]) -> Optional[str]:
    """The exception line of the last Python traceback in ``lines``."""
    start = None
    for index in range(len(lines) - 1, -1, -1):
        if _TRACEBACK.search(lines[index]):
            start = index
            break
    if start is None:
        return None
    for line in lines[start + 1 :]:
        if not line.startswith((" ", "\t")) and _EXCEPTION.match(line):
            return line
    return None


def _evidence(rule: _Rule, lines: Sequence[str]) -> Optional[str]:
    for pattern in rule.patterns:
        for line in reversed(lines):
            if pattern.search(line):
                return line
    return None


def scan_lines(
    lines: Sequence[str], path: Optional[str], subject: str
) -> Dict[str, Finding]:
    """Log heuristics over ``lines``: topic → finding (the last match is quoted)."""
    found: Dict[str, Finding] = {}
    for rule in RULES:
        match_line = _evidence(rule, lines)
        if match_line is None:
            continue
        message = rule.message
        if rule.topic == "import_error":
            module = _MODULE.search(match_line)
            if module:
                message = (
                    f"Python could not import `{module.group(1)}`: the job's "
                    "environment is missing it, or has a version that does not fit."
                )
        found[rule.topic] = Finding(
            code=f"log.{rule.topic}",
            severity=rule.severity,
            subject=subject,
            message=message,
            hint=rule.hint,
            evidence={"path": path or "", "line": _clip(match_line)},
        )
    for rule in RULES:
        if rule.topic in found and any(other in found for other in rule.yields_to):
            del found[rule.topic]
    specific = {
        "cuda_oom",
        "host_oom",
        "distributed_timeout",
        "import_error",
        "disk_quota",
    }
    if not specific & set(found):
        exception = _last_exception(lines)
        if exception:
            found["python_exception"] = Finding(
                code="log.python_exception",
                severity="error",
                subject=subject,
                message=f"The last Python error in the log: {_clip(exception)}",
                hint="The traceback above it in the log shows where it was raised.",
                evidence={"path": path or "", "line": _clip(exception)},
            )
    return found


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------

_COVERED = {rule.topic: rule.covered_by for rule in RULES if rule.covered_by}

#: States in which SLURM itself ended the job. The state says why; a traceback
#: or a ``Killed`` line in the log is usually its consequence, not its cause.
STOPPED_BY_SLURM = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "DEADLINE",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "TIMEOUT",
    }
)

#: Log findings that say only *that* something failed, not why.
_GENERIC_TOPICS = frozenset({"python_exception", "killed"})

#: Log finding code → rank, most specific first (the order of :data:`RULES`;
#: the last Python exception after all of them).
_TOPIC_RANK = {f"log.{rule.topic}": index for index, rule in enumerate(RULES)}
_TOPIC_RANK["log.python_exception"] = len(RULES)

#: The verdict's summary when SLURM no longer has a record of the job.
NO_RECORD_SUMMARY = "No SLURM record"


def _subject(job_id: str) -> str:
    return f"job {job_id}"


def _slug(code: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", code.lower()).strip("_") or "unknown"


def _for_job(finding: Finding, job_id: str) -> Finding:
    return replace(
        finding,
        message=finding.message.replace("<job_id>", job_id),
        hint=finding.hint.replace("<job_id>", job_id) if finding.hint else None,
    )


def _by_severity(findings: List[Finding]) -> List[Finding]:
    order = {severity: index for index, severity in enumerate(SEVERITIES)}
    return sorted(findings, key=lambda f: order.get(f.severity, len(order)))


def _scan_tails(tails: Sequence[LogTail], subject: str) -> Dict[str, Finding]:
    """The log findings of all readable tails: topic → the first file's finding."""
    topics: Dict[str, Finding] = {}
    for tail in tails:
        if tail.readable:
            for topic, finding in scan_lines(tail.lines, tail.path, subject).items():
                topics.setdefault(topic, finding)
    # Across files too: a "Killed" in stdout is the oom-kill reported in stderr.
    for rule in RULES:
        if rule.topic in topics and any(other in topics for other in rule.yields_to):
            del topics[rule.topic]
    return topics


def _unreadable(tails: Sequence[LogTail], subject: str) -> List[Finding]:
    """A ``log.unreadable`` finding for each log that should have been read."""
    findings: List[Finding] = []
    any_readable = any(tail.readable for tail in tails)
    for tail in tails:
        if tail.readable or tail.error == "discarded (/dev/null)":
            continue
        if tail.path is None and any_readable:
            continue  # e.g. no separate stderr path: nothing to report
        if tail.stream == "log":
            message = f"The log {tail.path} could not be read: {tail.error}."
            hint = (
                "Run `euler-slurm diagnose` on a machine that sees this file, as a "
                "user who may read it."
            )
        else:
            message = f"Its {tail.stream} could not be read: {tail.error}."
            hint = (
                f"Path: {tail.path}. Run `euler-slurm diagnose` on a machine that "
                "sees the job's files, as the job's user."
                if tail.path
                else "`scontrol show job <job_id>` shows StdOut/StdErr while SLURM "
                "still remembers the job."
            )
        findings.append(
            Finding(
                code="log.unreadable",
                severity="info",
                subject=subject,
                message=message,
                hint=hint,
            )
        )
    return findings


def choose_verdict(
    findings: Sequence[Finding],
    fallback: Finding,
    summary: str,
    state: Optional[str] = None,
) -> Verdict:
    """The one finding to tell a newcomer about, as a :class:`Verdict`.

    The most specific log finding of severity ``error``, then ``warning`` (in
    the order of :data:`RULES`, most specific first; the last Python exception
    after all of them; never ``log.unreadable``). When SLURM itself ended the
    job (:data:`STOPPED_BY_SLURM`), a Python exception or a ``Killed`` line
    does not count: the state says why. Without such a finding, ``fallback``
    (the job's state or pending reason).
    """
    candidates = [
        f for f in findings if f.code.startswith("log.") and f.code != "log.unreadable"
    ]
    if state in STOPPED_BY_SLURM:
        candidates = [f for f in candidates if f.code[4:] not in _GENERIC_TOPICS]
    chosen = fallback
    for severity in ("error", "warning"):
        ranked = sorted(
            (f for f in candidates if f.severity == severity),
            key=lambda f: _TOPIC_RANK.get(f.code, len(_TOPIC_RANK)),
        )
        if ranked:
            chosen = ranked[0]
            break
    return Verdict(
        summary=summary,
        cause=chosen.message,
        fix=chosen.hint,
        code=chosen.code,
        severity=chosen.severity,
        evidence=dict(chosen.evidence) if chosen.evidence else None,
    )


def diagnose_job(
    job: Job, log_tail_lines: int = 200, logs: Sequence[str] = ()
) -> Diagnosis:
    """The findings for a job record you already have (reads its log files).

    ``logs`` are further files to read and scan like its stdout/stderr (a
    file named twice is read once). A pending job's logs are not read.
    """
    subject = _subject(job.job_id)
    explanation = explain_job(job)
    prefix = "pending" if explanation.kind == "pending_reason" else "state"
    message = f"{explanation.label}: {explanation.meaning}"
    if explanation.detail:
        message += f" (SLURM adds: {explanation.detail})"
    primary = Finding(
        code=f"{prefix}.{_slug(explanation.code)}",
        severity=explanation.severity,
        subject=subject,
        message=message,
        hint=explanation.fix,
    )
    findings: List[Finding] = []
    if job.state == "RUNNING" and job.time_limit_s and job.time_used_s is not None:
        if job.time_used_s >= 0.9 * job.time_limit_s:
            findings.append(
                Finding(
                    code="job.near_time_limit",
                    severity="warning",
                    subject=subject,
                    message=(
                        f"It has used {job.time_used_s * 100 // job.time_limit_s}% of "
                        "its time limit; SLURM stops it when the limit is reached."
                    ),
                    hint=(
                        "Make sure it saves a checkpoint before the limit. A "
                        "running job's limit can only be raised by an administrator."
                    ),
                )
            )
    tails: List[LogTail] = []
    if not job.pending:
        tails = read_logs(job, log_tail_lines, logs)
        for topic, finding in _scan_tails(tails, subject).items():
            if _COVERED.get(topic) == job.state:
                # The state explanation says it already; keep the evidence.
                primary = replace(primary, evidence=finding.evidence)
                continue
            findings.append(finding)
        findings += _unreadable(tails, subject)
    primary = _for_job(primary, job.job_id)
    findings = _by_severity(
        [primary, *(_for_job(finding, job.job_id) for finding in findings)]
    )
    return Diagnosis(
        job=job,
        explanation=explanation,
        findings=findings,
        logs=tails,
        verdict=choose_verdict(findings, primary, explanation.label, job.state),
        job_id=job.job_id,
    )


def diagnose_logs(
    job_id: Any, logs: Sequence[str], log_tail_lines: int = 200
) -> Diagnosis:
    """Diagnose a job SLURM no longer knows from its log files alone.

    For a job that aged out of ``scontrol`` and has no accounting record. The
    first finding is ``job.not_found`` (``info``); ``job`` and ``explanation``
    are ``None``.
    """
    wanted = validate_job_id(job_id)
    subject = _subject(wanted)
    not_found = Finding(
        code="job.not_found",
        severity="info",
        subject=subject,
        message=(
            f"SLURM no longer has a record of job {wanted} (it ended too long ago, "
            "or accounting is off), so this diagnosis reads only its logs."
        ),
        hint=(
            f"`sacct -j {wanted}` on the cluster shows what the accounting "
            "database still has; cluster support knows how long it keeps jobs."
        ),
    )
    tails = read_logs(None, log_tail_lines, logs)
    found = list(_scan_tails(tails, subject).values()) + _unreadable(tails, subject)
    findings = [not_found] + _by_severity([_for_job(f, wanted) for f in found])
    return Diagnosis(
        job=None,
        explanation=None,
        findings=findings,
        logs=tails,
        verdict=choose_verdict(findings, not_found, NO_RECORD_SUMMARY),
        job_id=wanted,
    )


def diagnose(
    job_id: Any,
    log_tail_lines: int = 200,
    *,
    logs: Sequence[str] = (),
    runner: Optional[Runner] = None,
    client: Optional[SlurmClient] = None,
) -> Diagnosis:
    """Look up ``job_id`` and diagnose it (see the module docstring).

    ``logs`` are further log files to read (a launcher's own log, say). When
    SLURM no longer knows the job and ``logs`` are given, the diagnosis reads
    only them (:func:`diagnose_logs`). Raises
    :class:`~euler_slurm.jobs.JobNotFound` for an unknown job without
    ``logs``, and :class:`~euler_slurm.runner.SlurmUnavailable` without SLURM.
    """
    client = client if client is not None else SlurmClient(runner)
    try:
        job = client.get_job(job_id)
    except JobNotFound as error:
        if not logs:
            raise
        return diagnose_logs(error.job_id, logs, log_tail_lines)
    return diagnose_job(job, log_tail_lines, logs)


__all__ = [
    "Diagnosis",
    "Finding",
    "LogTail",
    "NO_RECORD_SUMMARY",
    "RULES",
    "STOPPED_BY_SLURM",
    "Verdict",
    "choose_verdict",
    "clean_line",
    "diagnose",
    "diagnose_job",
    "diagnose_logs",
    "read_logs",
    "read_tail",
    "scan_lines",
]
