"""SLURM jobs as typed records: :func:`list_jobs` and :func:`get_job`.

Where the facts come from:

* **active jobs** (pending, running, completing …): ``squeue``;
* **finished jobs**: ``sacct`` (the accounting database), when the cluster
  has accounting;
* **one job in detail** (log paths, exit code): ``scontrol show job``, which
  knows a job until a few minutes after it ended, then ``sacct``.

Each command is asked for JSON first (``--json``, Slurm 21.08 and later) and,
when that is not available, for delimiter-separated text (``squeue
--noheader --format``, ``sacct --noheader --parsable2 --format=``,
``scontrol --oneliner``). Both paths produce the same :class:`Job`; which one
answered is recorded in :attr:`JobListing.sources` and :attr:`Job.source`.

Filters (user, job ids) are applied again to JSON output, because Slurm
before 23.02 ignored them with ``--json``.
"""

from __future__ import annotations

import getpass
import json
import os
import re
import time
from dataclasses import dataclass, field, replace
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from ._parse import (
    PENDING_STATES,
    Gpus,
    epoch_from_iso,
    first_gpus,
    int_or_none,
    is_terminal,
    is_unlimited,
    iso_from_epoch,
    iso_from_text,
    iter_nonblank_lines,
    json_infinite,
    json_number,
    parse_duration,
    parse_scontrol_records,
    parse_state,
    split_exit_code,
    text_or_none,
    tres_count,
    tres_list_to_string,
)
from .runner import (
    DEFAULT_TIMEOUT,
    Runner,
    SlurmCommandError,
    SlurmError,
    SlurmUnavailable,
    default_runner,
    first_line,
)


class JobNotFound(SlurmError):
    """Neither the queue nor the accounting database knows the job."""

    def __init__(self, job_id: str, detail: Optional[str] = None) -> None:
        self.job_id = job_id
        message = f"SLURM does not know job {job_id}"
        if detail:
            message += f" ({detail})"
        super().__init__(message)


#: What a job id may look like: ``123``, ``123_4``, ``123_[1-3]``, ``123+1``.
JOB_ID_PATTERN = re.compile(r"^\d+(?:_(?:\d+|\[[\d,\-:%]+\]))?(?:\+\d+)?$")


def validate_job_id(job_id: Any) -> str:
    """``job_id`` as a string, or :class:`ValueError` when it is not one."""
    text = str(job_id).strip()
    if not JOB_ID_PATTERN.match(text):
        raise ValueError(
            f"{text!r} is not a SLURM job id (expected e.g. 123, 123_4 or 123_[1-3])"
        )
    return text


@dataclass(frozen=True)
class Job:
    """One SLURM job (or array task) as euler-slurm reports it.

    Times are ISO 8601 with the cluster's UTC offset. For a job that has not
    ended, ``start_time``/``end_time`` are SLURM's estimates.
    """

    #: The id SLURM shows and accepts: ``123``, ``123_4`` (array task),
    #: ``123_[1-3]`` (array tasks still pending), ``123+1`` (heterogeneous).
    job_id: str
    #: The unique numeric id (``JobIDRaw``; for an array task its own id).
    raw_job_id: Optional[int] = None
    array_job_id: Optional[int] = None
    array_task_id: Optional[int] = None
    #: The tasks of an array record that are still pending (``1-3``, ``1-9:2%4``).
    array_task_range: Optional[str] = None
    name: Optional[str] = None
    user: Optional[str] = None
    #: The normalized long state: ``PENDING``, ``RUNNING``, ``TIMEOUT`` …
    state: str = "UNKNOWN"
    state_flags: Tuple[str, ...] = ()
    #: Why it waits (pending) or ended (``None`` when SLURM says "None").
    state_reason: Optional[str] = None
    #: ``CANCELLED by <uid>``: the uid (sacct text only).
    cancelled_by: Optional[str] = None
    partition: Optional[str] = None
    #: The node list as SLURM prints it (compressed: ``eu-g1-[001-004]``).
    nodes: Optional[str] = None
    node_count: Optional[int] = None
    cpus: Optional[int] = None
    #: GPUs of the whole job (all nodes), from TRES or GRES.
    gpus: Optional[Gpus] = None
    time_used_s: Optional[int] = None
    #: ``None`` when unknown or unlimited (then ``time_limit_unlimited``).
    time_limit_s: Optional[int] = None
    time_limit_unlimited: bool = False
    submit_time: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    #: The batch script's return code (ended jobs only).
    exit_code: Optional[int] = None
    #: The signal that ended it (``ExitCode=0:9`` → 9), when one did.
    signal: Optional[int] = None
    work_dir: Optional[str] = None
    stdout_path: Optional[str] = None
    stderr_path: Optional[str] = None
    #: Which reader produced it: ``squeue-json``, ``squeue``, ``sacct-json``,
    #: ``sacct``, ``scontrol-json``, ``scontrol``.
    source: str = ""

    @property
    def terminal(self) -> bool:
        """The job has ended."""
        return is_terminal(self.state)

    @property
    def pending(self) -> bool:
        return self.state in PENDING_STATES

    @property
    def shell_exit_code(self) -> Optional[int]:
        """The exit status in the shell's convention: 128 + n for signal n."""
        if self.signal:
            return 128 + self.signal
        return self.exit_code

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "raw_job_id": self.raw_job_id,
            "array_job_id": self.array_job_id,
            "array_task_id": self.array_task_id,
            "array_task_range": self.array_task_range,
            "name": self.name,
            "user": self.user,
            "state": self.state,
            "state_flags": list(self.state_flags),
            "state_reason": self.state_reason,
            "cancelled_by": self.cancelled_by,
            "terminal": self.terminal,
            "partition": self.partition,
            "nodes": self.nodes,
            "node_count": self.node_count,
            "cpus": self.cpus,
            "gpus": self.gpus.to_dict() if self.gpus else None,
            "time_used_s": self.time_used_s,
            "time_limit_s": self.time_limit_s,
            "time_limit_unlimited": self.time_limit_unlimited,
            "submit_time": self.submit_time,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "exit_code": self.exit_code,
            "signal": self.signal,
            "shell_exit_code": self.shell_exit_code,
            "work_dir": self.work_dir,
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
            "source": self.source,
        }


@dataclass
class JobListing:
    """The result of :func:`list_jobs`: the jobs, and how they were read.

    Iterating over it (or ``len()``) behaves like the list of jobs.
    """

    jobs: List[Job] = field(default_factory=list)
    #: Per command: ``json``, ``text``, ``unavailable`` or ``skipped``.
    sources: Dict[str, str] = field(default_factory=dict)
    #: What makes the listing incomplete: ``{code, message}``.
    problems: List[Dict[str, str]] = field(default_factory=list)
    user: Optional[str] = None
    job_ids: Optional[List[str]] = None
    since_hours: Optional[int] = None

    def __iter__(self) -> Iterator[Job]:
        return iter(self.jobs)

    def __len__(self) -> int:
        return len(self.jobs)

    def __getitem__(self, index: int) -> Job:
        return self.jobs[index]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _display_id(
    raw_job_id: Optional[int],
    array_job_id: Optional[int],
    array_task_id: Optional[int],
    array_task_range: Optional[str],
    het_job_id: Optional[int] = None,
    het_offset: Optional[int] = None,
) -> str:
    if array_job_id and array_task_id is not None:
        return f"{array_job_id}_{array_task_id}"
    if array_job_id and array_task_range:
        return f"{array_job_id}_[{array_task_range}]"
    if het_job_id and het_offset is not None:
        return f"{het_job_id}+{het_offset}"
    return str(raw_job_id) if raw_job_id is not None else "?"


def _array_task(value: Any) -> Tuple[Optional[int], Optional[str]]:
    """An array task field: ``4`` → (4, None); ``1-3`` / ``[1-3%2]`` → (None, "1-3%2")."""
    text = text_or_none(value)
    if text is None:
        return None, None
    text = text.strip("[]")
    if text.isdigit():
        return int(text), None
    return None, text


def _reason(value: Any) -> Optional[str]:
    text = text_or_none(value)
    if text is None:
        return None
    text = re.sub(r"^\((.*)\)$", r"\1", text).strip()
    return None if text.upper() in ("NONE", "") else text


_PATTERN = re.compile(r"%(\d*)([%AaJjsuxtn])")


def expand_output_path(pattern: Any, job: "Job") -> Optional[str]:
    """An sbatch ``--output`` pattern (``slurm-%A_%a.out``) as the file path.

    Relative paths are resolved against the job's working directory. ``%N``
    (the node name) and other patterns euler-slurm cannot know stay in the
    path.
    """
    text = text_or_none(pattern)
    if text is None:
        return None

    def substitute(match: "re.Match[str]") -> str:
        width, code = match.group(1), match.group(2)
        if code == "%":
            return "%"
        values = {
            "A": job.array_job_id or job.raw_job_id,
            "a": job.array_task_id,
            "j": job.raw_job_id,
            "J": job.raw_job_id,
            "u": job.user,
            "x": job.name,
            "s": "batch",
            "t": 0,
            "n": 0,
        }
        value = values.get(code)
        if value is None:
            return match.group(0)
        if isinstance(value, int) and width:
            return f"{value:0{int(width)}d}"
        return str(value)

    path = _PATTERN.sub(substitute, text)
    if not path.startswith("/") and job.work_dir:
        path = os.path.join(job.work_dir, path)
    return path


def _finish(job: Job, stdout: Any = None, stderr: Any = None) -> Job:
    """Fill the log paths, and drop exit codes of jobs that have not ended."""
    changes: Dict[str, Any] = {}
    if stdout is not None:
        changes["stdout_path"] = expand_output_path(stdout, job)
    if stderr is not None:
        changes["stderr_path"] = expand_output_path(stderr, job)
    if not job.terminal:
        changes["exit_code"] = None
        changes["signal"] = None
    return replace(job, **changes) if changes else job


def _exit_from_json(value: Any) -> Tuple[Optional[int], Optional[int]]:
    """Slurm JSON ``exit_code`` as (return code, signal).

    A number (squeue/scontrol before v0.0.40) is the raw wait status;
    ``{status, return_code, signal: {signal_id | id}}`` the newer forms.
    """
    if value is None:
        return None, None
    if isinstance(value, Mapping):
        status = value.get("status")
        statuses = [
            str(s).upper() for s in (status if isinstance(status, list) else [status])
        ]
        if "PENDING" in statuses:
            return None, None
        signal_record = value.get("signal")
        signal = None
        if isinstance(signal_record, Mapping):
            signal = json_number(signal_record.get("id"))
            if signal is None:
                signal = json_number(signal_record.get("signal_id"))
        return json_number(value.get("return_code")), (signal or None)
    number = json_number(value)
    if number is None:
        return None, None
    signal = number & 0x7F
    return (number >> 8) & 0xFF, (signal or None)


def _minutes(value: Optional[int]) -> Optional[int]:
    """Slurm JSON time limits are minutes."""
    return value * 60 if value is not None else None


#: States of a job that has ended but is still in the queue.
ENDING_STATES = frozenset({"COMPLETING", "STAGE_OUT"})


def _elapsed(
    job_state: str, start: Optional[str], end: Optional[str], now: float
) -> Optional[int]:
    """Seconds run: 0 while pending, end - start once ended, now - start while running."""
    if job_state in PENDING_STATES:
        return 0
    started = epoch_from_iso(start)
    if started is None:
        return None
    if is_terminal(job_state) or job_state in ENDING_STATES:
        ended = epoch_from_iso(end)
        return max(0, int(ended - started)) if ended is not None else None
    return max(0, int(now - started))


# ---------------------------------------------------------------------------
# JSON readers (squeue / scontrol / sacct --json)
# ---------------------------------------------------------------------------


def job_from_squeue_json(
    raw: Mapping[str, Any], *, now: Optional[float] = None, source: str = "squeue-json"
) -> Job:
    """One entry of ``squeue --json`` / ``scontrol --json show job`` → :class:`Job`."""
    now = time.time() if now is None else now
    raw_id = json_number(raw.get("job_id"))
    array_job_id = json_number(raw.get("array_job_id")) or None
    array_task_id = json_number(raw.get("array_task_id")) if array_job_id else None
    array_range = text_or_none(raw.get("array_task_string")) if array_job_id else None
    if array_task_id is not None:
        array_range = None
    het_id = json_number(raw.get("het_job_id")) or None
    het_offset = json_number(raw.get("het_job_offset")) if het_id else None
    parsed = parse_state(raw.get("job_state"))
    node_count = json_number(raw.get("node_count"))
    start = iso_from_epoch(raw.get("start_time"))
    end = iso_from_epoch(raw.get("end_time"))
    limit = raw.get("time_limit")
    exit_code, signal = _exit_from_json(raw.get("exit_code"))
    tres_alloc = text_or_none(raw.get("tres_alloc_str"))
    job = Job(
        job_id=_display_id(
            raw_id, array_job_id, array_task_id, array_range, het_id, het_offset
        ),
        raw_job_id=raw_id,
        array_job_id=array_job_id,
        array_task_id=array_task_id,
        array_task_range=array_range,
        name=text_or_none(raw.get("name")),
        user=text_or_none(raw.get("user_name")),
        state=parsed.state,
        state_flags=parsed.flags,
        state_reason=_reason(raw.get("state_reason")),
        partition=text_or_none(raw.get("partition")),
        nodes=text_or_none(raw.get("nodes")),
        node_count=node_count,
        cpus=json_number(raw.get("cpus")) or tres_count(tres_alloc, "cpu"),
        gpus=first_gpus(
            (tres_alloc, 1),
            (raw.get("tres_req_str"), 1),
            (raw.get("tres_per_job"), 1),
            (raw.get("tres_per_node"), node_count or 1),
            (raw.get("gres_detail"), 1),
        ),
        time_limit_s=_minutes(json_number(limit)),
        time_limit_unlimited=json_infinite(limit),
        submit_time=iso_from_epoch(raw.get("submit_time")),
        start_time=start,
        end_time=end,
        exit_code=exit_code,
        signal=signal,
        work_dir=text_or_none(raw.get("current_working_directory")),
        source=source,
    )
    job = replace(job, time_used_s=_elapsed(job.state, start, end, now))
    return _finish(job, raw.get("standard_output"), raw.get("standard_error"))


def job_from_sacct_json(raw: Mapping[str, Any]) -> Job:
    """One entry of ``sacct --json`` → :class:`Job`."""
    raw_id = json_number(raw.get("job_id"))
    array = raw.get("array") if isinstance(raw.get("array"), Mapping) else {}
    array_job_id = json_number(array.get("job_id")) or None
    array_task_id = json_number(array.get("task_id")) if array_job_id else None
    array_range = None
    if array_job_id and array_task_id is None:
        array_range = text_or_none(array.get("task"))
    het = raw.get("het") if isinstance(raw.get("het"), Mapping) else {}
    het_id = json_number(het.get("job_id")) or None
    het_offset = json_number(het.get("job_offset")) if het_id else None
    state_record = raw.get("state") if isinstance(raw.get("state"), Mapping) else {}
    parsed = parse_state(state_record.get("current"))
    times = raw.get("time") if isinstance(raw.get("time"), Mapping) else {}
    tres = raw.get("tres") if isinstance(raw.get("tres"), Mapping) else {}
    allocated = tres_list_to_string(tres.get("allocated"))
    requested = tres_list_to_string(tres.get("requested"))
    required = raw.get("required") if isinstance(raw.get("required"), Mapping) else {}
    limit = times.get("limit")
    exit_code, signal = _exit_from_json(raw.get("exit_code"))
    job = Job(
        job_id=_display_id(
            raw_id, array_job_id, array_task_id, array_range, het_id, het_offset
        ),
        raw_job_id=raw_id,
        array_job_id=array_job_id,
        array_task_id=array_task_id,
        array_task_range=array_range,
        name=text_or_none(raw.get("name")),
        user=text_or_none(raw.get("user")),
        state=parsed.state,
        state_flags=parsed.flags,
        state_reason=_reason(state_record.get("reason")),
        partition=text_or_none(raw.get("partition")),
        nodes=text_or_none(raw.get("nodes")),
        node_count=json_number(raw.get("allocation_nodes")) or None,
        cpus=tres_count(allocated, "cpu") or json_number(required.get("CPUs")),
        gpus=first_gpus((allocated, 1), (requested, 1)),
        time_used_s=json_number(times.get("elapsed")),
        time_limit_s=_minutes(json_number(limit)),
        time_limit_unlimited=json_infinite(limit),
        submit_time=iso_from_epoch(times.get("submission")),
        start_time=iso_from_epoch(times.get("start")),
        end_time=iso_from_epoch(times.get("end")),
        exit_code=exit_code,
        signal=signal,
        work_dir=text_or_none(raw.get("working_directory")),
        source="sacct-json",
    )
    return _finish(
        job,
        raw.get("stdout_expanded") or raw.get("stdout"),
        raw.get("stderr_expanded") or raw.get("stderr"),
    )


def _json_errors(data: Mapping[str, Any]) -> List[str]:
    """The messages of a Slurm JSON document's ``errors`` list."""
    messages = []
    for error in data.get("errors") or []:
        if isinstance(error, Mapping):
            text = error.get("description") or error.get("error") or ""
            if error.get("error") and error.get("description"):
                text = f"{error['error']}: {error['description']}"
            if text:
                messages.append(str(text))
        elif error:
            messages.append(str(error))
    return messages


def _load_json(stdout: str) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(stdout)
    except ValueError:
        return None
    if isinstance(data, dict) and isinstance(data.get("jobs"), list):
        return data
    return None


# ---------------------------------------------------------------------------
# Text readers
# ---------------------------------------------------------------------------

#: ``squeue --format`` fields, in order. The name is last: a name containing
#: the delimiter keeps it (the extra fields are joined back into the name).
SQUEUE_FIELDS = (
    ("%i", "job_id"),
    ("%A", "raw_job_id"),
    ("%F", "array_job_id"),
    ("%K", "array_task"),
    ("%u", "user"),
    ("%T", "state"),
    ("%r", "reason"),
    ("%P", "partition"),
    ("%N", "nodes"),
    ("%D", "node_count"),
    ("%C", "cpus"),
    ("%b", "tres_per_node"),
    ("%M", "time_used"),
    ("%l", "time_limit"),
    ("%V", "submit_time"),
    ("%S", "start_time"),
    ("%e", "end_time"),
    ("%Z", "work_dir"),
    ("%j", "name"),
)
SQUEUE_FORMAT = "|".join(code for code, _ in SQUEUE_FIELDS)

#: ``squeue --Format`` for the allocated/requested TRES (the GPUs of jobs
#: submitted with ``--gpus``, which ``%b`` does not show).
SQUEUE_TRES_FORMAT = "JobID:40,tres-alloc:2000"

#: ``sacct --format=`` fields, in order (the name last, as above).
SACCT_FIELDS = (
    "JobID",
    "JobIDRaw",
    "User",
    "State",
    "Reason",
    "Partition",
    "NodeList",
    "NNodes",
    "AllocCPUS",
    "AllocTRES",
    "ReqTRES",
    "ElapsedRaw",
    "Timelimit",
    "Submit",
    "Start",
    "End",
    "ExitCode",
    "WorkDir",
    "JobName",
)
SACCT_FORMAT = ",".join(SACCT_FIELDS)


def _split_fields(line: str, count: int) -> Optional[List[str]]:
    fields = line.split("|")
    if len(fields) < count:
        return None
    if len(fields) > count:
        fields = fields[: count - 1] + ["|".join(fields[count - 1 :])]
    return fields


def parse_squeue_tres(output: str) -> Dict[str, str]:
    """``squeue --noheader --Format JobID,tres-alloc`` → job id (as printed) → TRES."""
    tres: Dict[str, str] = {}
    for line in iter_nonblank_lines(output):
        parts = line.split()
        if len(parts) >= 2 and "=" in parts[1]:
            tres[parts[0]] = parts[1]
    return tres


def job_from_squeue_line(
    line: str, tres: Optional[Mapping[str, str]] = None
) -> Optional[Job]:
    """One line of ``squeue --noheader --format SQUEUE_FORMAT``; ``None`` if malformed."""
    fields = _split_fields(line, len(SQUEUE_FIELDS))
    if fields is None:
        return None
    row = {key: value.strip() for (_, key), value in zip(SQUEUE_FIELDS, fields)}
    raw_id = int_or_none(row["raw_job_id"])
    if raw_id is None:
        return None
    task_id, task_range = _array_task(row["array_task"])
    array_job_id = (
        int_or_none(row["array_job_id"])
        if (task_id is not None or task_range)
        else None
    )
    parsed = parse_state(row["state"])
    node_count = int_or_none(row["node_count"])
    job = Job(
        job_id=text_or_none(row["job_id"]) or str(raw_id),
        raw_job_id=raw_id,
        array_job_id=array_job_id,
        array_task_id=task_id,
        array_task_range=task_range,
        name=text_or_none(row["name"]),
        user=text_or_none(row["user"]),
        state=parsed.state,
        state_flags=parsed.flags,
        state_reason=_reason(row["reason"]),
        partition=text_or_none(row["partition"]),
        nodes=text_or_none(row["nodes"]),
        node_count=node_count,
        cpus=int_or_none(row["cpus"]),
        gpus=first_gpus(
            ((tres or {}).get(str(raw_id)) or (tres or {}).get(row["job_id"]), 1),
            (row["tres_per_node"], node_count or 1),
        ),
        time_used_s=parse_duration(row["time_used"], bare="seconds"),
        time_limit_s=parse_duration(row["time_limit"]),
        time_limit_unlimited=is_unlimited(row["time_limit"]),
        submit_time=iso_from_text(row["submit_time"]),
        start_time=iso_from_text(row["start_time"]),
        end_time=iso_from_text(row["end_time"]),
        work_dir=text_or_none(row["work_dir"]),
        source="squeue",
    )
    return _finish(job)


def _array_from_id(job_id: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """``123_4`` → (123, 4, None); ``123_[1-3]`` → (123, None, "1-3")."""
    match = re.match(r"^(\d+)_(\d+|\[[^\]]+\])", job_id)
    if not match:
        return None, None, None
    task_id, task_range = _array_task(match.group(2))
    return int(match.group(1)), task_id, task_range


def job_from_sacct_line(line: str) -> Optional[Job]:
    """One line of ``sacct --noheader --parsable2 --format=SACCT_FORMAT``.

    Job steps (``123.batch``) and malformed lines are ``None``.
    """
    fields = _split_fields(line, len(SACCT_FIELDS))
    if fields is None:
        return None
    row = dict(zip(SACCT_FIELDS, (value.strip() for value in fields)))
    job_id = row["JobID"]
    if not job_id or "." in job_id:
        return None
    raw_id = int_or_none(row["JobIDRaw"]) or int_or_none(job_id)
    array_job_id, task_id, task_range = _array_from_id(job_id)
    parsed = parse_state(row["State"])
    exit_code, signal = split_exit_code(row["ExitCode"])
    elapsed = row["ElapsedRaw"]
    job = Job(
        job_id=job_id,
        raw_job_id=raw_id,
        array_job_id=array_job_id,
        array_task_id=task_id,
        array_task_range=task_range,
        name=text_or_none(row["JobName"]),
        user=text_or_none(row["User"]),
        state=parsed.state,
        state_flags=parsed.flags,
        state_reason=_reason(row["Reason"]),
        cancelled_by=parsed.cancelled_by,
        partition=text_or_none(row["Partition"]),
        nodes=text_or_none(row["NodeList"]),
        node_count=int_or_none(row["NNodes"]) or None,
        cpus=int_or_none(row["AllocCPUS"]) or None,
        gpus=first_gpus((row["AllocTRES"], 1), (row["ReqTRES"], 1)),
        time_used_s=int(elapsed)
        if elapsed.isdigit()
        else parse_duration(elapsed, bare="seconds"),
        time_limit_s=parse_duration(row["Timelimit"]),
        time_limit_unlimited=is_unlimited(row["Timelimit"]),
        submit_time=iso_from_text(row["Submit"]),
        start_time=iso_from_text(row["Start"]),
        end_time=iso_from_text(row["End"]),
        exit_code=exit_code,
        signal=signal,
        work_dir=text_or_none(row["WorkDir"]),
        source="sacct",
    )
    return _finish(job)


def job_from_scontrol(fields: Mapping[str, str], *, now: Optional[float] = None) -> Job:
    """One ``scontrol show job`` record (see ``parse_scontrol_records``)."""
    now = time.time() if now is None else now
    raw_id = int_or_none(fields.get("JobId"))
    array_job_id = int_or_none(fields.get("ArrayJobId"))
    task_id, task_range = _array_task(fields.get("ArrayTaskId"))
    if array_job_id is None:
        task_id, task_range = None, None
    het_id = int_or_none(fields.get("HetJobId"))
    het_offset = int_or_none(fields.get("HetJobOffset")) if het_id else None
    parsed = parse_state(fields.get("JobState"))
    user = text_or_none(fields.get("UserId"))
    if user:
        user = user.split("(", 1)[0] or user
    node_count = int_or_none(fields.get("NumNodes"))
    alloc_tres = text_or_none(fields.get("AllocTRES")) or text_or_none(
        fields.get("TRES")
    )
    exit_code, signal = split_exit_code(fields.get("ExitCode"))
    limit = fields.get("TimeLimit")
    start = iso_from_text(fields.get("StartTime"))
    end = iso_from_text(fields.get("EndTime"))
    run_time = parse_duration(fields.get("RunTime"), bare="seconds")
    reason = fields.get("Reason")
    if reason:
        reason = reason.replace(",_", ", ")
    job = Job(
        job_id=_display_id(
            raw_id, array_job_id, task_id, task_range, het_id, het_offset
        ),
        raw_job_id=raw_id,
        array_job_id=array_job_id,
        array_task_id=task_id,
        array_task_range=task_range,
        name=text_or_none(fields.get("JobName")),
        user=user,
        state=parsed.state,
        state_flags=parsed.flags,
        state_reason=_reason(reason),
        partition=text_or_none(fields.get("Partition")),
        nodes=text_or_none(fields.get("NodeList")),
        node_count=node_count,
        cpus=int_or_none(fields.get("NumCPUs")),
        gpus=first_gpus(
            (alloc_tres, 1),
            (fields.get("ReqTRES"), 1),
            (fields.get("TresPerJob"), 1),
            (fields.get("TresPerNode"), node_count or 1),
            (fields.get("Gres"), node_count or 1),
        ),
        time_used_s=run_time
        if run_time is not None
        else _elapsed(parsed.state, start, end, now),
        time_limit_s=parse_duration(limit),
        time_limit_unlimited=is_unlimited(limit),
        submit_time=iso_from_text(fields.get("SubmitTime")),
        start_time=start,
        end_time=end,
        exit_code=exit_code,
        signal=signal,
        work_dir=text_or_none(fields.get("WorkDir")),
        source="scontrol",
    )
    return _finish(job, fields.get("StdOut"), fields.get("StdErr"))


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------

_INVALID_JOB = re.compile(r"invalid job id", re.IGNORECASE)


def current_user() -> Optional[str]:
    """The user whose jobs ``list_jobs`` shows by default."""
    for name in ("SLURM_JOB_USER", "USER", "LOGNAME"):
        value = os.environ.get(name)
        if value:
            return value
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001 - no user database
        return None


def _matches(job: Job, wanted: Sequence[str]) -> bool:
    ids = {job.job_id, str(job.raw_job_id)}
    if job.array_job_id is not None:
        ids.add(str(job.array_job_id))
    return any(item in ids for item in wanted)


def _filtered(
    jobs: List[Job], user: Optional[str], job_ids: Optional[Sequence[str]]
) -> List[Job]:
    """Apply the filters again: JSON before Slurm 23.02 ignored them."""
    if user:
        jobs = [job for job in jobs if job.user in (None, user)]
    if job_ids:
        jobs = [job for job in jobs if _matches(job, job_ids)]
    return jobs


class SlurmClient:
    """Reads jobs through a :class:`~euler_slurm.runner.Runner`.

    ``use_json``: ``None`` asks for JSON first and falls back to text,
    ``True`` only JSON, ``False`` only text. ``clock`` is the current Unix time
    (the time a running job has used is computed from its start in JSON).
    """

    def __init__(
        self,
        runner: Optional[Runner] = None,
        *,
        use_json: Optional[bool] = None,
        timeout: float = DEFAULT_TIMEOUT,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.runner = runner if runner is not None else default_runner()
        self.use_json = use_json
        self.timeout = timeout
        self.clock = clock

    # -- plumbing ----------------------------------------------------------

    def _run(self, args: Sequence[str]) -> Tuple[int, str, str]:
        return self.runner.run(list(args), self.timeout)

    def _json(self, args: Sequence[str]) -> Optional[Dict[str, Any]]:
        """``args --json`` parsed, or ``None`` when JSON is not on offer.

        A JSON document that reports errors (the controller is down …) raises
        :class:`SlurmCommandError` instead of passing for "no jobs"; an unknown
        job id is an empty result.
        """
        if self.use_json is False:
            return None
        returncode, stdout, stderr = self._run([*args, "--json"])
        data = _load_json(stdout) if stdout.strip() else None
        if data is None:
            if self.use_json:
                raise SlurmCommandError(
                    [*args, "--json"], returncode, stderr or "no JSON output"
                )
            return None
        errors = _json_errors(data)
        if errors and any(_INVALID_JOB.search(error) for error in errors):
            return {**data, "jobs": []}
        if errors and (returncode != 0 or not data["jobs"]):
            raise SlurmCommandError([*args, "--json"], returncode, "; ".join(errors))
        return data

    # -- squeue --------------------------------------------------------------

    def _squeue(
        self, user: Optional[str], job_ids: Optional[Sequence[str]]
    ) -> Tuple[List[Job], str]:
        args = ["squeue"]
        if user:
            args += ["--user", user]
        if job_ids:
            args += [f"--jobs={','.join(job_ids)}"]
        data = self._json(args)
        if data is not None:
            now = self.clock()
            jobs = [
                job_from_squeue_json(raw, now=now)
                for raw in data["jobs"]
                if isinstance(raw, Mapping)
            ]
            return _filtered(jobs, user, job_ids), "json"
        returncode, stdout, stderr = self._run(
            [*args, "--noheader", "--format", SQUEUE_FORMAT]
        )
        if returncode != 0:
            if job_ids and _INVALID_JOB.search(stderr):
                return [], "text"
            raise SlurmCommandError(args, returncode, stderr)
        tres: Dict[str, str] = {}
        try:
            code, tres_out, _ = self._run(
                [*args, "--noheader", "--Format", SQUEUE_TRES_FORMAT]
            )
            if code == 0:
                tres = parse_squeue_tres(tres_out)
        except SlurmError:
            tres = {}
        jobs = [
            job_from_squeue_line(line, tres) for line in iter_nonblank_lines(stdout)
        ]
        return _filtered(
            [job for job in jobs if job is not None], user, job_ids
        ), "text"

    # -- sacct ---------------------------------------------------------------

    def _sacct(
        self,
        user: Optional[str],
        job_ids: Optional[Sequence[str]],
        since_hours: Optional[int],
    ) -> Tuple[List[Job], str]:
        args = ["sacct", "--allocations"]
        if user:
            args += ["--user", user]
        elif job_ids:
            # Someone else's job by id (allowed unless the cluster hides it).
            args += ["--allusers"]
        if job_ids:
            args += [f"--jobs={','.join(job_ids)}"]
        elif since_hours:
            args += [f"--starttime=now-{int(since_hours)}hours"]
        data = self._json(args)
        if data is not None:
            jobs = [
                job_from_sacct_json(raw)
                for raw in data["jobs"]
                if isinstance(raw, Mapping)
            ]
            return _filtered(jobs, user, job_ids), "json"
        returncode, stdout, stderr = self._run(
            [*args, "--noheader", "--parsable2", f"--format={SACCT_FORMAT}"]
        )
        if returncode != 0:
            raise SlurmCommandError(args, returncode, stderr)
        jobs = [job_from_sacct_line(line) for line in iter_nonblank_lines(stdout)]
        return _filtered(
            [job for job in jobs if job is not None], user, job_ids
        ), "text"

    # -- scontrol ------------------------------------------------------------

    def _scontrol(self, job_id: str) -> List[Job]:
        if self.use_json is not False:
            returncode, stdout, stderr = self._run(
                ["scontrol", "--json", "show", "job", job_id]
            )
            data = _load_json(stdout) if stdout.strip() else None
            errors = _json_errors(data) if data is not None else []
            if errors and any(_INVALID_JOB.search(error) for error in errors):
                return []
            if errors and (returncode != 0 or not data["jobs"]):
                raise SlurmCommandError(
                    ["scontrol", "--json", "show", "job", job_id],
                    returncode,
                    "; ".join(errors),
                )
            if data is not None:
                now = self.clock()
                return [
                    job_from_squeue_json(raw, now=now, source="scontrol-json")
                    for raw in data["jobs"]
                    if isinstance(raw, Mapping)
                ]
            if self.use_json:
                raise SlurmCommandError(["scontrol", "--json"], returncode, stderr)
        returncode, stdout, stderr = self._run(
            ["scontrol", "--oneliner", "show", "job", job_id]
        )
        if returncode != 0:
            if _INVALID_JOB.search(stderr):
                return []
            raise SlurmCommandError(
                ["scontrol", "show", "job", job_id], returncode, stderr
            )
        now = self.clock()
        return [
            job_from_scontrol(record, now=now)
            for record in parse_scontrol_records(stdout)
        ]

    # -- public --------------------------------------------------------------

    def list_jobs(
        self,
        user: Optional[str] = None,
        job_ids: Optional[Sequence[Any]] = None,
        *,
        include_finished: bool = True,
        since_hours: int = 24,
    ) -> JobListing:
        """Active jobs (squeue) and, unless ``include_finished=False``, jobs
        that ended in the last ``since_hours`` hours (sacct).

        With neither ``user`` nor ``job_ids``, the current user's jobs.
        """
        ids = [validate_job_id(item) for item in job_ids] if job_ids else None
        if user is None and not ids:
            user = current_user()
        listing = JobListing(
            user=user, job_ids=ids, since_hours=None if ids else since_hours
        )
        active, listing.sources["squeue"] = self._squeue(user, ids)
        listing.jobs.extend(active)
        if not include_finished:
            listing.sources["sacct"] = "skipped"
            return listing
        try:
            finished, listing.sources["sacct"] = self._sacct(user, ids, since_hours)
        except SlurmUnavailable as exc:
            listing.sources["sacct"] = "unavailable"
            listing.problems.append(
                {
                    "code": "sacct_unavailable",
                    "message": f"{exc}; finished jobs are not listed.",
                }
            )
            return listing
        except SlurmCommandError as exc:
            listing.sources["sacct"] = "unavailable"
            listing.problems.append(
                {
                    "code": "sacct_failed",
                    "message": (
                        "The accounting database did not answer, so finished jobs "
                        f"are not listed: {first_line(exc.stderr) or exc}"
                    ),
                }
            )
            return listing
        seen = {job.job_id for job in active} | {str(job.raw_job_id) for job in active}
        ended = [
            job
            for job in finished
            if job.terminal
            and job.job_id not in seen
            and str(job.raw_job_id) not in seen
        ]
        ended.sort(key=lambda job: job.end_time or "", reverse=True)
        listing.jobs.extend(ended)
        return listing

    def get_job(self, job_id: Any) -> Job:
        """One job: ``scontrol show job`` while SLURM remembers it, then sacct.

        Raises :class:`JobNotFound` when neither knows it.
        """
        wanted = validate_job_id(job_id)
        problem: Optional[SlurmError] = None
        try:
            found = self._pick(self._scontrol(wanted), wanted)
        except (SlurmUnavailable, SlurmCommandError) as exc:
            problem, found = exc, None
        if found is not None:
            return found
        detail = "not in the queue, and not in the accounting database"
        try:
            finished, _ = self._sacct(None, [wanted], None)
        except (SlurmUnavailable, SlurmCommandError) as exc:
            finished = []
            detail = (
                f"not in the queue; the accounting database could not be asked: {exc}"
            )
        found = self._pick(finished, wanted)
        if found is not None:
            return found
        if problem is not None:
            raise problem
        raise JobNotFound(wanted, detail)

    @staticmethod
    def _pick(jobs: Sequence[Job], wanted: str) -> Optional[Job]:
        for job in jobs:
            if job.job_id == wanted:
                return job
        for job in jobs:
            if str(job.raw_job_id) == wanted:
                return job
        return jobs[0] if jobs else None


def list_jobs(
    user: Optional[str] = None,
    job_ids: Optional[Sequence[Any]] = None,
    *,
    runner: Optional[Runner] = None,
    use_json: Optional[bool] = None,
    include_finished: bool = True,
    since_hours: int = 24,
) -> JobListing:
    """Jobs of ``user`` (default: you) or with ``job_ids``; see :class:`SlurmClient`."""
    client = SlurmClient(runner, use_json=use_json)
    return client.list_jobs(
        user, job_ids, include_finished=include_finished, since_hours=since_hours
    )


def get_job(
    job_id: Any, *, runner: Optional[Runner] = None, use_json: Optional[bool] = None
) -> Job:
    """One job by id; raises :class:`JobNotFound`."""
    return SlurmClient(runner, use_json=use_json).get_job(job_id)


__all__ = [
    "JOB_ID_PATTERN",
    "Gpus",
    "Job",
    "JobListing",
    "JobNotFound",
    "SACCT_FORMAT",
    "SQUEUE_FORMAT",
    "SlurmClient",
    "current_user",
    "expand_output_path",
    "get_job",
    "job_from_sacct_json",
    "job_from_sacct_line",
    "job_from_scontrol",
    "job_from_squeue_json",
    "job_from_squeue_line",
    "list_jobs",
    "validate_job_id",
]
