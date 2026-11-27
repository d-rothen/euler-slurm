"""Parsing primitives shared by the job readers and the explanations.

Everything here is pure: SLURM text or JSON values in, plain Python out. An
unknown value becomes ``None`` (or is kept as SLURM wrote it); nothing here
guesses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

# ---------------------------------------------------------------------------
# Blank values
# ---------------------------------------------------------------------------

#: What SLURM prints for "no value", upper-cased.
BLANK_VALUES = frozenset(
    {
        "",
        "N/A",
        "(NULL)",
        "NULL",
        "NONE",
        "(NONE)",
        "NONE ASSIGNED",
        "UNKNOWN",
        "NOT_SET",
        "INVALID",
    }
)


def text_or_none(value: Any) -> Optional[str]:
    """A stripped string, or ``None`` for SLURM's blank markers."""
    if value is None:
        return None
    text = str(value).strip()
    return None if text.upper() in BLANK_VALUES else text


def int_or_none(value: Any) -> Optional[int]:
    """The leading integer of ``value`` (``"8"``, ``"1-1"`` → 1), else ``None``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    match = re.match(r"^\s*(\d+)", str(value or ""))
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# Slurm JSON numbers
# ---------------------------------------------------------------------------

#: Sentinels older Slurm JSON (data_parser v0.0.37–0.0.39) prints for numbers.
NO_VAL = 0xFFFFFFFE
INFINITE = 0xFFFFFFFF
NO_VAL64 = 0xFFFFFFFFFFFFFFFE
INFINITE64 = 0xFFFFFFFFFFFFFFFF
_SENTINELS = {NO_VAL, INFINITE, NO_VAL64, INFINITE64, 0xFFFE, 0xFFFF}


def json_number(value: Any) -> Optional[int]:
    """A Slurm JSON number: plain, or ``{set, infinite, number}`` (v0.0.40+).

    Unset, infinite and sentinel values are ``None``.
    """
    if isinstance(value, Mapping):
        if value.get("set") is False or value.get("infinite") is True:
            return None
        return json_number(value.get("number"))
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = int(value)
        return None if number in _SENTINELS else number
    if isinstance(value, str) and value.strip().isdigit():
        return json_number(int(value.strip()))
    return None


def json_infinite(value: Any) -> bool:
    """Whether a Slurm JSON number says "infinite" (e.g. an UNLIMITED time limit)."""
    if isinstance(value, Mapping):
        return value.get("infinite") is True
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value
        in (
            INFINITE,
            INFINITE64,
        )
    )


# ---------------------------------------------------------------------------
# Job states
# ---------------------------------------------------------------------------

#: squeue/sacct short state codes → long names.
STATE_CODES: Dict[str, str] = {
    "BF": "BOOT_FAIL",
    "CA": "CANCELLED",
    "CD": "COMPLETED",
    "CF": "CONFIGURING",
    "CG": "COMPLETING",
    "DL": "DEADLINE",
    "F": "FAILED",
    "NF": "NODE_FAIL",
    "OOM": "OUT_OF_MEMORY",
    "PD": "PENDING",
    "PR": "PREEMPTED",
    "R": "RUNNING",
    "RD": "RESV_DEL_HOLD",
    "RF": "REQUEUE_FED",
    "RH": "REQUEUE_HOLD",
    "RQ": "REQUEUED",
    "RS": "RESIZING",
    "RV": "REVOKED",
    "S": "SUSPENDED",
    "SE": "SPECIAL_EXIT",
    "SI": "SIGNALING",
    "SO": "STAGE_OUT",
    "ST": "STOPPED",
    "TO": "TIMEOUT",
}

#: Every long state name SLURM reports for a job.
LONG_STATES = frozenset(STATE_CODES.values())

#: The base states (``JOB_STATE_BASE``); the rest are flags on top of one.
BASE_STATES = frozenset(
    {
        "PENDING",
        "RUNNING",
        "SUSPENDED",
        "COMPLETED",
        "CANCELLED",
        "FAILED",
        "TIMEOUT",
        "NODE_FAIL",
        "PREEMPTED",
        "BOOT_FAIL",
        "DEADLINE",
        "OUT_OF_MEMORY",
    }
)

#: Flags that squeue/scontrol print *instead of* the base state, in the order
#: Slurm's ``job_state_string()`` checks them. Slurm JSON lists the base state
#: and its flags; this order makes the JSON and the text paths agree.
STATE_FLAG_PRECEDENCE = (
    "COMPLETING",
    "STAGE_OUT",
    "CONFIGURING",
    "RESIZING",
    "REQUEUED",
    "REQUEUE_FED",
    "REQUEUE_HOLD",
    "SPECIAL_EXIT",
    "STOPPED",
    "REVOKED",
    "RESV_DEL_HOLD",
    "SIGNALING",
)

#: States of a job that has ended (it will not run again unless requeued).
TERMINAL_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "TIMEOUT",
    }
)

#: States of a job that waits to start.
PENDING_STATES = frozenset(
    {
        "PENDING",
        "CONFIGURING",
        "REQUEUED",
        "REQUEUE_HOLD",
        "REQUEUE_FED",
        "RESV_DEL_HOLD",
        "SPECIAL_EXIT",
    }
)

_CANCELLED_BY = re.compile(r"^(\S+)\s+by\s+(\S+)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedState:
    """A job state as euler-slurm reports it."""

    #: The long name (``PENDING``, ``CANCELLED`` …); what squeue's ``%T`` shows.
    state: str
    #: Further flags Slurm JSON listed (``REQUEUED``, ``LAUNCH_FAILED`` …).
    flags: Tuple[str, ...] = ()
    #: Who cancelled it (``CANCELLED by 12345`` → ``"12345"``, a uid).
    cancelled_by: Optional[str] = None


def normalize_state_name(value: str) -> str:
    """``"cd"`` → ``"COMPLETED"``; ``"CANCELLED+"`` → ``"CANCELLED"``.

    Unknown names are returned upper-cased, not guessed.
    """
    name = value.strip().upper().rstrip("+").strip()
    return STATE_CODES.get(name, name)


def parse_state(value: Any) -> ParsedState:
    """A state from any SLURM form.

    Text: long or short codes, ``CANCELLED by <uid>``, a trailing ``+``
    (sacct's truncation marker), ``BASE+FLAG``. JSON: a string, or a list of
    the base state and its flags (v0.0.40+).
    """
    if isinstance(value, (list, tuple)):
        names = [normalize_state_name(str(item)) for item in value if str(item).strip()]
        if not names:
            return ParsedState("UNKNOWN")
        base = next((name for name in names if name in BASE_STATES), names[0])
        flags = tuple(name for name in names if name != base)
        shown = next((flag for flag in STATE_FLAG_PRECEDENCE if flag in flags), base)
        if shown == base:
            return ParsedState(shown, flags=flags)
        # The flag is shown; the base state stays among the flags.
        return ParsedState(shown, flags=(base,) + tuple(f for f in flags if f != shown))
    text = text_or_none(value)
    if text is None:
        return ParsedState("UNKNOWN")
    cancelled_by = None
    match = _CANCELLED_BY.match(text)
    if match:
        text, cancelled_by = match.group(1), match.group(2)
    parts = [part for part in text.strip().upper().split("+") if part.strip()]
    if not parts:
        return ParsedState("UNKNOWN")
    names = [normalize_state_name(part) for part in parts]
    return ParsedState(names[0], flags=tuple(names[1:]), cancelled_by=cancelled_by)


def is_terminal(state: str) -> bool:
    """Whether a (long) state means the job has ended."""
    return state in TERMINAL_STATES


# ---------------------------------------------------------------------------
# Durations and times
# ---------------------------------------------------------------------------

_DURATION = re.compile(r"^(?:(\d+)-)?(\d+)(?::(\d+))?(?::(\d+)(?:\.\d+)?)?$")
_UNLIMITED = frozenset({"UNLIMITED", "INFINITE"})


def is_unlimited(value: Any) -> bool:
    """``UNLIMITED`` / ``infinite`` (time limits)."""
    return isinstance(value, str) and value.strip().upper() in _UNLIMITED


def parse_duration(value: Any, *, bare: str = "minutes") -> Optional[int]:
    """A SLURM duration in seconds; ``None`` when unknown or unlimited.

    Accepts ``D-HH:MM:SS``, ``D-HH:MM``, ``D-HH``, ``HH:MM:SS``, ``MM:SS``
    and a bare number, which is minutes (``bare="minutes"``, SLURM's
    convention for time limits) or seconds (``bare="seconds"``).
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if bare == "seconds" else value * 60
    text = text_or_none(value)
    if text is None or is_unlimited(text):
        return None
    match = _DURATION.match(text)
    if match is None:
        return None
    days, first, second, third = match.groups()
    numbers = [int(part) for part in (first, second, third) if part is not None]
    if days is not None:
        hours, minutes, seconds = (numbers + [0, 0, 0])[:3]
        return ((int(days) * 24 + hours) * 60 + minutes) * 60 + seconds
    if len(numbers) == 3:
        return (numbers[0] * 60 + numbers[1]) * 60 + numbers[2]
    if len(numbers) == 2:
        return numbers[0] * 60 + numbers[1]
    return numbers[0] if bare == "seconds" else numbers[0] * 60


def format_duration(seconds: Optional[int]) -> str:
    """Seconds as SLURM prints them: ``3:04``, ``2:03:04``, ``1-02:03:04``."""
    if seconds is None:
        return "-"
    seconds = max(0, int(seconds))
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    if days:
        return f"{days}-{hours:02d}:{minutes:02d}:{secs:02d}"
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def iso_from_epoch(value: Any) -> Optional[str]:
    """A Unix time (Slurm JSON) as ISO 8601 in the local time zone; 0 → ``None``."""
    number = json_number(value)
    if not number or number <= 0:
        return None
    try:
        moment = datetime.fromtimestamp(number, tz=timezone.utc).astimezone()
    except (OverflowError, OSError, ValueError):
        return None
    return moment.isoformat(timespec="seconds")


def epoch_from_iso(value: Optional[str]) -> Optional[float]:
    """The inverse of :func:`iso_from_epoch` / :func:`iso_from_text`."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def iso_from_text(value: Any) -> Optional[str]:
    """A SLURM timestamp (``2026-11-20T14:30:02``, local time) as ISO 8601
    with the local offset. ``Unknown``, ``None``, ``N/A`` → ``None``."""
    text = text_or_none(value)
    if text is None:
        return None
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment.astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# GPUs from GRES / TRES
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gpus:
    """GPUs of a job: how many, and of which GRES type (when SLURM names one)."""

    count: int
    type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"count": self.count, "type": self.type}

    def __str__(self) -> str:
        if self.type:
            return f"{self.count}x {self.type}"
        return f"{self.count} GPU{'' if self.count == 1 else 's'}"


_TRES_GPU = re.compile(r"^gres/gpu(?::([^=]+))?=(\d+)$", re.IGNORECASE)
_GRES_PREFIX = re.compile(r"^(?:gres[:/])?", re.IGNORECASE)
_PAREN_SUFFIX = re.compile(r"\(([^)]*)\)\s*$")


def _split_entries(spec: str) -> List[str]:
    """Split on commas outside parentheses (``gpu:a100:4(IDX:0,2-3),mps:100``)."""
    entries, depth, current = [], 0, []
    for char in spec:
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            entries.append("".join(current))
            current = []
        else:
            current.append(char)
    entries.append("".join(current))
    return [entry.strip() for entry in entries if entry.strip()]


def _index_count(detail: str) -> Optional[int]:
    """``IDX:0-3`` → 4 (GPU indexes in ``gres_detail``)."""
    match = re.search(r"IDX:([\d,\-]+)", detail)
    if not match:
        return None
    return len(expand_ranges(match.group(1)))


def parse_gpus(spec: Any, *, multiplier: int = 1) -> Optional[Gpus]:
    """GPUs named by a GRES or TRES string (or a list of them); ``None`` for none.

    Understands TRES (``cpu=8,mem=48G,gres/gpu=1,gres/gpu:rtx_4090=1``) and GRES
    (``gres/gpu:rtx_4090:1``, ``gres:gpu:2``, ``gpu:a100:4(S:0-1)``,
    ``gpu:rtx_4090:1(IDX:0)``, ``gpu``). An untyped total (``gres/gpu=N``) is
    the count; typed entries give the type, and the count when there is no
    total. Several types: the count is their sum and ``type`` is ``None``.
    ``multiplier`` scales per-node specs to the job (the node count).
    """
    if spec is None:
        return None
    if isinstance(spec, (list, tuple)):
        spec = ",".join(str(item) for item in spec if item)
    text = text_or_none(spec)
    if text is None:
        return None
    untyped: Optional[int] = None
    typed: Dict[str, int] = {}
    for raw in _split_entries(text):
        paren = _PAREN_SUFFIX.search(raw)
        entry = _PAREN_SUFFIX.sub("", raw).strip()
        tres = _TRES_GPU.match(entry)
        if tres:
            count = int(tres.group(2))
            if tres.group(1):
                typed[tres.group(1)] = typed.get(tres.group(1), 0) + count
            else:
                untyped = (untyped or 0) + count
            continue
        parts = _GRES_PREFIX.sub("", entry).split(":")
        if parts[0].lower() != "gpu" or "=" in entry:
            continue
        gpu_type: Optional[str] = None
        count_text: Optional[str] = None
        rest = parts[1:]
        if rest and rest[0] and not rest[0].isdigit():
            gpu_type, rest = rest[0], rest[1:]
        if rest:
            count_text = re.match(r"^\d*", rest[0]).group(0) or None
        if count_text is not None:
            count = int(count_text)
        else:
            count = (_index_count(paren.group(1)) if paren else None) or 1
        if gpu_type:
            typed[gpu_type] = typed.get(gpu_type, 0) + count
        else:
            untyped = (untyped or 0) + count
    total = untyped if untyped is not None else (sum(typed.values()) if typed else None)
    if not total:
        return None
    gpu_type = next(iter(typed)) if len(typed) == 1 else None
    return Gpus(count=total * max(1, multiplier), type=gpu_type)


def first_gpus(*candidates: Tuple[Any, int]) -> Optional[Gpus]:
    """The GPUs of the first ``(spec, multiplier)`` that names any, with the
    type filled from a later one when the first is untyped."""
    found: Optional[Gpus] = None
    for spec, multiplier in candidates:
        gpus = parse_gpus(spec, multiplier=multiplier)
        if gpus is None:
            continue
        if found is None:
            found = gpus
        if found.type is None and gpus.type is not None:
            found = Gpus(count=found.count, type=gpus.type)
        if found.type is not None:
            break
    return found


def tres_list_to_string(items: Any) -> Optional[str]:
    """sacct JSON TRES (``[{type, name, count}]``) as a TRES string."""
    if not isinstance(items, list):
        return None
    parts = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        kind = str(item.get("type") or "").strip()
        name = str(item.get("name") or "").strip()
        count = json_number(item.get("count"))
        if not kind or count is None:
            continue
        parts.append(f"{kind}/{name}={count}" if name else f"{kind}={count}")
    return ",".join(parts) or None


def tres_count(tres: Optional[str], key: str) -> Optional[int]:
    """The count of one TRES (``cpu`` in ``cpu=8,mem=48G``)."""
    for entry in (tres or "").split(","):
        name, _, value = entry.partition("=")
        if name.strip() == key:
            return int_or_none(value)
    return None


# ---------------------------------------------------------------------------
# Ranges and host lists
# ---------------------------------------------------------------------------


def expand_ranges(text: str) -> List[int]:
    """``1-3,5,7-11:2%4`` → ``[1, 2, 3, 5, 7, 9, 11]`` (array task ranges;
    the ``%N`` throttle is dropped)."""
    values: List[int] = []
    body = text.strip().strip("[]").split("%", 1)[0]
    for part in body.split(","):
        part = part.strip()
        if not part:
            continue
        step = 1
        if ":" in part:
            part, step_text = part.split(":", 1)
            step = max(1, int(step_text))
        if "-" in part:
            low, high = part.split("-", 1)
            values.extend(range(int(low), int(high) + 1, step))
        else:
            values.append(int(part))
    return values


def _split_top_level(text: str) -> List[str]:
    items, depth, current = [], 0, []
    for char in text:
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        if char == "," and depth == 0:
            items.append("".join(current))
            current = []
        else:
            current.append(char)
    items.append("".join(current))
    return [item.strip() for item in items if item.strip()]


def _expand_host(pattern: str) -> List[str]:
    start = pattern.find("[")
    if start < 0:
        return [pattern]
    end = pattern.find("]", start)
    if end < 0:
        return [pattern]
    prefix, body, rest = pattern[:start], pattern[start + 1 : end], pattern[end + 1 :]
    names: List[str] = []
    for part in body.split(","):
        part = part.strip()
        if "-" in part:
            low, high = part.split("-", 1)
            width = len(low)
            names.extend(
                f"{prefix}{n:0{width}d}" for n in range(int(low), int(high) + 1)
            )
        elif part:
            names.append(f"{prefix}{part}")
    return [name + suffix for name in names for suffix in _expand_host(rest)]


def expand_hostlist(text: Any) -> List[str]:
    """A compressed SLURM node list as host names, like ``scontrol show hostnames``.

    ``eu-g1-[001-003,007]`` → ``eu-g1-001 … eu-g1-003, eu-g1-007``.
    """
    value = text_or_none(text)
    if value is None:
        return []
    hosts: List[str] = []
    for item in _split_top_level(value):
        hosts.extend(_expand_host(item))
    return hosts


# ---------------------------------------------------------------------------
# scontrol key=value records
# ---------------------------------------------------------------------------

_KEY_TOKEN = re.compile(r"^([A-Za-z][A-Za-z0-9_/:]*)=(.*)$")


def parse_scontrol_records(output: str) -> List[Dict[str, str]]:
    """``scontrol show job`` output (one-line ``-o`` or multi-line) as records.

    A ``JobId=`` starts a record. A token without ``key=`` continues the
    previous value (``WorkDir=/a b`` → ``"/a b"``). The first occurrence of a
    key wins (``-d`` repeats some per node).
    """
    records: List[Dict[str, str]] = []
    current: Optional[Dict[str, str]] = None
    last_key: Optional[str] = None
    for token in output.split():
        match = _KEY_TOKEN.match(token)
        if match:
            key, value = match.group(1), match.group(2)
            if key == "JobId":
                current = {}
                records.append(current)
            if current is None:
                last_key = None
                continue
            if key in current:
                last_key = None
                continue
            current[key] = value
            last_key = key
        elif current is not None and last_key is not None:
            current[last_key] = f"{current[last_key]} {token}"
    return records


def split_exit_code(value: Any) -> Tuple[Optional[int], Optional[int]]:
    """``"1:0"`` → ``(1, None)``; ``"0:9"`` → ``(0, 9)`` (return code, signal)."""
    text = text_or_none(value)
    if text is None:
        return None, None
    match = re.match(r"^\s*(\d+)(?::(\d+))?\s*$", text)
    if not match:
        return None, None
    signal = int(match.group(2)) if match.group(2) else 0
    return int(match.group(1)), (signal or None)


def iter_nonblank_lines(text: str) -> Iterable[str]:
    for line in text.splitlines():
        if line.strip():
            yield line.rstrip("\n")
