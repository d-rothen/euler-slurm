import json
import re
from pathlib import Path

import pytest

from euler_slurm import catalog as package_catalog
from euler_slurm._parse import LONG_STATES, PENDING_STATES, STATE_CODES, TERMINAL_STATES
from euler_slurm.explain import (
    EXIT_CATEGORIES,
    EXIT_CODES,
    REASON_CATEGORIES,
    REASONS,
    SEVERITIES,
    SIGNAL_NAMES,
    STATE_CATEGORIES,
    STATES,
    UNKNOWN_CATEGORY,
    catalog,
    explain,
    explain_exit,
    explain_job,
    explain_reason,
    explain_state,
    split_reason,
)
from euler_slurm.jobs import Job

README = Path(__file__).parent.parent / "README.md"

#: The reasons and states the newcomer docs promise to explain.
REQUIRED_REASONS = (
    "Priority",
    "Resources",
    "Dependency",
    "DependencyNeverSatisfied",
    "BeginTime",
    "JobHeldUser",
    "JobHeldAdmin",
    "QOSMaxJobsPerUserLimit",
    "QOSMaxGRESPerUser",
    "AssocGrpGRES",
    "AssocMaxJobsLimit",
    "ReqNodeNotAvail",
    "PartitionTimeLimit",
    "PartitionNodeLimit",
    "InvalidAccount",
    "InvalidQOS",
    "launch failed requeued held",
)
REQUIRED_STATES = (
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "FAILED",
    "CANCELLED",
)


def _assert_complete(explanation):
    assert explanation.known
    assert explanation.label and explanation.meaning and explanation.fix
    assert explanation.meaning.endswith((".", ")")), explanation.code
    assert explanation.fix.endswith((".", ")", "`")), explanation.code
    assert explanation.severity in SEVERITIES


@pytest.mark.parametrize("code", sorted(REASONS))
def test_every_reason_is_explained_with_a_fix(code):
    explanation = explain_reason(code)
    assert explanation.code == code and explanation.kind == "pending_reason"
    _assert_complete(explanation)
    assert explanation.blocked == (explanation.severity == "error")
    # Found case-insensitively too, and through the dispatcher.
    assert explain_reason(code.upper()).code == code
    assert explain(code).code == code


@pytest.mark.parametrize("state", sorted(STATES))
def test_every_state_is_explained_with_a_fix(state):
    explanation = explain_state(state)
    assert explanation.code == state and explanation.kind == "state"
    _assert_complete(explanation)


@pytest.mark.parametrize("code", sorted(EXIT_CODES))
def test_every_exit_code_is_explained_with_a_fix(code):
    explanation = explain_exit(code)
    assert explanation.code == str(code) and explanation.kind == "exit_code"
    _assert_complete(explanation)


def test_required_entries_exist():
    for code in REQUIRED_REASONS:
        assert code in REASONS, code
    for state in REQUIRED_STATES:
        assert state in STATES, state
    assert set(STATES) == set(LONG_STATES)  # every state SLURM can report


@pytest.mark.parametrize("short", sorted(STATE_CODES))
def test_short_state_codes(short):
    assert explain(short).code == STATE_CODES[short]


def test_blocked_reasons_say_so():
    for code in (
        "DependencyNeverSatisfied",
        "JobHeldUser",
        "PartitionTimeLimit",
        "InvalidAccount",
    ):
        assert explain_reason(code).blocked, code
    for code in (
        "Priority",
        "Resources",
        "Dependency",
        "BeginTime",
        "QOSMaxGRESPerUser",
    ):
        assert not explain_reason(code).blocked, code


@pytest.mark.parametrize(
    ("raw", "code", "detail"),
    [
        ("(Priority)", "Priority", None),
        (
            "ReqNodeNotAvail, UnavailableNodes:eu-g1-[001-004]",
            "ReqNodeNotAvail",
            "UnavailableNodes:eu-g1-[001-004]",
        ),
        (
            "ReqNodeNotAvail,_UnavailableNodes:eu-g1-[001-004]",
            "ReqNodeNotAvail",
            "UnavailableNodes:eu-g1-[001-004]",
        ),
        ("launch failed requeued held", "launch failed requeued held", None),
        ("launch_failed_requeued_held", "launch failed requeued held", None),
        (
            "(Nodes required for job are DOWN, DRAINED or reserved for jobs in higher priority partitions)",
            "Nodes required for job are DOWN, DRAINED or reserved for jobs in higher priority partitions",
            None,
        ),
    ],
)
def test_reason_forms(raw, code, detail):
    explanation = explain_reason(raw)
    assert explanation.known and explanation.code.lower() == code.lower()
    assert explanation.detail == detail
    assert split_reason(raw)[1] == detail


def test_unknown_reason_is_honest_not_guessed():
    explanation = explain_reason("QOSMaxFrobnicatePerUser")
    assert not explanation.known and explanation.severity == "info"
    assert "QOSMaxFrobnicatePerUser" in explanation.meaning
    assert "does not guess" in explanation.meaning
    assert (
        "GPU" not in explanation.meaning
        and "frobnicate"
        not in explanation.meaning.lower().replace("qosmaxfrobnicateperuser", "")
    )
    assert "man squeue" in explanation.fix


def test_unknown_state_is_honest():
    explanation = explain_state("TELEPORTED")
    assert not explanation.known and "does not guess" in explanation.meaning


@pytest.mark.parametrize(
    ("text", "code", "label_part"),
    [
        ("137", "137", "SIGKILL"),
        ("exit 143", "143", "SIGTERM"),
        ("exit code 139", "139", "Segmentation fault"),
        ("0:9", "137", "SIGKILL"),
        ("0:15", "143", "SIGTERM"),
        ("0:11", "139", "Segmentation fault"),
        ("0:125", "0:125", "too much memory"),
        ("131", "131", "SIGQUIT"),
        ("42", "42", "Exit code 42"),
    ],
)
def test_exit_codes(text, code, label_part):
    explanation = explain(text)
    assert explanation.kind == "exit_code" and explanation.code == code
    assert label_part in explanation.label


def test_exit_meanings_for_newcomers():
    assert "memory" in explain_exit(137).meaning
    assert "time limit" in explain_exit(143).meaning
    assert (
        "native code" in explain_exit(139).meaning.lower()
        or "Native code" in explain_exit(139).meaning
    )
    assert (
        not explain_exit(42).known
        and "SLURM gives this code no meaning" in explain_exit(42).meaning
    )
    assert not explain_exit(None).known


def test_failed_state_includes_the_exit_code():
    explanation = explain_state("FAILED", exit_code=0, signal=9)
    assert "exit code 137" in explanation.label
    assert "SIGKILL" in explanation.meaning and "sacct -j" in explanation.fix
    assert explain_state("F", exit_code=1).label == "Failed (exit code 1)"


def test_cancelled_by_names_the_user():
    explanation = explain("CANCELLED by 0")
    assert explanation.code == "CANCELLED" and "root (uid 0)" in explanation.meaning
    assert "uid 4290001" in explain_state("CANCELLED", cancelled_by="4290001").meaning


def test_for_job_fills_in_the_job_id():
    explanation = explain_reason("JobHeldUser").for_job("123_4")
    assert explanation.fix == "Release it: `scontrol release 123_4`."
    assert "<job_id>" not in explain_reason("Priority").for_job(9).fix


def test_explain_job_picks_reason_or_state():
    pending = Job(job_id="5", state="PENDING", state_reason="QOSMaxGRESPerUser")
    assert explain_job(pending).code == "QOSMaxGRESPerUser"
    waiting = Job(job_id="5", state="PENDING")
    assert explain_job(waiting).code == "PENDING"
    held = Job(
        job_id="6", state="REQUEUE_HOLD", state_reason="launch failed requeued held"
    )
    assert explain_job(held).code == "launch failed requeued held"
    assert "scontrol release 6" in explain_job(held).fix
    failed = Job(
        job_id="7", state="FAILED", exit_code=1, state_reason="NonZeroExitCode"
    )
    assert (
        explain_job(failed).code == "FAILED"
        and "exit code 1" in explain_job(failed).label
    )
    oom = Job(job_id="8", state="OUT_OF_MEMORY", exit_code=0, signal=125)
    assert explain_job(oom).code == "OUT_OF_MEMORY"


def test_to_dict_shape():
    data = explain("Priority").to_dict()
    assert set(data) == {
        "kind",
        "code",
        "known",
        "label",
        "meaning",
        "fix",
        "severity",
        "category",
        "blocked",
        "detail",
        "raw",
    }
    assert data["category"] == "queue"


def test_readme_reason_table_only_names_known_codes():
    """The README's "what does my pending reason mean" table stays true."""
    text = README.read_text()
    section = text.split("## What does my pending reason mean?", 1)[1].split(
        "\n## ", 1
    )[0]
    codes = re.findall(r"^\| `([^`]+)`", section, flags=re.MULTILINE)
    assert len(codes) >= 15
    for code in codes:
        assert explain(code).known, code


# ---------------------------------------------------------------------------
# Categories and the catalog
# ---------------------------------------------------------------------------


def test_every_entry_has_a_category_from_its_closed_set():
    for table, categories in (
        (REASONS, REASON_CATEGORIES),
        (STATES, STATE_CATEGORIES),
        (EXIT_CODES, EXIT_CATEGORIES),
    ):
        for code, entry in table.items():
            assert entry.category in categories, code
    for words in (
        *REASON_CATEGORIES.values(),
        *STATE_CATEGORIES.values(),
        *EXIT_CATEGORIES.values(),
    ):
        assert words and words[0].isupper() and not words.endswith(".")
    assert UNKNOWN_CATEGORY not in {
        *REASON_CATEGORIES,
        *STATE_CATEGORIES,
        *EXIT_CATEGORIES,
    }


@pytest.mark.parametrize(
    ("code", "category"),
    [
        ("None", "queue"),
        ("Priority", "queue"),
        ("Resources", "queue"),
        ("Dependency", "dependency"),
        ("DependencyNeverSatisfied", "dependency"),
        ("BeginTime", "schedule"),
        ("Reservation", "schedule"),
        ("Licenses", "schedule"),
        ("DeadLine", "schedule"),
        ("JobHeldUser", "hold"),
        ("JobHeldAdmin", "hold"),
        ("launch failed requeued held", "hold"),
        ("QOSMaxGRESPerUser", "limit"),
        ("AssocGrpGRES", "limit"),
        ("QOSMaxJobsPerUserLimit", "limit"),
        ("QOSMaxGRESPerJob", "request"),
        ("PartitionTimeLimit", "request"),
        ("InvalidAccount", "request"),
        ("BadConstraints", "request"),
        ("AssocGrpBillingMinutes", "budget"),
        ("ReqNodeNotAvail", "nodes"),
        ("PartitionDown", "nodes"),
        ("FrontEndDown", "nodes"),
        ("SystemFailure", "nodes"),
        ("Prolog", "startup"),
        ("Cleaning", "startup"),
        ("BurstBufferStageIn", "startup"),
        ("NonZeroExitCode", "requeued"),
        ("TimeLimit", "requeued"),
    ],
)
def test_reason_categories(code, category):
    assert explain_reason(code).category == category


def test_categories_agree_with_blocked_and_states():
    # A usage limit clears by itself; a request that does not fit never starts.
    for code, entry in REASONS.items():
        if entry.category == "limit":
            assert not explain_reason(code).blocked, code
        if entry.category in ("request", "budget", "hold"):
            assert explain_reason(code).blocked, code
    expected = {
        "waiting": {"PENDING", "CONFIGURING", "REQUEUED", "REQUEUE_FED"},
        "held": {"REQUEUE_HOLD", "SPECIAL_EXIT", "RESV_DEL_HOLD"},
        "running": {"RUNNING", "COMPLETING", "STAGE_OUT", "SIGNALING", "RESIZING"},
        "paused": {"SUSPENDED", "STOPPED"},
        "succeeded": {"COMPLETED"},
        "failed": {
            "FAILED",
            "TIMEOUT",
            "OUT_OF_MEMORY",
            "NODE_FAIL",
            "BOOT_FAIL",
            "DEADLINE",
        },
        "interrupted": {"PREEMPTED"},
        "cancelled": {"CANCELLED", "REVOKED"},
    }
    for category, states in expected.items():
        assert {s for s, e in STATES.items() if e.category == category} == states


def test_explanation_categories_for_derived_and_unknown_codes():
    assert explain_exit(0).category == "success"
    assert explain_exit(1).category == explain_exit(255).category == "error"
    assert explain_exit(127).category == "environment"
    assert explain_exit(0, 9).category == "signal"  # 137
    assert explain_exit(131).category == "signal"  # generic: 128 + SIGQUIT
    assert explain_exit(0, 125).category == "memory"
    assert explain_exit(42).category == "error"
    assert explain_exit(None).category == UNKNOWN_CATEGORY
    assert explain_reason("QOSMaxFrobnicatePerUser").category == UNKNOWN_CATEGORY
    assert explain_state("TELEPORTED").category == UNKNOWN_CATEGORY
    assert explain_state("FAILED", exit_code=1).category == "failed"
    assert explain("PD").category == "waiting"


def test_catalog_shape_and_sorting():
    data = catalog()
    assert package_catalog is catalog
    assert list(data) == [
        "pending_reasons",
        "states",
        "exit_codes",
        "signals",
        "categories",
    ]
    reasons = data["pending_reasons"]
    assert [r["code"] for r in reasons] == sorted(REASONS) and len(reasons) == 86
    assert list(reasons[0]) == [
        "code",
        "kind",
        "label",
        "meaning",
        "fix",
        "severity",
        "blocked",
        "category",
    ]
    states = data["states"]
    assert [s["code"] for s in states] == sorted(STATES)
    assert list(states[0]) == [
        "code",
        "kind",
        "short",
        "label",
        "meaning",
        "fix",
        "severity",
        "category",
        "terminal",
        "pending",
    ]
    exits = data["exit_codes"]
    assert [e["code"] for e in exits] == [str(c) for c in sorted(EXIT_CODES)] + [
        "0:125"
    ]
    assert list(exits[0]) == [
        "code",
        "kind",
        "label",
        "meaning",
        "fix",
        "severity",
        "category",
    ]
    assert data["signals"][8] == {"number": 9, "name": "SIGKILL"}
    assert len(data["signals"]) == len(SIGNAL_NAMES)
    assert data["categories"] == {
        "pending_reason": REASON_CATEGORIES,
        "state": STATE_CATEGORIES,
        "exit_code": EXIT_CATEGORIES,
    }
    assert data["categories"]["pending_reason"]["limit"] == (
        "At a usage limit; starts by itself"
    )
    json.dumps(data)


def test_catalog_entries_are_complete_and_match_explain():
    data = catalog()
    closed = {
        "pending_reason": REASON_CATEGORIES,
        "state": STATE_CATEGORIES,
        "exit_code": EXIT_CATEGORIES,
    }
    for group, kind in (
        ("pending_reasons", "pending_reason"),
        ("states", "state"),
        ("exit_codes", "exit_code"),
    ):
        for entry in data[group]:
            assert entry["kind"] == kind
            assert entry["label"] and entry["meaning"] and entry["fix"], entry
            assert entry["severity"] in SEVERITIES
            assert entry["category"] in closed[kind], entry["code"]
            explained = explain(entry["code"])
            assert (explained.code, explained.label, explained.fix) == (
                entry["code"],
                entry["label"],
                entry["fix"],
            )
    # The placeholder stays for tools to fill in.
    by_code = {r["code"]: r for r in data["pending_reasons"]}
    assert by_code["JobHeldUser"]["fix"] == "Release it: `scontrol release <job_id>`."
    assert by_code["JobHeldUser"]["blocked"] is True
    assert by_code["Priority"]["blocked"] is False
    oom = data["exit_codes"][-1]
    assert oom["category"] == "memory" and oom["label"] == explain_exit(0, 125).label


def test_catalog_short_codes_and_state_flags():
    states = {s["code"]: s for s in catalog()["states"]}
    assert states["PENDING"]["short"] == "PD"
    assert states["OUT_OF_MEMORY"]["short"] == "OOM"
    assert states["COMPLETED"]["short"] == "CD"
    for code, entry in states.items():
        assert entry["short"] is None or STATE_CODES[entry["short"]] == code
        assert entry["terminal"] == (code in TERMINAL_STATES)
        assert entry["pending"] == (code in PENDING_STATES)
    assert states["FAILED"]["terminal"] and not states["FAILED"]["pending"]
    assert states["REQUEUE_HOLD"]["pending"] and not states["RUNNING"]["pending"]
