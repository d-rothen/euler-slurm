import pytest

from euler_slurm._parse import (
    INFINITE,
    NO_VAL,
    Gpus,
    expand_hostlist,
    expand_ranges,
    first_gpus,
    format_duration,
    is_unlimited,
    iso_from_epoch,
    iso_from_text,
    json_infinite,
    json_number,
    parse_duration,
    parse_gpus,
    parse_scontrol_records,
    parse_state,
    split_exit_code,
    text_or_none,
    tres_count,
    tres_list_to_string,
)
from tests.conftest import epoch, fixture_text


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("1-02:03:04", 93784),
        ("2-00:00:00", 172800),
        ("1-02:03", 93780),
        ("1-02", 93600),
        ("02:03:04", 7384),
        ("4:00:00", 14400),
        ("03:04", 184),
        ("0:00", 0),
        ("45", 2700),  # a bare number is minutes (time limits)
        ("00:00:01.250", 1),
        ("UNLIMITED", None),
        ("infinite", None),
        ("Partition_Limit", None),
        ("N/A", None),
        ("NOT_SET", None),
        ("INVALID", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_duration(text, seconds):
    assert parse_duration(text) == seconds


def test_parse_duration_bare_seconds_and_ints():
    assert parse_duration("45", bare="seconds") == 45
    assert parse_duration(240) == 14400
    assert parse_duration(2520, bare="seconds") == 2520
    assert parse_duration(True) is None


def test_unlimited():
    assert is_unlimited("UNLIMITED") and is_unlimited("infinite")
    assert not is_unlimited("4:00:00") and not is_unlimited(None)


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(0, "0:00"), (184, "3:04"), (7384, "2:03:04"), (93784, "1-02:03:04"), (None, "-")],
)
def test_format_duration(seconds, text):
    assert format_duration(seconds) == text


@pytest.mark.parametrize(
    ("raw", "state", "flags", "by"),
    [
        ("RUNNING", "RUNNING", (), None),
        ("R", "RUNNING", (), None),
        ("pd", "PENDING", (), None),
        ("CA", "CANCELLED", (), None),
        ("OOM", "OUT_OF_MEMORY", (), None),
        ("TO", "TIMEOUT", (), None),
        ("CANCELLED by 12345", "CANCELLED", (), "12345"),
        ("CANCELLED+", "CANCELLED", (), None),
        ("PENDING+REQUEUED", "PENDING", ("REQUEUED",), None),
        (["RUNNING"], "RUNNING", (), None),
        (["COMPLETED", "COMPLETING"], "COMPLETING", ("COMPLETED",), None),
        (
            ["PENDING", "REQUEUED", "LAUNCH_FAILED"],
            "REQUEUED",
            ("PENDING", "LAUNCH_FAILED"),
            None,
        ),
        (["CONFIGURING", "RUNNING"], "CONFIGURING", ("RUNNING",), None),
        ("WEIRD_STATE", "WEIRD_STATE", (), None),
        (None, "UNKNOWN", (), None),
        ([], "UNKNOWN", (), None),
    ],
)
def test_parse_state(raw, state, flags, by):
    parsed = parse_state(raw)
    assert (parsed.state, parsed.flags, parsed.cancelled_by) == (state, flags, by)


def test_json_numbers():
    assert json_number({"set": True, "infinite": False, "number": 8}) == 8
    assert json_number({"set": False, "infinite": False, "number": 0}) is None
    assert json_number({"set": True, "infinite": True, "number": 0}) is None
    assert json_number(240) == 240
    assert json_number(NO_VAL) is None
    assert json_number(INFINITE) is None
    assert json_number("12") == 12
    assert json_number(True) is None and json_number(None) is None
    assert json_infinite({"set": True, "infinite": True, "number": 0})
    assert json_infinite(INFINITE)
    assert not json_infinite(240) and not json_infinite(NO_VAL)


def test_times_are_iso_with_offset():
    assert iso_from_epoch(epoch("2026-11-20T14:31:10")) == "2026-11-20T14:31:10+00:00"
    assert iso_from_epoch({"set": True, "infinite": False, "number": 0}) is None
    assert iso_from_epoch(0) is None
    assert iso_from_text("2026-11-20T14:31:10") == "2026-11-20T14:31:10+00:00"
    assert iso_from_text("2026-11-20T15:31:10+01:00") == "2026-11-20T14:31:10+00:00"
    for blank in ("Unknown", "None", "N/A", "(null)", "", None):
        assert iso_from_text(blank) is None
    assert iso_from_text("Tomorr 14:30") is None


def test_text_or_none():
    for blank in ("", "N/A", "(null)", "None", "None assigned", "Unknown", "NOT_SET"):
        assert text_or_none(blank) is None
    assert text_or_none("  gpu.4h ") == "gpu.4h"


@pytest.mark.parametrize(
    ("spec", "multiplier", "gpus"),
    [
        (
            "cpu=8,mem=48G,node=1,billing=8,gres/gpu=1,gres/gpu:rtx_4090=1",
            1,
            Gpus(1, "rtx_4090"),
        ),
        ("cpu=8,mem=64G,node=1,billing=8,gres/gpu=1", 1, Gpus(1, None)),
        (
            "billing=8,cpu=8,gres/gpu:a100_80gb=4,gres/gpu=4,mem=48G",
            1,
            Gpus(4, "a100_80gb"),
        ),
        ("gres/gpu:rtx_4090:1", 1, Gpus(1, "rtx_4090")),
        ("gres/gpu:rtx_4090:4", 2, Gpus(8, "rtx_4090")),
        ("gres:gpu:2", 1, Gpus(2, None)),
        ("gres/gpu:1", 1, Gpus(1, None)),
        ("gpu:a100:4(S:0-1)", 1, Gpus(4, "a100")),
        ("gpu:rtx_4090:1(IDX:0)", 1, Gpus(1, "rtx_4090")),
        ("gpu(IDX:0-3)", 1, Gpus(4, None)),
        ("gpu:rtx_4090", 1, Gpus(1, "rtx_4090")),
        ("gpu", 1, Gpus(1, None)),
        ("gpu:3g.20gb:2", 1, Gpus(2, "3g.20gb")),
        ("gpu:a100:2,gpu:v100:2", 1, Gpus(4, None)),
        (["gpu:rtx_4090:1(IDX:0)", "gpu:rtx_4090:1(IDX:1)"], 1, Gpus(2, "rtx_4090")),
        ("cpu=8,mem=48G,gres/gpumem=40G,gres/shard=2", 1, None),
        ("gres/gpu=0", 1, None),
        ("N/A", 1, None),
        ("(null)", 1, None),
        ("tmpdisk:1000", 1, None),
        (None, 1, None),
    ],
)
def test_parse_gpus(spec, multiplier, gpus):
    assert parse_gpus(spec, multiplier=multiplier) == gpus


def test_first_gpus_fills_the_type_from_a_later_source():
    assert first_gpus(("gres/gpu=1", 1), (["gpu:a100_80gb:1(IDX:3)"], 1)) == Gpus(
        1, "a100_80gb"
    )
    assert first_gpus(("N/A", 1), ("gres/gpu:rtx_4090:4", 2)) == Gpus(8, "rtx_4090")
    assert first_gpus(("", 1), (None, 1)) is None
    assert str(Gpus(1, "rtx_4090")) == "1x rtx_4090" and str(Gpus(2)) == "2 GPUs"


def test_tres_helpers():
    items = [
        {"type": "cpu", "name": "", "id": 1, "count": 8},
        {"type": "gres", "name": "gpu", "id": 1001, "count": 1},
        {"type": "gres", "name": "gpu:rtx_4090", "id": 1002, "count": 1},
        {
            "type": "mem",
            "name": "",
            "count": {"set": True, "infinite": False, "number": 4},
        },
        "junk",
    ]
    tres = tres_list_to_string(items)
    assert tres == "cpu=8,gres/gpu=1,gres/gpu:rtx_4090=1,mem=4"
    assert tres_count(tres, "cpu") == 8 and tres_count(tres, "node") is None
    assert tres_list_to_string("nope") is None


def test_expand_ranges():
    assert expand_ranges("1-3,5") == [1, 2, 3, 5]
    assert expand_ranges("[1-9:2%4]") == [1, 3, 5, 7, 9]
    assert expand_ranges("7") == [7]


@pytest.mark.parametrize(
    ("text", "hosts"),
    [
        ("eu-g6-023", ["eu-g6-023"]),
        ("eu-g1-[001-003,007]", ["eu-g1-001", "eu-g1-002", "eu-g1-003", "eu-g1-007"]),
        ("a[1-2],b3", ["a1", "a2", "b3"]),
        ("n[1-2]-ib[3-4]", ["n1-ib3", "n1-ib4", "n2-ib3", "n2-ib4"]),
        ("None assigned", []),
        ("", []),
    ],
)
def test_expand_hostlist(text, hosts):
    assert expand_hostlist(text) == hosts


def test_scontrol_one_line_record_keeps_spaces_in_values():
    (record,) = parse_scontrol_records(fixture_text("scontrol_running.txt"))
    assert record["JobId"] == "91000003"
    assert record["WorkDir"] == "/cluster/home/alice/my proj"
    assert record["StdOut"] == "/cluster/home/alice/my proj/logs/slurm-91000003.out"
    assert record["AllocNode:Sid"] == "eu-login-11:2045"
    assert record["CPUs/Task"] == "8"
    assert record["Power"] == ""


def test_scontrol_multiline_and_detail_records():
    (record,) = parse_scontrol_records(fixture_text("scontrol_multiline.txt"))
    assert record["JobState"] == "RUNNING"
    assert record["StdOut"].endswith("slurm-91000003.out")
    assert record["Nodes"] == "eu-g6-023"
    records = parse_scontrol_records(fixture_text("scontrol_array.txt"))
    assert [r["JobId"] for r in records] == ["91000021", "91000020"]


@pytest.mark.parametrize(
    ("text", "pair"),
    [
        ("1:0", (1, None)),
        ("0:9", (0, 9)),
        ("0:0", (0, None)),
        ("0:125", (0, 125)),
        ("", (None, None)),
    ],
)
def test_split_exit_code(text, pair):
    assert split_exit_code(text) == pair
