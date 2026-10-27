from __future__ import annotations

import json
from typing import get_args

import pytest

from babymon import factlog
from babymon.domain import (
    AlertKind,
    AudioSample,
    CameraStatus,
    DeliveryResult,
    Fact,
    MotionSample,
    SetControls,
    Tick,
    VisionFailed,
    VisionReason,
)
from tests.conftest import at, observation, seen

EVERY_FACT: list[Fact] = [
    Tick(at(1)),
    MotionSample(at(1), 0.07),
    AudioSample(at(1), 3.5, 0.62),
    seen(2, observation(head="left", eyes="open")),
    VisionFailed(at(3), VisionReason.CONFIRM, "timeout"),
    CameraStatus(at(4), up=False, detail="stalled"),
    SetControls(at(5), armed=None, nap_mode=True),
    DeliveryResult(at(6), AlertKind.CRYING, delivered=False),
]


def test_the_fixture_covers_every_fact_type():
    assert {type(f) for f in EVERY_FACT} == set(get_args(Fact.__value__))


@pytest.mark.parametrize("fact", EVERY_FACT, ids=lambda f: type(f).__name__)
def test_a_fact_survives_a_line(fact):
    line = factlog.encode(fact)
    assert "\n" not in line
    assert factlog.decode(line) == fact


def test_the_line_is_a_flat_object_with_a_type_name():
    line = json.loads(factlog.encode(SetControls(at(5), armed=None, nap_mode=True)))
    assert line == {
        "type": "set_controls",
        "at": "2026-10-27T02:00:05Z",
        "armed": None,
        "nap_mode": True,
    }


@pytest.mark.parametrize(
    "line",
    [
        "[]",
        '{"type": "nope", "at": "2026-10-27T02:00:00Z"}',
        '{"type": "tick"}',
        '{"type": "tick", "at": "2026-10-27T02:00:00"}',
        '{"type": "motion", "at": "2026-10-27T02:00:00Z", "level": "high"}',
        "not json",
    ],
)
def test_a_bad_line_is_a_value_error(line):
    with pytest.raises(ValueError):
        factlog.decode(line)


def test_read_skips_blank_lines_and_names_the_bad_line(tmp_path):
    path = tmp_path / "facts.jsonl"
    path.write_text(factlog.encode(Tick(at(1))) + "\n\n" + factlog.encode(Tick(at(2))) + "\n")
    assert list(factlog.read(path)) == [Tick(at(1)), Tick(at(2))]

    path.write_text(factlog.encode(Tick(at(1))) + "\n{oops\n")
    with pytest.raises(ValueError, match=r"facts\.jsonl:2"):
        list(factlog.read(path))
