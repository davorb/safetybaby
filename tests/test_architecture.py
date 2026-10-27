"""Structural invariants the build enforces."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_pure_core_imports_no_io_library():
    lint = Path(sys.executable).parent / "lint-imports"
    result = subprocess.run([str(lint)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_alert_kind_has_exactly_one_rule_and_one_policy_row():
    from babymon.brain import ALERT_POLICY
    from babymon.domain import AlertKind
    from babymon.rules import RULES

    assert sorted(r.kind for r in RULES) == sorted(AlertKind)
    assert set(ALERT_POLICY) == set(AlertKind)


def test_situation_round_trips_through_json():
    from babymon.domain import Situation
    from tests.conftest import Run, observation, seen, start

    run = Run(start()).feed(seen(5, observation())).seconds(1, 10)
    s = run.situation()
    assert Situation.model_validate(s.model_dump(mode="json")) == s
