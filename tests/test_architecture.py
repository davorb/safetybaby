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
