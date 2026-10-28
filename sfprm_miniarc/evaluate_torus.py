#!/usr/bin/env python3
"""Evaluate a torus SFPRM checkpoint — sphere evaluate.py untouched."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluate as sphere_eval  # noqa: E402
from models.sfprm_torus_core import SFPRMTorus  # noqa: E402

sphere_eval.SFPRM = SFPRMTorus  # type: ignore[misc, assignment]

if __name__ == "__main__":
    # Re-enter evaluate.main via the same CLI.
    from evaluate import main  # noqa: E402

    raise SystemExit(main())
