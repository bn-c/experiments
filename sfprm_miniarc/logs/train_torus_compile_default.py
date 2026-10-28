#!/usr/bin/env python3
"""Launch train_torus.py with torch.compile forced to mode=default (ROCm desk)."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
_orig = torch.compile


def _compile(fn=None, *a, **kw):
    kw["mode"] = "default"
    if fn is None:
        return lambda f: _orig(f, *a, **kw)
    return _orig(fn, *a, **kw)


torch.compile = _compile  # type: ignore[assignment]
sys.argv = ["train_torus.py", *sys.argv[1:]]
print(f"WRAPPER: torus train; forcing torch.compile mode=default; argv={sys.argv}", flush=True)
runpy.run_path(str(_ROOT / "train_torus.py"), run_name="__main__")
