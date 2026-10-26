#!/usr/bin/env python3
"""Materialize Mini-ARC task JSONs into data/mini_arc/ from Hugging Face.

HF source (Kim et al., NeurIPS 2022 nCSI workshop — same ~149 × 5×5 tasks as
KSB21ST/MINI-ARC):

  dataset id : andrewcropper/ilp-datasets
  subset     : mini-arc/   (149 task dirs; not a ``datasets`` config name —
               ``load_dataset`` cannot open this repo as tabular configs)
  splits     : each task has train/ and test/ Prolog files; we convert those
               into one ARC-schema JSON per task (train demos + test query)

Loading path: ``huggingface_hub.snapshot_download`` of ``mini-arc/**``, then
Prolog → ARC ``{train,test}`` grids, then optional ``datasets.Dataset.from_list``
for a typed table before writing local JSON cache for ``dataset.py``.

Hub page: https://huggingface.co/datasets/andrewcropper/ilp-datasets
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from datasets import Dataset
from huggingface_hub import snapshot_download

HF_DATASET_ID = "andrewcropper/ilp-datasets"
HF_SUBSET = "mini-arc"  # path prefix inside the dataset repo (not a load_dataset config)
HF_HUB_URL = f"https://huggingface.co/datasets/{HF_DATASET_ID}"
EXPECTED_TASKS = 149

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "mini_arc"

_IN_RE = re.compile(r"in\((\w+),(\d+),(\d+),(\d+)\)\.")
_EMPTY_RE = re.compile(r"empty\((\w+),(\d+),(\d+)\)\.")
_HEIGHT_RE = re.compile(r"height\((\w+),(\d+)\)\.")
_WIDTH_RE = re.compile(r"width\((\w+),(\d+)\)\.")
_POS_OUT_RE = re.compile(r"pos\(out\((\w+),(\d+),(\d+),(\d+)\)\)\.")


def _parse_input_grids(bk_text: str) -> dict[str, list[list[int]]]:
    cells: dict[str, dict[tuple[int, int], int]] = defaultdict(dict)
    dims: dict[str, dict[str, int]] = defaultdict(dict)
    for m in _IN_RE.finditer(bk_text):
        ex, r, c, v = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        cells[ex][(r, c)] = v
    for m in _EMPTY_RE.finditer(bk_text):
        ex, r, c = m.group(1), int(m.group(2)), int(m.group(3))
        cells[ex][(r, c)] = 0
    for m in _HEIGHT_RE.finditer(bk_text):
        dims[m.group(1)]["h"] = int(m.group(2))
    for m in _WIDTH_RE.finditer(bk_text):
        dims[m.group(1)]["w"] = int(m.group(2))

    grids: dict[str, list[list[int]]] = {}
    for ex, cell_map in cells.items():
        h = dims.get(ex, {}).get("h", 5)
        w = dims.get(ex, {}).get("w", 5)
        g = [[0] * w for _ in range(h)]
        for (r, c), v in cell_map.items():
            g[r - 1][c - 1] = v
        grids[ex] = g
    return grids


def _parse_output_grids(
    exs_text: str, example_ids: list[str], dims: dict[str, tuple[int, int]]
) -> dict[str, list[list[int]]]:
    cells: dict[str, dict[tuple[int, int], int]] = defaultdict(dict)
    for m in _POS_OUT_RE.finditer(exs_text):
        ex, r, c, v = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        cells[ex][(r, c)] = v
    grids: dict[str, list[list[int]]] = {}
    for ex in example_ids:
        h, w = dims.get(ex, (5, 5))
        g = [[0] * w for _ in range(h)]
        for (r, c), v in cells.get(ex, {}).items():
            g[r - 1][c - 1] = v
        grids[ex] = g
    return grids


def _task_from_dir(task_dir: Path) -> dict:
    """Convert one ILP mini-arc task directory into ARC-schema train/test dict."""
    train_bk = (task_dir / "train" / "bk.pl").read_text()
    train_exs = (task_dir / "train" / "exs.pl").read_text()
    test_bk = (task_dir / "test" / "bk.pl").read_text()
    test_exs = (task_dir / "test" / "exs.pl").read_text()

    train_in = _parse_input_grids(train_bk)
    train_ids = sorted(train_in.keys())  # e1, e2, ...
    train_dims = {ex: (len(g), len(g[0])) for ex, g in train_in.items()}
    train_out = _parse_output_grids(train_exs, train_ids, train_dims)

    test_in = _parse_input_grids(test_bk)
    test_ids = sorted(test_in.keys())  # t1, ...
    test_dims = {ex: (len(g), len(g[0])) for ex, g in test_in.items()}
    test_out = _parse_output_grids(test_exs, test_ids, test_dims)

    if not train_ids or not test_ids:
        raise RuntimeError(f"Missing train/test examples in {task_dir}")

    return {
        "train": [{"input": train_in[ex], "output": train_out[ex]} for ex in train_ids],
        "test": [{"input": test_in[ex], "output": test_out[ex]} for ex in test_ids],
    }


def _list_task_dirs(snapshot: Path) -> list[Path]:
    root = snapshot / HF_SUBSET
    if not root.is_dir():
        raise FileNotFoundError(f"Expected {root} after snapshot_download")
    return sorted(p for p in root.iterdir() if p.is_dir())


def download(force: bool = False) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Drop any leftover GitHub zip extract staging (legacy download_miniarc).
    staging = OUT_DIR / "_extract"
    if staging.exists():
        shutil.rmtree(staging)

    existing = list(OUT_DIR.glob("*.json"))
    if existing and not force:
        print(f"Found {len(existing)} JSON files in {OUT_DIR}; skip download (use --force).")
        return len(existing)

    print(f"Fetching Mini-ARC from Hugging Face: {HF_DATASET_ID} / {HF_SUBSET}/")
    print(f"Hub: {HF_HUB_URL}  (train+test Prolog per task → ARC JSON)")
    snapshot = Path(
        snapshot_download(
            repo_id=HF_DATASET_ID,
            repo_type="dataset",
            allow_patterns=[f"{HF_SUBSET}/**"],
        )
    )

    task_dirs = _list_task_dirs(snapshot)
    rows: list[dict] = []
    for task_dir in task_dirs:
        task = _task_from_dir(task_dir)
        rows.append({"task_id": task_dir.name, "train": task["train"], "test": task["test"]})

    # Prefer datasets library: materialize a typed table (validates row schema).
    ds = Dataset.from_list(rows)
    if len(ds) != len(rows):
        raise RuntimeError(f"datasets.Dataset length mismatch: {len(ds)} vs {len(rows)}")

    if force:
        for old in OUT_DIR.glob("*.json"):
            old.unlink()

    for row in rows:
        dest = OUT_DIR / f"{row['task_id']}.json"
        dest.write_text(json.dumps({"train": row["train"], "test": row["test"]}))

    count = len(list(OUT_DIR.glob("*.json")))
    print(f"Wrote {count} Mini-ARC tasks to {OUT_DIR} (HF {HF_DATASET_ID}:{HF_SUBSET})")
    return count


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    force = "--force" in argv
    n = download(force=force)
    if n < 100:
        print(f"Warning: expected ~{EXPECTED_TASKS} tasks, got {n}", file=sys.stderr)
        return 1
    if n != EXPECTED_TASKS:
        print(f"Note: expected {EXPECTED_TASKS} tasks, got {n}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
