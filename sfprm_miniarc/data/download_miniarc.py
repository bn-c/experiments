#!/usr/bin/env python3
"""Download Mini-ARC task JSONs into data/mini_arc/.

Source: https://github.com/KSB21ST/MINI-ARC (NeurIPS 2022 workshop dataset).
Raw tasks live under data/MiniARC/ (149 JSON files, 5×5 grids, ARC schema).
"""

from __future__ import annotations

import io
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

REPO_ZIP_URL = "https://github.com/KSB21ST/MINI-ARC/archive/refs/heads/master.zip"
SOURCE_DOC = "https://github.com/KSB21ST/MINI-ARC/tree/master/data/MiniARC"

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "mini_arc"


def download(force: bool = False) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(OUT_DIR.glob("*.json"))
    if existing and not force:
        print(f"Found {len(existing)} JSON files in {OUT_DIR}; skip download (use --force).")
        return len(existing)

    print(f"Downloading Mini-ARC from {REPO_ZIP_URL}")
    print(f"Documented source tree: {SOURCE_DOC}")
    with urlopen(REPO_ZIP_URL, timeout=120) as resp:
        payload = resp.read()

    staging = OUT_DIR / "_extract"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        members = [m for m in zf.namelist() if "/data/MiniARC/" in m and m.endswith(".json")]
        if not members:
            raise RuntimeError("Zip archive contained no data/MiniARC/*.json files")
        for name in members:
            zf.extract(name, staging)

    # Flatten extracted files into OUT_DIR
    for src in staging.rglob("*.json"):
        dest = OUT_DIR / src.name
        shutil.copy2(src, dest)
    shutil.rmtree(staging)

    count = len(list(OUT_DIR.glob("*.json")))
    print(f"Wrote {count} Mini-ARC tasks to {OUT_DIR}")
    return count


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    force = "--force" in argv
    n = download(force=force)
    if n < 100:
        print(f"Warning: expected ~149 tasks, got {n}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
