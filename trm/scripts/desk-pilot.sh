#!/usr/bin/env bash
# ROCm desk pilot: 1K HF train rows, 5K test eval. Run from repo after `uv sync --extra rocm`.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CHECKPOINT="${ROOT}/checkpoints/checkpoint.last.pt"

uv run python -m trm.main train --train-size 1000
uv run python -m trm.main eval --checkpoint "$CHECKPOINT" --n-samples 5000
