#!/usr/bin/env bash
# Eval-only damp sweep on tip-terminal-brake best.pt (zero train).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT="${CKPT:-$ROOT/checkpoints/bipartite-sn-tm-tip-terminal-brake-10k/bipartite-sn-tm-tip-terminal-brake-s10000/best.pt}"
OUT="${OUT:-$ROOT/logs/eval-damp-keval}"
mkdir -p "$OUT"
KS_SCREEN=(24 32 64 128 256 512)
KS_FULL=(8 12 16 24 32 64 128 256 512)

run_one() {
  local tag="$1"; shift
  local log="$OUT/${tag}.txt"
  echo "=== $tag $* ===" | tee "$log"
  uv run python "$ROOT/scripts/bipartite_sn_tm_keval.py" \
    --checkpoint "$CKPT" "$@" 2>&1 | tee -a "$log"
}

MODE="${1:-screen}"

if [[ "$MODE" == "screen" ]]; then
  # Undamped reference (high-K only)
  run_one undamped-screen --ks "${KS_SCREEN[@]}" --damp-beta 0.0
  # β-only torus
  for b in 0.05 0.1 0.25 0.5 1.0 2.0; do
    run_one "beta-${b}" --ks "${KS_SCREEN[@]}" --damp-beta "$b"
  done
  # γ-only torus
  for g in 0.9 0.95 0.98 0.99; do
    run_one "gamma-${g}" --ks "${KS_SCREEN[@]}" --damp-gamma "$g"
  done
  # β + u_c
  for b in 0.25 0.5 1.0; do
    run_one "beta-${b}-uc" --ks "${KS_SCREEN[@]}" --damp-beta "$b" --damp-uc
  done
  # γ + u_c
  for g in 0.95 0.98; do
    run_one "gamma-${g}-uc" --ks "${KS_SCREEN[@]}" --damp-gamma "$g" --damp-uc
  done
elif [[ "$MODE" == "full" ]]; then
  # Pass remaining args as tags to re-run full grid; or defaults.
  shift || true
  if [[ $# -eq 0 ]]; then
    run_one undamped-full --ks "${KS_FULL[@]}" --damp-beta 0.0
    run_one beta-0.25-full --ks "${KS_FULL[@]}" --damp-beta 0.25
    run_one beta-0.5-full --ks "${KS_FULL[@]}" --damp-beta 0.5
    run_one gamma-0.98-full --ks "${KS_FULL[@]}" --damp-gamma 0.98
  else
    # Usage: full --tag name --damp-beta 0.25 [--damp-uc] ...
    while [[ $# -gt 0 ]]; do
      tag="$1"; shift
      # next args until next --tag or end: collect keval flags
      flags=()
      while [[ $# -gt 0 && "$1" != --tag ]]; do
        flags+=("$1"); shift
      done
      [[ "${1:-}" == --tag ]] && shift
      run_one "${tag}-full" --ks "${KS_FULL[@]}" "${flags[@]}"
    done
  fi
else
  echo "usage: $0 [screen|full]" >&2
  exit 2
fi
