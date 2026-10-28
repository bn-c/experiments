#!/usr/bin/env bash
# Phase A/B GPU fidelity benches (exclusive GPU). Winner stack + I/O; toggle --compile / batch.
set -euo pipefail
cd "$(dirname "$0")/.."
export WANDB_ENTITY="${WANDB_ENTITY:-nanodegree-forge}"

PHASE="${1:-A}"
LR="${LR:-1e-4}"
NUM_STEPS="${NUM_STEPS:-8}"
TRAIN_SIZE="${TRAIN_SIZE:-100000}"
VAL_SIZE="${VAL_SIZE:-2048}"
EVAL_EVERY="${EVAL_EVERY:-500}"
SEED="${SEED:-42}"

common=(
  train
  --train-size "$TRAIN_SIZE" --val-size "$VAL_SIZE"
  --num-steps "$NUM_STEPS" --lr "$LR" --lr-schedule cosine
  --grad-clip 1.0 --early-stop-drop 0 --early-stop-patience 0
  --kinetic-weight 0.05 --constraint-weight 0 --supervise-all-steps
  --eval-every "$EVAL_EVERY" --log-every 50
  --ckpt-every 0 --num-workers 4 --pin-memory
  --seed "$SEED" --no-wandb
)

run_one() {
  local name="$1"; shift
  local ckpt="checkpoints/gpu-fid/${name}"
  mkdir -p "$ckpt" logs
  echo "=== $name ==="
  # shellcheck disable=SC2068
  uv run riemannian-sudoku ${common[@]} --ckpt-dir "$ckpt" --wandb-name "$name" "$@" \
    2>&1 | tee "logs/${name}.log"
}

case "$PHASE" in
  A)
    # 2k A/B batch 64: baseline vs compile
    STEPS="${STEPS:-2000}"
    BATCH=64
    run_one "A-base-b${BATCH}-s${STEPS}" --steps "$STEPS" --batch-size "$BATCH"
    run_one "A-compile-b${BATCH}-s${STEPS}" --steps "$STEPS" --batch-size "$BATCH" --compile
    ;;
  B2k)
    STEPS="${STEPS:-2000}"
    BATCH=128
    LR_B="$(python3 -c "print(f'{1e-4 * (128/64)**0.5:.6g}')")"
    run_one "B-compile-b${BATCH}-s${STEPS}-lr${LR_B}" \
      --steps "$STEPS" --batch-size "$BATCH" --lr "$LR_B" --compile
    ;;
  B20k)
    STEPS="${STEPS:-20000}"
    BATCH=128
    LR_B="$(python3 -c "print(f'{1e-4 * (128/64)**0.5:.6g}')")"
    run_one "B-compile-b${BATCH}-s${STEPS}-lr${LR_B}" \
      --steps "$STEPS" --batch-size "$BATCH" --lr "$LR_B" --compile \
      --ckpt-every 1000
    ;;
  *)
    echo "usage: $0 A|B2k|B20k" >&2
    exit 2
    ;;
esac
