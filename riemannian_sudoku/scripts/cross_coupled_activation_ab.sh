#!/usr/bin/env bash
# 10k cooled cosine A/B: cross_coupled --no-gauge MLP activation (SiLU vs SwiGLU).
# Recipe matches no-gauge G1 / cooled 10k: λ_k=0.5, batch 128, lr≈1.414e-4, T_max=10k.
# Arms: S = silu (same width as gelu); W = swiglu (mid hidden ≈170, fair-ish params).
# GELU baseline: reuse gauge-ablation G1 fro9t7fo (@5k) / cooled 982j5dmb (@10k resume).
set -euo pipefail
cd "$(dirname "$0")/.."
export WANDB_ENTITY="${WANDB_ENTITY:-nanodegree-forge}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.forge.coreweave.com}"

LR="${LR:-1.414213562e-4}"
NUM_STEPS="${NUM_STEPS:-8}"
BATCH="${BATCH:-128}"
STEPS="${STEPS:-10000}"
TRAIN_SIZE="${TRAIN_SIZE:-100000}"
VAL_SIZE="${VAL_SIZE:-2048}"
EVAL_EVERY="${EVAL_EVERY:-500}"
SEED="${SEED:-42}"
LOG_EVERY="${LOG_EVERY:-50}"
# RUN_S=1 RUN_W=1 by default; set to 0 to skip an arm.
RUN_S="${RUN_S:-1}"
RUN_W="${RUN_W:-1}"

mkdir -p logs checkpoints/cross-coupled-activation-ab
SUMMARY="logs/cc-activation-ab-summary.txt"
: >"$SUMMARY"

run_arm() {
  local arm="$1"
  local act="$2"
  local name="cc-act-ab-${arm}-${act}-nogauge-lk0.5-b${BATCH}-s${STEPS}"
  local ckpt="checkpoints/cross-coupled-activation-ab/${name}"
  local log="logs/${name}.log"
  mkdir -p "$ckpt"
  echo "=== $name ===" | tee -a "$SUMMARY"

  uv run riemannian-sudoku train \
    --train-size "$TRAIN_SIZE" --val-size "$VAL_SIZE" \
    --steps "$STEPS" --batch-size "$BATCH" \
    --lr "$LR" --num-steps "$NUM_STEPS" \
    --lr-schedule cosine --grad-clip 1.0 \
    --early-stop-drop 0.05 --early-stop-patience 2 \
    --kinetic-weight 0.5 --constraint-weight 0 --supervise-all-steps \
    --eval-every "$EVAL_EVERY" --log-every "$LOG_EVERY" \
    --ckpt-every 1000 --num-workers 4 --pin-memory \
    --seed "$SEED" \
    --mixer cross_coupled --no-gauge \
    --mlp-activation "$act" \
    --ckpt-dir "$ckpt" --wandb-name "$name" \
    >"$log" 2>&1

  {
    echo "ARM=$arm ACT=$act NAME=$name"
    grep -E '^(RUN_ID|N_PARAMS|VAL_BLANK|BEST_BLANK|EARLY_STOPPED|STOP_STEP|MIXER|USE_GAUGE|MLP_ACTIVATION)=' "$log" || true
    echo "----"
  } | tee -a "$SUMMARY"
  echo "=== done $name ==="
}

if [[ "$RUN_S" -eq 1 ]]; then
  run_arm S silu
else
  echo "SKIP S (silu)" | tee -a "$SUMMARY"
fi

if [[ "$RUN_W" -eq 1 ]]; then
  run_arm W swiglu
else
  echo "SKIP W (swiglu)" | tee -a "$SUMMARY"
fi

echo "All arms finished."
cat "$SUMMARY"
