#!/usr/bin/env bash
# 10k cooled cosine A/B: geometric cross-attention vs SwiGLU no-gauge baseline.
# Recipe matches Arm W / activation-10k: λ_k=0.5 (B2), batch 128, lr≈1.414e-4, T_max=10k.
# Arms: B0 = reuse SwiGLU aychd5jf (skip train); B1 = geo_karcher (λ_k=0); B2 = geo_log (λ_k=0.5).
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
# RUN_B1=1 RUN_B2=1 by default; B0 is reference-only (no train).
RUN_B1="${RUN_B1:-1}"
RUN_B2="${RUN_B2:-1}"

mkdir -p logs checkpoints/geo-attention-ab
SUMMARY="logs/geo-attention-ab-summary.txt"
: >"$SUMMARY"

{
  echo "B0_REF=aychd5jf MIXER=cross_coupled ACT=swiglu NO_GAUGE=1 LK=0.5"
  echo "B0_BLANK_5K=0.343 B0_BLANK_10K=0.379"
  echo "----"
} | tee -a "$SUMMARY"

run_arm() {
  local arm="$1"
  local mixer="$2"
  local lk="$3"
  local name="geo-attn-ab-${arm}-${mixer}-lk${lk}-b${BATCH}-s${STEPS}"
  local ckpt="checkpoints/geo-attention-ab/${name}"
  local log="logs/${name}.log"
  mkdir -p "$ckpt"
  echo "=== $name ===" | tee -a "$SUMMARY"

  uv run riemannian-sudoku train \
    --train-size "$TRAIN_SIZE" --val-size "$VAL_SIZE" \
    --steps "$STEPS" --batch-size "$BATCH" \
    --lr "$LR" --num-steps "$NUM_STEPS" \
    --lr-schedule cosine --grad-clip 1.0 \
    --early-stop-drop 0.05 --early-stop-patience 2 \
    --kinetic-weight "$lk" --constraint-weight 0 --supervise-all-steps \
    --eval-every "$EVAL_EVERY" --log-every "$LOG_EVERY" \
    --ckpt-every 1000 --num-workers 4 --pin-memory \
    --seed "$SEED" \
    --mixer "$mixer" \
    --ckpt-dir "$ckpt" --wandb-name "$name" \
    >"$log" 2>&1

  {
    echo "ARM=$arm MIXER=$mixer LK=$lk NAME=$name"
    grep -E '^(RUN_ID|N_PARAMS|VAL_BLANK|BEST_BLANK|EARLY_STOPPED|STOP_STEP|MIXER|USE_GAUGE|MLP_ACTIVATION)=' "$log" || true
    echo "----"
  } | tee -a "$SUMMARY"
  echo "=== done $name ==="
}

if [[ "$RUN_B1" -eq 1 ]]; then
  run_arm B1 geo_karcher 0.0
else
  echo "SKIP B1 (geo_karcher)" | tee -a "$SUMMARY"
fi

if [[ "$RUN_B2" -eq 1 ]]; then
  run_arm B2 geo_log 0.5
else
  echo "SKIP B2 (geo_log)" | tee -a "$SUMMARY"
fi

echo "All arms finished."
cat "$SUMMARY"
