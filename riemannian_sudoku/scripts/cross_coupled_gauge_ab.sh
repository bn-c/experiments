#!/usr/bin/env bash
# Short ≤5k W&B A/B: does live T² gauge help vs cells-only cross_coupled?
# Recipe matches kinetic A3 (λ_k=0.5, λ_g=0.5 with gauge); G1 drops gauge entirely.
# Arms: G0 = with gauge (reuse prior A3 W&B if identical); G1 = --no-gauge.
set -euo pipefail
cd "$(dirname "$0")/.."
export WANDB_ENTITY="${WANDB_ENTITY:-nanodegree-forge}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.forge.coreweave.com}"

LR="${LR:-1.414213562e-4}"
NUM_STEPS="${NUM_STEPS:-8}"
BATCH="${BATCH:-128}"
STEPS="${STEPS:-5000}"
TRAIN_SIZE="${TRAIN_SIZE:-100000}"
VAL_SIZE="${VAL_SIZE:-2048}"
EVAL_EVERY="${EVAL_EVERY:-500}"
SEED="${SEED:-42}"
LOG_EVERY="${LOG_EVERY:-50}"
# Set RUN_G0=1 to retrain G0; default skips (reuse kinetic A3 vykzljx6).
RUN_G0="${RUN_G0:-0}"

mkdir -p logs checkpoints/cross-coupled-gauge-ab
SUMMARY="logs/cc-gauge-ab-summary.txt"
: >"$SUMMARY"

run_arm() {
  local arm="$1"
  local no_gauge="$2" # 0|1
  local lam_k="$3"
  local lam_g="$4"
  local gauge_tag
  if [[ "$no_gauge" -eq 1 ]]; then
    gauge_tag="nogauge"
  else
    gauge_tag="gauge"
  fi
  local name="cc-gauge-ab-${arm}-${gauge_tag}-lk${lam_k}-lg${lam_g}-b${BATCH}-s${STEPS}"
  local ckpt="checkpoints/cross-coupled-gauge-ab/${name}"
  local log="logs/${name}.log"
  mkdir -p "$ckpt"
  echo "=== $name (no_gauge=$no_gauge λ_k=$lam_k λ_g=$lam_g) ===" | tee -a "$SUMMARY"

  local extra=()
  if [[ "$no_gauge" -eq 1 ]]; then
    extra+=(--no-gauge)
  fi

  uv run riemannian-sudoku train \
    --train-size "$TRAIN_SIZE" --val-size "$VAL_SIZE" \
    --steps "$STEPS" --batch-size "$BATCH" \
    --lr "$LR" --num-steps "$NUM_STEPS" \
    --lr-schedule cosine --grad-clip 1.0 \
    --early-stop-drop 0.05 --early-stop-patience 2 \
    --kinetic-weight "$lam_k" --gauge-kinetic-weight "$lam_g" \
    --constraint-weight 0 --supervise-all-steps \
    --eval-every "$EVAL_EVERY" --log-every "$LOG_EVERY" \
    --ckpt-every 1000 --num-workers 4 --pin-memory \
    --seed "$SEED" \
    --mixer cross_coupled \
    "${extra[@]}" \
    --ckpt-dir "$ckpt" --wandb-name "$name" \
    >"$log" 2>&1

  {
    echo "ARM=$arm NAME=$name NO_GAUGE=$no_gauge LAM_K=$lam_k LAM_G=$lam_g"
    grep -E '^(RUN_ID|VAL_BLANK|BEST_BLANK|EARLY_STOPPED|STOP_STEP|MIXER|USE_GAUGE)=' "$log" || true
    echo "----"
  } | tee -a "$SUMMARY"
  echo "=== done $name ==="
}

if [[ "$RUN_G0" -eq 1 ]]; then
  run_arm G0 0 0.5 0.5
else
  echo "SKIP G0: reuse kinetic A3 W&B vykzljx6 (λ_k=0.5 λ_g=0.5 with gauge)" | tee -a "$SUMMARY"
fi

# G1: cells-only; λ_g N/A (pass 0; ignored when no gauge channel).
run_arm G1 1 0.5 0

echo "All arms finished."
cat "$SUMMARY"
