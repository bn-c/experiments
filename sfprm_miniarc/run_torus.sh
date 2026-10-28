#!/usr/bin/env bash
# Torus SFPRM A/B train (product-of-circles). Sphere run.sh / train.py untouched.
# Full desk recipe matches dinw2si9 (seq-25, compile mode=default, batch 768, 12k).
# Smoke: ./run_torus.sh --smoke
set -euo pipefail
cd "$(dirname "$0")"

SMOKE=0
if [[ "${1:-}" == "--smoke" ]]; then
  SMOKE=1
  shift
fi

export PATH="/etc/profiles/per-user/tech/bin:/run/current-system/sw/bin:${PATH:-}"
PY="${PY:-$(command -v python || true)}"
if [[ -x /home/tech/Documents/experiments/.venv/bin/python ]]; then
  PY=/home/tech/Documents/experiments/.venv/bin/python
fi
if [[ -z "${PY}" ]]; then
  echo "python not found" >&2
  exit 1
fi

echo "[1/3] Ensuring Mini-ARC tasks are present..."
"$PY" data/download_miniarc.py

if [[ "$SMOKE" -eq 1 ]]; then
  echo "[2/3] Torus smoke training (tiny steps, no compile)..."
  "$PY" train_torus.py \
    --steps 3 \
    --bptt_k 2 \
    --batch_size 4 \
    --max_tasks 4 \
    --eval_every 3 \
    --ckpt_every 3 \
    --n_color_perms 2 \
    --max_loops 8 \
    --max_iter_train 4 \
    --no-wandb \
    --out_dir ./checkpoints/smoke-torus \
    "$@"

  echo "[3/3] Torus smoke evaluate..."
  "$PY" evaluate_torus.py \
    --checkpoint ./checkpoints/smoke-torus/model_best.pt \
    --max_tasks 4 \
    --max_loops 8 \
    --no-wandb
  echo "Torus smoke OK."
  exit 0
fi

# Full 12k A/B vs sphere dinw2si9.
if [[ -f /home/tech/Documents/experiments/trm/.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /home/tech/Documents/experiments/trm/.env
  set +a
fi
export WANDB_PROJECT="${WANDB_PROJECT:-sfprm-miniarc}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-manifold-ab-12k}"
export WANDB_ENTITY="${WANDB_ENTITY:-nanodegree-forge}"

OUT_DIR=${OUT_DIR:-./checkpoints/torus-ab-12k}
BATCH_SIZE=${BATCH_SIZE:-768}
STEPS=${STEPS:-12000}
LOG=logs/full-train-torus-$(date -u +%Y%m%dT%H%M%SZ).log
mkdir -p "$OUT_DIR" logs

echo "[2/3] Training torus SFPRM ($STEPS steps, batch=$BATCH_SIZE, compile mode=default)..."
echo "Logging to $LOG"
echo "git=$(git -C /home/tech/Documents/experiments rev-parse --short HEAD) branch=$(git -C /home/tech/Documents/experiments branch --show-current)" | tee -a "$LOG"

"$PY" logs/train_torus_compile_default.py \
  --steps "$STEPS" \
  --bptt_k 6 \
  --batch_size "$BATCH_SIZE" \
  --max_seq_len 25 \
  --lr_muon 5e-3 \
  --lr_adam 1e-3 \
  --weight_decay 0.1 \
  --compile \
  --out_dir "$OUT_DIR" \
  --wandb-project sfprm-miniarc \
  --wandb-entity "$WANDB_ENTITY" \
  "$@" \
  2>&1 | tee -a "$LOG"

echo "[3/3] Evaluating torus best ckpt (1-shot + vote)..." | tee -a "$LOG"
"$PY" evaluate_torus.py \
  --checkpoint "$OUT_DIR/model_best.pt" \
  --tau 0.01 \
  --max_loops 200 \
  --max_seq_len 25 \
  --wandb-project sfprm-miniarc \
  --wandb-entity "$WANDB_ENTITY" \
  2>&1 | tee -a "$LOG"

echo "DONE at $(date -Is)" | tee -a "$LOG"
