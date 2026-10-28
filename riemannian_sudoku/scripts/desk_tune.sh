#!/usr/bin/env bash
# Clean desk tune: LR × num_steps × batch. Lock best by val blank + row.
# Grid: LR ∈ {1e-4, 3e-4, 1e-3} × num_steps ∈ {8, 12} × batch ∈ {32, 64}
set -euo pipefail
cd "$(dirname "$0")/.."
export WANDB_ENTITY="${WANDB_ENTITY:-nanodegree-forge}"
OUT="${1:-logs/desk-tune-clean.tsv}"
mkdir -p "$(dirname "$OUT")" logs checkpoints/tune
echo -e "lr\tnum_steps\tbatch\tsteps\trun_id\tval_blank\tval_row\tval_col\tval_box\tval_exact\tbest_blank" > "$OUT"

TRAIN_SIZE="${TRAIN_SIZE:-8192}"
VAL_SIZE="${VAL_SIZE:-512}"
STEPS="${STEPS:-1500}"
EVAL_EVERY="${EVAL_EVERY:-250}"
LOG_EVERY="${LOG_EVERY:-50}"
TAG="${TUNE_TAG:-clean}"

for LR in 1e-4 3e-4 1e-3; do
  for NS in 8 12; do
    for BATCH in 32 64; do
      NAME="${TAG}-lr${LR}-ns${NS}-b${BATCH}"
      CKPT="checkpoints/tune/${NAME}"
      mkdir -p "$CKPT"
      echo "=== $NAME ==="
      LOG=$(uv run riemannian-sudoku train \
        --train-size "$TRAIN_SIZE" --val-size "$VAL_SIZE" \
        --steps "$STEPS" --batch-size "$BATCH" \
        --lr "$LR" --num-steps "$NS" \
        --eval-every "$EVAL_EVERY" --log-every "$LOG_EVERY" \
        --ckpt-every 500 --ckpt-dir "$CKPT" \
        --wandb-name "$NAME" \
        --seed 42 2>&1 | tee "logs/${NAME}.log")
      RUN_ID=$(echo "$LOG" | grep -E '^RUN_ID=' | tail -1 | cut -d= -f2)
      VB=$(echo "$LOG" | grep -E '^VAL_BLANK=' | tail -1 | cut -d= -f2)
      VR=$(echo "$LOG" | grep -E '^VAL_ROW=' | tail -1 | cut -d= -f2)
      VC=$(echo "$LOG" | grep -E '^VAL_COL=' | tail -1 | cut -d= -f2)
      VX=$(echo "$LOG" | grep -E '^VAL_BOX=' | tail -1 | cut -d= -f2)
      VE=$(echo "$LOG" | grep -E '^VAL_EXACT=' | tail -1 | cut -d= -f2)
      BB=$(echo "$LOG" | grep -E '^BEST_BLANK=' | tail -1 | cut -d= -f2)
      echo -e "${LR}\t${NS}\t${BATCH}\t${STEPS}\t${RUN_ID}\t${VB}\t${VR}\t${VC}\t${VX}\t${VE}\t${BB}" | tee -a "$OUT"
    done
  done
done
echo "Wrote $OUT"
