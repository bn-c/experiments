#!/usr/bin/env bash
# Setup-scrutiny exclusive ladder: Phase 1 A0–A4, then Phase 2 B1–B3 on winner.
# Locked optimizer protocol from us1ine3b; one knob per cell. Do not edit the plan file.
set -euo pipefail
cd "$(dirname "$0")/.."
export WANDB_ENTITY="${WANDB_ENTITY:-nanodegree-forge}"

LR=1e-4
NUM_STEPS=8
BATCH=64
STEPS_SHORT=5000
TRAIN_SHORT=16000
VAL_SHORT=512
EVAL_EVERY=500
LOG_EVERY=50
LR_SCHEDULE=cosine
GRAD_CLIP=1.0
EARLY_STOP_DROP=0.15
EARLY_STOP_PATIENCE=2
SEED=42

mkdir -p logs checkpoints/scrutiny
TSV=logs/scrutiny-ladder.tsv
echo -e "phase\trun\tkinetic\tconstraint\textra\trun_id\tbest_blank\tval_blank\tval_row\tval_col\tval_box\tval_exact\tearly_stopped\tstop_step" > "$TSV"

run_one() {
  local phase="$1" run="$2" kin="$3" con="$4" ckpt="$5" name="$6" train_size="$7" val_size="$8" steps="$9"
  shift 9 || true
  local extra_args=("$@")
  mkdir -p "$ckpt"
  echo "=== $phase $run  λk=$kin λc=$con train=$train_size steps=$steps ==="
  # shellcheck disable=SC2068
  LOG=$(uv run riemannian-sudoku train \
    --train-size "$train_size" --val-size "$val_size" \
    --steps "$steps" --batch-size "$BATCH" \
    --lr "$LR" --num-steps "$NUM_STEPS" \
    --lr-schedule "$LR_SCHEDULE" \
    --grad-clip "$GRAD_CLIP" \
    --early-stop-drop "$EARLY_STOP_DROP" \
    --early-stop-patience "$EARLY_STOP_PATIENCE" \
    --kinetic-weight "$kin" --constraint-weight "$con" \
    --eval-every "$EVAL_EVERY" --log-every "$LOG_EVERY" \
    --ckpt-every 1000 --ckpt-dir "$ckpt" \
    --wandb-name "$name" \
    --seed "$SEED" \
    ${extra_args[@]+"${extra_args[@]}"} \
    2>&1 | tee "logs/${name}.log")
  local rid vb vr vc vbx ve bb es ss
  rid=$(echo "$LOG" | grep -E '^RUN_ID=' | tail -1 | cut -d= -f2)
  vb=$(echo "$LOG" | grep -E '^VAL_BLANK=' | tail -1 | cut -d= -f2)
  vr=$(echo "$LOG" | grep -E '^VAL_ROW=' | tail -1 | cut -d= -f2)
  vc=$(echo "$LOG" | grep -E '^VAL_COL=' | tail -1 | cut -d= -f2)
  vbx=$(echo "$LOG" | grep -E '^VAL_BOX=' | tail -1 | cut -d= -f2)
  ve=$(echo "$LOG" | grep -E '^VAL_EXACT=' | tail -1 | cut -d= -f2)
  bb=$(echo "$LOG" | grep -E '^BEST_BLANK=' | tail -1 | cut -d= -f2)
  es=$(echo "$LOG" | grep -E '^EARLY_STOPPED=' | tail -1 | cut -d= -f2)
  ss=$(echo "$LOG" | grep -E '^STOP_STEP=' | tail -1 | cut -d= -f2)
  local extra_tag=""
  if ((${#extra_args[@]})); then
    extra_tag=$(IFS=,; echo "${extra_args[*]}")
  fi
  echo -e "${phase}\t${run}\t${kin}\t${con}\t${extra_tag}\t${rid}\t${bb}\t${vb}\t${vr}\t${vc}\t${vbx}\t${ve}\t${es}\t${ss}" >> "$TSV"
  echo "DONE $run RUN_ID=$rid BEST_BLANK=$bb VAL_ROW=$vr"
}

# --- Phase 1: loss weight ladder ---
run_one A A0 0.05 0.1  checkpoints/scrutiny-A0 scrutiny-A0-control "$TRAIN_SHORT" "$VAL_SHORT" "$STEPS_SHORT"
run_one A A1 0.05 0.0  checkpoints/scrutiny-A1 scrutiny-A1-c0 "$TRAIN_SHORT" "$VAL_SHORT" "$STEPS_SHORT"
run_one A A2 0.05 0.2  checkpoints/scrutiny-A2 scrutiny-A2-c0.2 "$TRAIN_SHORT" "$VAL_SHORT" "$STEPS_SHORT"
run_one A A3 0.0  0.1  checkpoints/scrutiny-A3 scrutiny-A3-k0 "$TRAIN_SHORT" "$VAL_SHORT" "$STEPS_SHORT"
run_one A A4 0.1  0.1  checkpoints/scrutiny-A4 scrutiny-A4-k0.1 "$TRAIN_SHORT" "$VAL_SHORT" "$STEPS_SHORT"

# Lock winner: max best_blank, tie-break val_row among A rows.
WINNER=$(awk -F'\t' 'NR>1 && $1=="A" {
  if ($7+0 > best+0 || ($7+0==best+0 && $9+0 > brow+0)) { best=$7; brow=$9; w=$2; k=$3; c=$4 }
}
END { print w "\t" k "\t" c "\t" best "\t" brow }' "$TSV")
WIN_RUN=$(echo "$WINNER" | cut -f1)
WIN_K=$(echo "$WINNER" | cut -f2)
WIN_C=$(echo "$WINNER" | cut -f3)
echo "PHASE1_WINNER=$WIN_RUN λk=$WIN_K λc=$WIN_C" | tee logs/scrutiny-phase1-winner.txt
echo "$WINNER" >> logs/scrutiny-phase1-winner.txt

# --- Phase 2: protocol probes on winner weights ---
run_one B B1 "$WIN_K" "$WIN_C" checkpoints/scrutiny-B1 scrutiny-B1-noaug "$TRAIN_SHORT" "$VAL_SHORT" "$STEPS_SHORT" --no-augment
run_one B B2 "$WIN_K" "$WIN_C" checkpoints/scrutiny-B2 scrutiny-B2-allsteps "$TRAIN_SHORT" "$VAL_SHORT" "$STEPS_SHORT" --supervise-all-steps
run_one B B3 "$WIN_K" "$WIN_C" checkpoints/scrutiny-B3 scrutiny-B3-fulldata 100000 2048 "$STEPS_SHORT"

echo "LADDER_DONE"
cat "$TSV"
