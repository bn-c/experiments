#!/usr/bin/env bash
# Exclusive short A/B: eager vs compile(mode=default), same seed/batch 64.
set -euo pipefail
cd "$(dirname "$0")/.."
STEPS="${STEPS:-400}"
WARM="${WARM:-100}"
BATCH=64
SEED=42

common=(
  train
  --train-size 100000 --val-size 2048
  --steps "$STEPS" --batch-size "$BATCH"
  --lr 1e-4 --num-steps 8 --lr-schedule cosine
  --grad-clip 1.0 --early-stop-drop 0 --early-stop-patience 0
  --kinetic-weight 0.05 --constraint-weight 0 --supervise-all-steps
  --num-workers 4 --pin-memory
  --eval-every "$STEPS" --log-every 50 --ckpt-every 0
  --seed "$SEED" --no-wandb
  --throughput-warmup "$WARM"
)

mkdir -p logs checkpoints/gpu-fid

echo "=== VERIFY eager (steps=$STEPS warmup=$WARM) ==="
uv run riemannian-sudoku "${common[@]}" \
  --ckpt-dir "checkpoints/gpu-fid/V-eager-b${BATCH}-s${STEPS}" \
  2>&1 | tee "logs/V-eager-b${BATCH}-s${STEPS}.log"

echo "=== VERIFY compile (steps=$STEPS warmup=$WARM) ==="
uv run riemannian-sudoku "${common[@]}" --compile \
  --ckpt-dir "checkpoints/gpu-fid/V-compile-b${BATCH}-s${STEPS}" \
  2>&1 | tee "logs/V-compile-b${BATCH}-s${STEPS}.log"

echo "=== MICRO fixed-batch (no DataLoader loop / no eval) ==="
uv run python <<'PY'
import time
import torch
from torch.utils.data import DataLoader
from riemannian_sudoku.data import SudokuDataset, SudokuTorchDataset, split_train_val
from riemannian_sudoku.model import RiemannianSudokuSolver

device = torch.device('cuda')
df = SudokuDataset().load_split('train', n_samples=8192)
train_df, _ = split_train_val(df, val_size=512, seed=42)
loader = DataLoader(
    SudokuTorchDataset(train_df, augment=True, seed=42),
    batch_size=64, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True,
)
batch = next(iter(loader))
p = batch['puzzle'].to(device, non_blocking=True)
s = batch['solution'].to(device, non_blocking=True)
m = batch['mask'].to(device, non_blocking=True)

def bench(label, model, warmup=40, iters=150):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(warmup):
        out = model(p, s, m)
        opt.zero_grad(set_to_none=True)
        out['loss'].backward()
        opt.step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = model(p, s, m)
        opt.zero_grad(set_to_none=True)
        out['loss'].backward()
        opt.step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f'MICRO_{label}_IT_S={iters/dt:.4f} warmup={warmup} iters={iters} wall={dt:.2f}')

m = RiemannianSudokuSolver(kinetic_weight=0.05, constraint_weight=0.0, supervise_all_steps=True).to(device)
bench('eager', m)
m2 = RiemannianSudokuSolver(kinetic_weight=0.05, constraint_weight=0.0, supervise_all_steps=True).to(device)
m2.forward = torch.compile(m2.forward, mode='default')
bench('compile', m2)
print(f'TORCH={torch.__version__} HIP={torch.version.hip}')
PY

echo VERIFY_DONE=0
