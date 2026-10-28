# Riemannian Sudoku

Peer spike: Sudoku solver on the positive orthant of \(S^8\) (\(p = s^2\)) with
factorized row/col/box axial attention, fixed Clifford \(T^3\) positional encoding,
and an Exp-map + abs-renorm retraction.

Trains / evaluates on Hugging Face `sapientinc/sudoku-extreme` (desk HF default cache).

```bash
cd riemannian_sudoku
uv sync
uv run pytest
# smoke
uv run riemannian-sudoku train --train-size 1024 --val-size 128 --steps 200 --batch-size 32
# full desk (after tune): last.pt every 1000 steps, best.pt by val blank
uv run riemannian-sudoku train \
  --train-size 100000 --val-size 2048 --steps 50000 --batch-size 64 \
  --lr 3e-4 --num-steps 8 --eval-every 500 --log-every 50 --ckpt-every 1000
# resume
uv run riemannian-sudoku train --resume checkpoints/last.pt --steps 50000 ...
```

W&B project: `riemannian-sudoku`. Logs blank / exact-board / row / col / box on train+val.
