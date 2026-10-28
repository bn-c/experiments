# Riemannian Sudoku

Peer spike: Sudoku solver on the positive orthant of \(S^8\) (\(p = s^2\)) with
factorized row/col/box axial attention, fixed Clifford \(T^3\) positional encoding,
and an Exp-map + abs-renorm retraction.

Trains / evaluates on Hugging Face `sapientinc/sudoku-extreme` (desk HF default cache).

```bash
cd riemannian_sudoku
uv sync
uv run pytest
uv run riemannian-sudoku train --train-size 1024 --val-size 128 --steps 200 --batch-size 32
```

W&B project: `riemannian-sudoku`.
