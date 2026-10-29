# Riemannian Sudoku

Peer spike: Sudoku solver on the positive orthant of \(S^8\) (\(p = s^2\)) with
an Exp-map + abs-renorm retraction. Default mixer `--mixer euclid_axial` uses
factorized row/col/box axial attention and fixed Clifford \(T^3\) PE.
`--mixer cross_coupled` replaces that with a live \(T^2\) (row×col) gauge and a
shared ambient MLP → per-manifold tangent project → parallel Exp.
`--mixer geo_karcher` / `geo_log` are cells-only geodesic affinities + 2-step
Karcher (state←mean, or `v=Log` then Exp).

Trains / evaluates on Hugging Face `sapientinc/sudoku-extreme` (desk HF default cache).

```bash
cd riemannian_sudoku
uv sync
uv run pytest
# smoke
uv run riemannian-sudoku train --train-size 1024 --val-size 128 --steps 200 --batch-size 32
# desk tune (LR × num_steps × batch); then full with locked hparams
bash scripts/desk_tune.sh logs/desk-tune-clean.tsv
# locked after clean sweep: lr=1e-4, num_steps=8, batch=64
bash scripts/full_train.sh   # or: LR=1e-4 NUM_STEPS=8 BATCH=64 STEPS=50000 ...
# resume (mixer is read from ckpt args; pass --mixer only to override)
uv run riemannian-sudoku train --resume checkpoints/full/last.pt --steps 50000 ...
# cross_coupled A/B: --mixer cross_coupled  (see scripts/cross_coupled_ab.sh)
```

W&B project: `riemannian-sudoku`. Logs blank / exact-board / row / col / box on train+val.
Desk full-train report lives in the agent-store docs (`riemannian-sudoku-full-train.md`).
