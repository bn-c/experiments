# Riemannian Sudoku

Peer spike: Sudoku solver on the positive orthant of \(S^8\) (\(p = s^2\)) with
an Exp-map + abs-renorm retraction.

## Supported on `main` (desk default)

**Canonical mixer:** `--mixer bipartite_sn_tm` (A2 bipartite \(S^{63}\leftrightarrow T^{32}\)
with FiLM vf\_x, y0=\(\pi/4\), logit init 4, unnormalized house claims, supervise-all-steps ramp).
Promoted via PR #64; W&B run name `bipartite-sn-tm-a2-y0-logit-s10000`.

**Legacy mixers still in CLI** (for resume / comparison): `euclid_axial`, `cross_coupled`.

**Superseded variants** (geo\_transformer, geo\_karcher, alternate A2 forks, etc.) live on
**closed unmerged** `origin/cursor/*` PR branches — not merged to `main`. See agent-store
ledger `experiment-cleanup-ledger-20261030.md`.

Trains / evaluates on Hugging Face `sapientinc/sudoku-extreme` (desk HF default cache;
do not delete `~/.cache/huggingface` dataset snapshots).

```bash
cd riemannian_sudoku
uv sync
uv run pytest
# smoke
uv run riemannian-sudoku train --train-size 1024 --val-size 128 --steps 200 --batch-size 32
# 10k A2 desk run (locked hparams)
bash scripts/bipartite_sn_tm_a2_y0_logit_10k.sh
# full 50k (defaults: bipartite_sn_tm, checkpoints/bipartite-sn-tm-a2-y0-logit-full/…)
bash scripts/full_train.sh
# resume (mixer read from ckpt args unless --mixer override)
uv run riemannian-sudoku train --resume checkpoints/bipartite-sn-tm-a2-y0-logit-10k/bipartite-sn-tm-a2-y0-logit-s10000/last.pt ...
```

W&B project: `riemannian-sudoku`. Logs blank / exact-board / row / col / box on train+val.
