# Riemannian Sudoku

Peer spike: Sudoku solver on a Riemannian product manifold with leapfrog house
dynamics and Exp-map geodesic steps.

## Supported on `main` (desk default)

**Canonical mixer:** `--mixer bipartite_sn_tm` with **global board \(\mathbb{T}^{16}\)**
(tip default). Geometry: product \(S^8 \times S^{54} \leftrightarrow \mathbb{T}^{16}\)
(+ global \(\mathbb{T}^{16}\)). VF: learned S^8-only chart `w_x_chart` (9→18) + FiLM
`vf_x`; clues frozen on both \(x\) and \(c\). ~**681 877** params.

Tip 10k desk: W&B **`f7fv0ymk`** (blank@10k val **0.5375**). High-K test exact peaks
**0.075@K=64** (soft dip at K=128). Full write-up: [ARCHITECTURE.md](ARCHITECTURE.md).

Historical single-sphere blank bar: `yr1h706t` y0-logit (**0.5479** blank@8) —
higher blank@8, weaker high-K exact. Tip prioritizes product chart + high-K climb.

**Legacy mixers still in CLI** (resume / comparison): `euclid_axial`, `cross_coupled`,
`geo_*`.

**Superseded arms** (A2 \(S^{63}\), dendritic, y0-logit desks, Gram-era product
ladders, etc.) live on **closed** `origin/cursor/*` PR branches — not on `main`.

Trains / evaluates on Hugging Face `sapientinc/sudoku-extreme` (desk HF default cache;
do not delete `~/.cache/huggingface` dataset snapshots).

```bash
cd riemannian_sudoku
uv sync
uv run pytest
# smoke (defaults: bipartite_sn_tm + global torus)
uv run riemannian-sudoku train --train-size 1024 --val-size 128 --steps 200 --batch-size 32
# 10k tip desk (product + chart + global T^{16})
bash scripts/bipartite_sn_tm_product_chart_global_t16_10k.sh
# full 50k
bash scripts/full_train.sh
# Phase-1 ablation (no global z)
uv run riemannian-sudoku train --no-use-global-torus ...
# resume (mixer / use_global_torus read from ckpt args unless overridden)
uv run riemannian-sudoku train --resume checkpoints/bipartite-sn-tm-product-chart-global-t16-10k/bipartite-sn-tm-product-chart-global-t16-s10000/last.pt ...
```

W&B project: `riemannian-sudoku`. Logs blank / exact-board / row / col / box on train+val.
