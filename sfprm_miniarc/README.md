# SFPRM Mini-ARC (paper-faithful)

Spherical Fixed-Point Reasoner (weight-tied nGPT on \(S^{d-1}\)) on Mini-ARC,
aligned with [arXiv:2610.25646](https://arxiv.org/abs/2610.25646) Appendix B /
Table 3, scaled for a **1×16GB GPU** desk run.

## Dims (16GB desk vs paper)

| | Paper Mini-ARC | This desk default |
| --- | --- | --- |
| hidden / heads / MLP | 512 / 8 / ×4 | **384 / 8 / ×4** (`d_ff=1536`) |
| L_layers | 2 | 2 |
| SwiGLU, RoPE, causal conv k=4 | yes | yes |
| puzzle-emb length | 1 | 1 |
| vocab / packing | 12 / seq 25 | 12 / **packed demos+query ≤128** (desk) |
| trainable params | 6.83M (listed) | **~4.79M** (counted; 512-wide SwiGLU×2L ≈8.48M) |

Paper ratios kept (heads=8, MLP×4, L=2); width scaled so batch 512 + BPTT K=6 +
carry bank fit ~16GB. Override with `--d_model 512 --d_ff 2048` if you have headroom.

## Paper knobs matched

- **Solver:** per-sample η, patience P=5, γ=0.9, η_min=0.001, guarded Slerp,
  train `max_iter=8`, carry+detach across segments (no global learned η / restart-each-step)
- **Halt / eval:** chordal `max_t(1−⟨z,ẑ⟩)`, τ=0.01, eval `max_iter=200`
- **Reg:** weight drop 0.05, variational drop 0.2, EMA 0.999, Q-head loss 0.5,
  λ init **0.1**
- **Optim:** Muon lr **5e-3**, weight decay **0.1**, AdamW 1e-3, puzzle-emb lr/wd **1e-2 / 0**

## Desk knobs kept (intentionally not paper)

- Steps **12,000** (paper ~68,993)
- Aug **149×64** via D₄×`n_color_perms=8` + current packing (paper 149×100, seq 25)

## Remaining gaps vs paper

- **Vote-over-augs** Pass@1 (paper B.1/B.4) not implemented — eval is unaugmented task Pass@1
- Packed multi-demo context (len≤128) vs paper’s 25-cell + emb formulation (PAD slots are masked in attn + chordal residual)
- Width 384 vs paper 512 (16GB); full-width flag available
- Global batch 512 vs paper 768
- γ_eval=0.9 (paper Mini-ARC train/eval both 0.9) — OK; Sudoku uses 0.99 at eval

## Layout

```text
sfprm_miniarc/
├── data/download_miniarc.py
├── data/dataset.py
├── models/spherical_ops.py   # guarded Slerp, chordal residual
├── models/sfprm_core.py      # nGPT block + SFPRM
├── models/solver.py          # patience η schedule
├── models/ema.py
├── optim/muon.py
├── train.py
├── evaluate.py
└── run.sh
```

## Setup / run

```bash
cd sfprm_miniarc
pip install -r requirements.txt
./run.sh --smoke          # CPU-friendly
./run.sh                  # full 12k (needs ≥16GB GPU)
# optional: python train.py --compile  # torch.compile on model.step (default off)
```

Eval:

```bash
python evaluate.py --checkpoint ./checkpoints/model_best.pt --tau 0.01 --max_loops 200
```
