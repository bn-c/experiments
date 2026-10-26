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
| vocab / sequence | 12 / **seq 25** (demo-pair I→O) | 12 / **seq 25** (`--pack_mode demo_pair`) |
| trainable params | 6.83M (listed) | **~4.79M** (counted; 512-wide SwiGLU×2L ≈8.48M) |

Paper ratios kept (heads=8, MLP×4, L=2); width scaled so batch 512 + BPTT K=6 +
carry bank fit ~16GB. Override with `--d_model 512 --d_ff 2048` if you have headroom.

### Sequence layout (before → after)

| | Before (EX-33 packed) | After (default, paper) |
| --- | --- | --- |
| Mode | `--pack_mode packed` | `--pack_mode demo_pair` |
| Tokens | demos `in‖out` + query_in, PAD ≤128 | flatten(input) only, length **25** |
| Targets | query_out at query positions | flatten(output) on all 25 cells |
| Train rows | 1 packed sample / task×aug | **1 sample / demo-pair×aug** |
| Eval rows | packed demos+test query | test I→O only (25) |

Legacy packed path is kept: `--pack_mode packed --max_seq_len 128`.
Old EX-33 checkpoints are inferred as packed when `max_seq_len>25` and no
`pack_mode` was saved.

## Paper knobs matched

- **Solver:** per-sample η, patience P=5, γ=0.9, η_min=0.001, guarded Slerp,
  train `max_iter=8`, carry+detach across segments (no global learned η / restart-each-step)
- **Halt / eval:** chordal `max_t(1−⟨z,ẑ⟩)`, τ=0.01, eval `max_iter=200`
- **Reg:** weight drop 0.05, variational drop 0.2, EMA 0.999, Q-head loss 0.5,
  λ init **0.1**
- **Optim:** Muon lr **5e-3**, weight decay **0.1**, AdamW 1e-3, puzzle-emb lr/wd **1e-2 / 0**
- **Sequence:** demo-pair seq-25 + puzzle emb (train on demos; eval / vote on test)

## Desk knobs kept (intentionally not paper)

- Steps **12,000** (paper ~68,993)
- Aug **D₄×`n_color_perms=8`** (=64; paper 100×)
- Width 384 vs paper 512; global batch 512 vs paper 768

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
├── evaluate.py               # unaugmented chordal-halt Pass@1
├── eval_vote_augs.py         # D₄×color vote-over-augs (batched)
├── scripts/bench_compile.py  # eager vs torch.compile on model.step
└── run.sh
```

## Setup / run

Root venv (ROCm 7.1 nightlies + `triton-rocm`; Python 3.12-only clamp — see root README):

```bash
# from repo root
uv sync --extra rocm
```

Package-local:

```bash
cd sfprm_miniarc
pip install -r requirements.txt
./run.sh --smoke          # CPU-friendly
./run.sh                  # full 12k (needs ≥16GB GPU)
# optional: python train.py --compile  # torch.compile on model.step (default off)
# legacy packing: python train.py --pack_mode packed --max_seq_len 128
```

Eval:

```bash
python evaluate.py --checkpoint ./checkpoints/model_best.pt --tau 0.01 --max_loops 200
python eval_vote_augs.py --checkpoint ./checkpoints/model_best.pt   # vote-over-augs
# EX-33 packed ckpt (auto-inferred if max_seq_len=128 in args):
#   python evaluate.py --checkpoint ... --pack_mode packed --max_seq_len 128
```

### `torch.compile` microbench

One-off timing of paper `SFPRM.step(z, z_input, eta)` (static shapes, `no_grad` — **not** train throughput):

```bash
cd sfprm_miniarc
python scripts/bench_compile.py --batch 64 --d-model 384 --seq-len 26
```

Desk ROCm 7.1 (baseline tree, pre-paper API) saw ~2.1–2.4×; re-run after the paper `step` signature change before citing numbers.
