# TRM Sudoku (staging)

Experiment package for [**Tiny Recursive Models**](https://arxiv.org/abs/2510.04871) (*Less is More: Recursive Reasoning with Tiny Networks*, Jolicoeur-Martineau et al., 2025) on Hugging Face [`sapientinc/sudoku-extreme`](https://huggingface.co/datasets/sapientinc/sudoku-extreme).

This directory is a **staging area**: the code mirrors the paper’s TRM-MLP loop and training recipe structure, but we have **not** reproduced the reported **~87.4%** Sudoku-Extreme test accuracy yet.

## Paper defaults (Sudoku-Extreme)

| Setting | Value |
| --- | --- |
| Backbone | 2-layer **MLP-Mixer** (no self-attention), hidden 512 |
| Recursion | `n=6` latent updates `z ← f(x,y,z)`, then `y ← f(y,z)` |
| `deep_recursion` | `T=3` blocks: `T−1` under `no_grad`, then one full backprop block |
| Deep supervision | `N_sup=16` outer improvement steps per batch item |
| EMA | decay **0.999** on weights (eval uses EMA shadow) |
| Train pool | **1K** base puzzles (paper); test **423K** official HF test split |
| Augmentation | **1000 rule-preserving shuffles per puzzle** in the paper |

### v1 simplifications

- **Augmentation**: we load the first `--train-size` rows (default **1000**) from the HF train CSV only. We do **not** apply 1000× Sudoku-preserving permutations per puzzle yet; that is the main gap vs. the paper’s small-sample regime.
- **ACT / Q-head**: omitted (paper’s adaptive halting and extra forward pass are not implemented).
- **Stable-max loss**: standard cross-entropy on **empty** cells only.
- **Training length**: smoke-friendly defaults (`--tiny`); full **60k epochs**, batch **768**, and LR warmup are documented but not wired as defaults.

## Layout

```
trm/
  pyproject.toml
  uv.lock
  README.md
  src/trm/
    main.py          # CLI: train | eval
    config.py
    dataset/         # HF sudoku-extreme loader (copied minimal pattern from enso/)
    model/           # TRM-MLP + MLP-Mixer + EMA
    training/
    eval/
  tests/
```

## Setup

From `trm/`:

```bash
# CPU / generic CUDA wheels from PyPI
uv sync

# AMD desk (ROCm nightly index, same pattern as enso/)
uv sync --extra rocm
```

Optional `.env` variables (copy from `.env.example`):

- `WANDB_API_KEY` / `WANDB_PROJECT` / `WANDB_ENTITY` — experiment tracking (same names as [`../enso/`](../enso/))

## Smoke commands

**CPU smoke train** (tiny model, 32 puzzles, 1 epoch):

```bash
cd trm
uv run python -m trm.main train --tiny
```

**ROCm smoke train** (after `uv sync --extra rocm`):

```bash
cd trm
uv run python -m trm.main train --tiny
```

**Eval** on HF `test` split (requires a checkpoint from `train`):

```bash
cd trm
uv run python -m trm.main train --tiny   # writes checkpoints/checkpoint.last.pt
uv run python -m trm.main eval --checkpoint checkpoints/checkpoint.last.pt --tiny --n-samples 64
```

Override paper recursion hyperparameters:

```bash
uv run python -m trm.main train --n 6 --T 3 --max-steps 16 --train-size 1000
```

## Tests

```bash
cd trm
uv run pytest
```

## Experiment tracking

Training metrics are logged to [Weights & Biases](https://wandb.ai) when `WANDB_API_KEY` (and project/entity) are set in `.env`:

- Per-step: `train/loss` (deep-supervision sum per batch)
- Per-epoch: `train/epoch_loss`
- Final checkpoint uploaded as a W&B model artifact when a run is active

Runs are named `trm-sudoku-<UTC timestamp>`. Without W&B credentials, training proceeds unchanged.

## References

- Paper: [arXiv:2510.04871](https://arxiv.org/abs/2510.04871)
- Dataset: [sapientinc/sudoku-extreme](https://huggingface.co/datasets/sapientinc/sudoku-extreme)
- Sibling experiment: [`../enso/`](../enso/) (Sudoku EBM / JEPA — different approach)
