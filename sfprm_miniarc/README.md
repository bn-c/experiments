# SFPRM Mini-ARC baseline

Spherical Fixed-Point Reasoner Model (weight-tied nGPT on \(S^{d-1}\)) trained on
[Mini-ARC](https://github.com/KSB21ST/MINI-ARC) (149 × 5×5 ARC-style tasks).

Architecture matches the execution guide (`d_model=512`, `d_ff=1536`, one
weight-tied block) → **~2.70M params** (the guide’s “~6.83M” figure is
inconsistent with those dims). Target (single ≥16GB GPU, ~1.5–3h): **~48–53%
Pass@1** with adaptive chordal halting at eval.

## Layout

```text
sfprm_miniarc/
├── data/download_miniarc.py   # pulls KSB21ST/MINI-ARC data/MiniARC
├── data/dataset.py            # D₄ + color-perm augs, in-device cache
├── models/spherical_ops.py    # slerp / project / geodesic damping
├── models/sfprm_core.py       # weight-tied NormalizedTransformerBlock + SFPRM
├── optim/muon.py              # Newton–Schulz Muon
├── train.py                   # truncated BPTT, Muon+AdamW, W&B
├── evaluate.py                # chordal halting Pass@1
└── run.sh
```

## Setup

```bash
cd sfprm_miniarc
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Full train (12k steps)

```bash
./run.sh
# or:
python data/download_miniarc.py
python train.py --steps 12000 --bptt_k 6 --batch_size 512 --compile --out_dir ./checkpoints
python evaluate.py --checkpoint ./checkpoints/model_best.pt --epsilon 1e-3 --max_loops 100
```

Acceptance (guide): ≤3.5h on 4080/3090/V100, ≤6.5GB VRAM, ≥48% Pass@1, mean
halt steps roughly 15–35.

## Smoke test (CPU/CUDA, tiny)

```bash
./run.sh --smoke
# or:
python train.py --steps 3 --bptt_k 2 --batch_size 4 --max_tasks 4 --eval_every 3 --no-wandb
```

## Design notes / gap fills

### Mini-ARC source

Downloaded from **https://github.com/KSB21ST/MINI-ARC** (`data/MiniARC/`, 149
JSON files, standard ARC `{train,test}` schema). See `data/download_miniarc.py`.

### Sequence packing

`vocab_size=11`: colors `0–9`, **`PAD_ID=10`**. Each sample is:

`[demo₀_in ‖ demo₀_out ‖ … ‖ demoₖ_in ‖ demoₖ_out ‖ query_in ‖ PAD…]`

with `k ≤ 2` so length ≤ 128. **`query_mask`** covers the 25 query-input
positions; the head predicts the query **output** colors at those positions
(in-place transform). Color `0` is treated as background and is not permuted;
D₄ uses 4 rotations × optional left-right flip (8).

### Optimizer split

- **Muon** (`lr=0.02`, momentum `0.95`): 2D `Linear` weights inside `core_block` only.
- **AdamW** (`lr=1e-3`, `wd=1e-2`): `token_emb`, `pos_emb`, `head`, `raw_lambda`, `raw_eta`.

Embeddings/head are 2D but stay on AdamW (standard Muon practice). Muon uses
Keller Jordan Newton–Schulz5 orthogonalization (`optim/muon.py`).

### W&B

Project default: `sfprm-miniarc` (`--wandb-project`).

- Enabled automatically when `WANDB_API_KEY` is set.
- `--wandb` forces on (offline mode if no key); `--no-wandb` disables.
- Logs `train/loss`, `train/lr_*`, `train/lambda`, `train/eta`, and periodic
  `eval/pass_at_1`, `eval/mean_halt_steps`. Best ckpt by eval Pass@1.

Mirrors the optional W&B pattern used in `trm/` / `enso/` (init + metric keys +
artifact upload), with CLI flags instead of pydantic config.
