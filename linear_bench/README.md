# linear_bench

Cheap harness for swapping **linear** vs **MLP** formulations and running short
train/eval loops on CPU (seconds–minutes). Not a research-paper clone of every
benchmark — just enough to iterate across formulations and domains.

**Optimizer:** AdamW (`lr` + optional `weight_decay`, default `0.01`). Both land in
W&B config as `optimizer` / `weight_decay` when logging is on.

**Policy:** prefer HuggingFace Hub for every corpus that has a real Hub source.
Only `synthetic_lrb` stays in-repo (controlled ground truth; no Hub dataset).

## Quick smoke

From repo root (needs `torch`, `numpy`, `datasets`, `pillow` — already in the
root env):

```bash
python -m linear_bench --config linear_bench/configs/smoke_mnist1d.json   # HF christopher/mnist1d
python -m linear_bench --config linear_bench/configs/smoke_synthetic.json # in-repo generator
python -m linear_bench --config linear_bench/configs/smoke_hf_mnist.json
python -m linear_bench --config linear_bench/configs/smoke_hf_cifar10.json
```

Metrics print as JSON and (if `out` is set) land under `outputs/linear_bench/`.
Override seed/epochs/device without editing the file:

```bash
python -m linear_bench --config linear_bench/configs/smoke_hf_mnist.json --epochs 1 --seed 1
```

## Weights & Biases

Opt-in (default **off** — smoke/CI never need W&B). Same soft-import pattern as
SFPRM: install `wandb` when you want logging; without `WANDB_API_KEY` runs go
**offline** (local `wandb/` dir). One project (`linear-bench`); filter by
domain/model via run metadata — not separate projects.

```bash
# flag
python -m linear_bench --config linear_bench/configs/mnist1d_linear.json --wandb

# or in JSON: "wandb": true, optional "wandb_project": "linear-bench"
python -m linear_bench --config path/to.json   # if config has wandb: true

# disable even if config enables
python -m linear_bench --config path/to.json --no-wandb

# project / entity / offline
WANDB_API_KEY=… python -m linear_bench --config … --wandb --wandb-project linear-bench
WANDB_MODE=offline python -m linear_bench --config … --wandb
```

### How to read the dashboard

| Axis | Value | Use |
| --- | --- | --- |
| **name** | `{dataset}-{model}` e.g. `mnist-linear` | Scan the run list |
| **group** | dataset (`mnist`, `cifar10`, …) | Compare models within one domain |
| **job_type** | model (`linear`, `mlp`) | Filter formulation |
| **tags** | dataset, model, task, `seed-N` | Sidebar filters |
| **config** | dataset, model, seed, task, hyperparams | Run detail / parallel coords |
| **metrics** | `train/loss`, `train/acc`\|`train/mse`, `val/loss`, `val/acc`\|`val/mse`, `time/epoch_sec`, `repr/*`; summary: `summary/train_*`, `summary/val_*`, `summary/wall_sec` | Shared chart keys across runs |

Domain and model live in name/tags/group/config — **not** as metric keys — so
charts do not mix unrelated series. Default project: `linear-bench`.

## Representation / DoF probes (`repr/*`)

Logged **every epoch** after val (default **on**; set `"repr_analysis": false`
to skip). Cheap SVD on the primary weight + a fixed probe batch (cap
`repr_max_samples`, default 256). Schema-only keys — domain still via
group/tags.

| key | meaning |
| --- | --- |
| `repr/weight_eff_rank` | Participation ratio of singular values of the primary weight (`linear`: that `Linear`; `mlp`: **last** Linear before logits) |
| `repr/weight_rank_ratio` | `eff_rank / min(rows, cols)` |
| `repr/act_participation_ratio` | Participation ratio of centered probe-batch features (`linear`: outputs; `mlp`: penultimate post-activation) |
| `repr/dead_frac` | Fraction of units with mean \|act\| ≈ 0 on the probe batch |
| `repr/feature_align` | **LRB only** when meta has `W_true`/`A`: mean abs row-cosine of weight vs `W_true.T` (else linear CKA vs continuous codes). Omitted on real vision sets |

JSON history/final carry the same `repr/*` keys; `meta` drops the raw `W_true`
matrix (keeps `W_true_shape`).

## Baseline configs (not smoke)

Full grid: **every domain × {linear, mlp}**. Paths listed in
`configs/baseline_grid.txt` for the desk runner. All baselines set
`"wandb": true` (desk can still pass `--wandb` / `--no-wandb`).

| domain | linear | mlp | notes |
| --- | --- | --- | --- |
| `mnist1d` | `mnist1d_linear.json` | `mnist1d_mlp.json` | full HF MNIST-1D, 40 epochs |
| `mnist` | `mnist_linear.json` | `mnist_mlp.json` | full MNIST, 30 epochs |
| `fashion_mnist` | `fashion_mnist_linear.json` | `fashion_mnist_mlp.json` | full Fashion-MNIST, 30 epochs |
| `cifar10` | `cifar10_linear.json` | `cifar10_mlp.json` | 10k train / 2k test, 20 epochs |
| `svhn` | `svhn_linear.json` | `svhn_mlp.json` | 10k train / 2k test, 20 epochs |
| `synthetic_lrb` | `synthetic_lrb_linear.json` | `synthetic_lrb_mlp.json` | 8k train, 50 epochs |

Default knobs: linear `lr=0.01`; MLP `width=256`, `depth=2`, `relu`, `lr=0.001`.
Batch 128, seed 0, device cpu. Smoke JSON stays tiny + W&B off.

## Layout

```text
linear_bench/
├── __main__.py          # python -m linear_bench
├── train.py             # config → train → metrics
├── repr_analysis.py     # per-epoch weight/act/LRB probes
├── wandb_util.py        # optional W&B (soft import)
├── models/              # registry: linear, mlp
├── data/                # mnist1d, synthetic_lrb, hf_vision
└── configs/             # smoke + baselines
```

## Add a formulation

1. Create `linear_bench/models/my_thing.py` with
   `build_my_thing(in_dim, out_dim, kwargs) -> nn.Module`.
2. Register it in `linear_bench/models/__init__.py`:

```python
from .my_thing import build_my_thing
REGISTRY["my_thing"] = build_my_thing
```

3. Point a config at it:

```json
{ "model": { "name": "my_thing", "...": "knobs" } }
```

## Datasets: HF vs local

| name | default source | Hub id / notes |
| --- | --- | --- |
| `mnist1d` | **HF** | [`christopher/mnist1d`](https://huggingface.co/datasets/christopher/mnist1d) (Greydanus mirror). Fallbacks: `pickle` (GitHub pkl), `synthetic` (offline stand-in). |
| `mnist` | **HF** | `ylecun/mnist` |
| `cifar10` / `cifar100` | **HF** | `uoft-cs/cifar10`, `uoft-cs/cifar100` |
| `svhn` | **HF** | `ufldl-stanford/svhn` (`cropped_digits`) |
| `stl10` | **HF** | `tanganke/stl10` (ImageNette-scale stand-in; classic ImageNette scripts broken on current `datasets`) |
| `fashion_mnist` | **HF** | `zalando-datasets/fashion_mnist` |
| `hf` | **HF** | arbitrary `namespace/name` via `hf_path` |
| `synthetic_lrb` | **local only** | In-repo Bernoulli-Gaussian → `x = A z + ε`. No Hub dataset — knobs define ground truth. |

### MNIST-1D sources

```json
{ "dataset": "mnist1d", "data": { "source": "hf", "max_train": 256, "max_test": 64 } }
{ "dataset": "mnist1d", "data": { "source": "pickle" } }      // Greydanus GitHub pkl
{ "dataset": "mnist1d", "data": { "source": "synthetic", "n_train": 256, "n_test": 64 } }
```

Default is `hf`. If Hub load fails, the loader demotes to pickle, then synthetic.

### Pointing at HuggingFace datasets

**Preset:**

```json
{
  "dataset": "cifar10",
  "data": { "max_train": 512, "max_test": 128 }
}
```

**Arbitrary Hub dataset** via `hf` — use `namespace/name` ids (current
`datasets` rejects bare names like `cifar10`):

```json
{
  "dataset": "hf",
  "data": {
    "hf_path": "uoft-cs/cifar100",
    "image_key": "img",
    "label_key": "fine_label",
    "n_classes": 100,
    "max_train": 256,
    "max_test": 64
  }
}
```

Vision presets flatten images for linear/MLP (optional `resize`, train-normalize).
First run caches under the HuggingFace hub cache.

### Synthetic LRB (local)

Kept in-repo because the experiment *is* the controllable generator
(`n`, `n_p`, `sparsity`, `noise`, `seed`). There is no meaningful Hub “dataset”
for that ground-truth matrix. Prefer HF for every real corpus above.

## Optional later: NAS-Bench-201 / LEMUR 2

- **NAS-Bench-201**: zero-GPU table lookup — skip as a hard dep; plug later via
  a tiny `score_arch(config) -> metrics` helper.
- **LEMUR 2**: out of scope; revisit if a tiny offline slice appears.

## Test

```bash
python -m pytest tests/test_linear_bench.py -q
```

HF-domain cases need network (or a warm HF cache) on first run.
