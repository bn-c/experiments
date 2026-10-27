# linear_bench

Cheap harness for swapping **linear** vs **MLP** formulations and running short
train/eval loops on CPU (seconds–minutes). Not a research-paper clone of every
benchmark — just enough to iterate across formulations and domains.

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

## Layout

```text
linear_bench/
├── __main__.py          # python -m linear_bench
├── train.py             # config → train → metrics
├── models/              # registry: linear, mlp
├── data/                # mnist1d, synthetic_lrb, hf_vision
└── configs/             # smoke + small defaults
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
