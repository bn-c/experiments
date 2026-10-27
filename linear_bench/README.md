# linear_bench

Cheap harness for swapping **linear** vs **MLP** formulations and running short
train/eval loops on CPU (seconds–minutes). Not a research-paper clone of every
benchmark — just enough to iterate across formulations and domains.

## Quick smoke

From repo root (needs `torch`, `numpy`, `datasets`, `pillow` — already in the
root env):

```bash
python -m linear_bench --config linear_bench/configs/smoke_mnist1d.json
python -m linear_bench --config linear_bench/configs/smoke_synthetic.json
# HuggingFace multi-domain (downloads+caches on first run)
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

## Datasets

| name | task | source / knobs |
| --- | --- | --- |
| `mnist1d` | classification (10-way, length-40) | `source` (`official`\|`synthetic`), `max_train`, `max_test`, `cache` |
| `synthetic_lrb` | regression: recover sparse `z` from `x = A z + ε` | `n`, `n_p`, `sparsity`, `noise`, `n_train`, `n_test` |
| `cifar10` / `cifar100` / `mnist` / `svhn` / `stl10` / `fashion_mnist` | HF vision classification (flattened) | `max_train`, `max_test`, `resize`, … |
| `hf` | same, arbitrary Hub dataset | `hf_path`, optional `hf_name`, `image_key`, `label_key`, `n_classes`, … |

`synthetic_lrb` is LRB-inspired (controllable ambient dim, feature count,
sparsity, superposition when `n_p > n`). It is **not** a reimplementation of
the full Linear Representation Benchmark.

`stl10` stands in for ImageNette-scale RGB (96→32 by default) — classic
ImageNette Hub scripts are no longer loadable via current `datasets`.

### Pointing at HuggingFace datasets

**Preset** (Hub id resolved for you):

```json
{
  "dataset": "cifar10",
  "data": { "max_train": 512, "max_test": 128 }
}
```

**Arbitrary Hub dataset** via the generic `hf` entry — use `namespace/name`
ids (current `datasets` rejects bare names like `cifar10`):

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

Or override a preset’s path:

```json
{
  "dataset": "mnist",
  "data": { "hf_path": "ylecun/mnist", "max_train": 128, "max_test": 64 }
}
```

Images are resized (optional), cast to float `[0,1]`, flattened for linear/MLP,
and train-normalized. First run caches under the HuggingFace hub cache.

Preset Hub ids:

| preset | `hf_path` | notes |
| --- | --- | --- |
| `cifar10` | `uoft-cs/cifar10` | `img` / `label` |
| `cifar100` | `uoft-cs/cifar100` | `fine_label` |
| `mnist` | `ylecun/mnist` | grayscale kept as L |
| `svhn` | `ufldl-stanford/svhn` | config `cropped_digits` |
| `stl10` | `tanganke/stl10` | default `resize: 32` |
| `fashion_mnist` | `zalando-datasets/fashion_mnist` | grayscale |

## Optional later: NAS-Bench-201 / LEMUR 2

- **NAS-Bench-201**: zero-GPU table lookup for cell architectures. Skip as a
  hard dep here. To plug later, wrap the official API (or a cached accuracy
  table) behind a tiny `score_arch(config) -> metrics` helper — no training
  loop needed.
- **LEMUR 2**: out of scope for this harness; revisit if a tiny offline slice
  appears.

## Test

```bash
python -m pytest tests/test_linear_bench.py -q
```

HF-domain cases need network (or a warm HF cache) on first run.
