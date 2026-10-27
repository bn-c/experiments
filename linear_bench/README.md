# linear_bench

Cheap harness for swapping **linear** vs **MLP** formulations and running short
train/eval loops on CPU (seconds–minutes). Not a research-paper clone of every
benchmark — just enough to iterate.

## Quick smoke

From repo root (needs `torch` + `numpy`, already in the root env):

```bash
python -m linear_bench --config linear_bench/configs/smoke_mnist1d.json
python -m linear_bench --config linear_bench/configs/smoke_synthetic.json
```

Metrics print as JSON and (if `out` is set) land under `outputs/linear_bench/`.
Override seed/epochs/device without editing the file:

```bash
python -m linear_bench --config linear_bench/configs/smoke_mnist1d.json --epochs 1 --seed 1
```

Official MNIST-1D (downloads+caches `data/mnist1d_data.pkl` once):

```bash
python -m linear_bench --config linear_bench/configs/mnist1d_linear.json
```

## Layout

```text
linear_bench/
├── __main__.py          # python -m linear_bench
├── train.py             # config → train → metrics
├── models/              # registry: linear, mlp
├── data/                # mnist1d, synthetic_lrb
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

| name | task | knobs |
| --- | --- | --- |
| `mnist1d` | classification (10-way, length-40) | `source` (`official`\|`synthetic`), `max_train`, `max_test`, `cache` |
| `synthetic_lrb` | regression: recover sparse `z` from `x = A z + ε` | `n`, `n_p`, `sparsity`, `noise`, `n_train`, `n_test` |

`synthetic_lrb` is LRB-inspired (controllable ambient dim, feature count,
sparsity, superposition when `n_p > n`). It is **not** a reimplementation of
the full Linear Representation Benchmark.

## Optional later: NAS-Bench-201 / LEMUR 2

- **NAS-Bench-201**: zero-GPU table lookup for cell architectures. Skip as a
  hard dep here. To plug later, wrap the official API (or a cached accuracy
  table) behind `linear_bench.data` / a tiny `score_arch(config) -> metrics`
  helper and call it from the CLI with `--lookup` — no training loop needed.
- **LEMUR 2**: out of scope for this harness; revisit if a tiny offline slice
  appears.

## Test

```bash
python -m pytest tests/test_linear_bench.py -q
```
