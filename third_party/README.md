# Third-party vendored projects

These trees are **not** part of the `drf-vit` Python package (`pyproject.toml` only ships `drf` and `fracsrf`). Each project has its own `uv` environment and lifecycle.

## Enso (`enso/`)

[Enso](https://github.com/MVPandey/Enso) — Sudoku JEPA / energy-based model replication (vendored as a git submodule).

```bash
cd third_party/enso
uv sync
uv run python -m ebm.main train --n-samples 100000 --epochs 20
```

Setup (Kaggle dataset token, Python 3.13): see the Project Context doc `docs/enso-setup.md` in Agent Store, or Enso's `README.md`.
