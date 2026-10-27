"""Smoke: linear_bench train completes and writes a primary metric."""

from __future__ import annotations

import json
from pathlib import Path

from linear_bench.train import train_one, write_metrics


def test_smoke_train_writes_metric(tmp_path: Path):
    cfg = {
        "dataset": "synthetic_lrb",
        "data": {
            "n": 16,
            "n_p": 24,
            "sparsity": 0.2,
            "noise": 0.01,
            "n_train": 64,
            "n_test": 32,
            "seed": 0,
        },
        "model": {"name": "mlp", "width": 32, "depth": 1, "activation": "relu"},
        "epochs": 1,
        "batch_size": 16,
        "lr": 0.05,
        "seed": 0,
        "device": "cpu",
    }
    metrics = train_one(cfg)
    assert metrics["primary_metric"] == "mse"
    assert isinstance(metrics["primary_value"], float)
    assert metrics["epochs"] == 1
    assert metrics["n_params"] > 0

    out = tmp_path / "metrics.json"
    write_metrics(metrics, out)
    loaded = json.loads(out.read_text())
    assert "primary_value" in loaded
    assert loaded["final"]["mse"] == loaded["primary_value"]


def test_mnist1d_synthetic_smoke():
    cfg = {
        "dataset": "mnist1d",
        "data": {"source": "synthetic", "n_train": 64, "n_test": 32, "seed": 1},
        "model": {"name": "linear"},
        "epochs": 1,
        "batch_size": 32,
        "lr": 0.05,
        "seed": 1,
        "device": "cpu",
    }
    metrics = train_one(cfg)
    assert metrics["primary_metric"] == "acc"
    assert 0.0 <= float(metrics["primary_value"]) <= 1.0
