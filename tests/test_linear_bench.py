"""Smoke: linear_bench train completes and writes a primary metric."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from linear_bench.data import HF_PRESETS, REGISTRY
from linear_bench.train import train_one, write_metrics
from linear_bench.wandb_util import (
    epoch_metric_payload,
    mlp_activation,
    resolve_wandb_enabled,
    run_name,
    run_tags,
)


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
    assert metrics["wandb"] is False  # smoke stays offline / no W&B
    assert cfg["optimizer"] == "AdamW"
    assert cfg["weight_decay"] == 0.01  # default when omitted
    assert metrics["elapsed_sec"] > 0
    assert "train_mse" in metrics["final"]
    assert "epoch_sec" in metrics["final"]
    assert metrics["final"]["epoch_sec"] > 0
    # epoch_sec is cumulative; final ≈ full wall (same clock).
    assert abs(metrics["final"]["epoch_sec"] - metrics["elapsed_sec"]) < 0.5
    # synthetic_lrb is intentionally local (no Hub corpus)
    assert "huggingface" not in str(metrics["meta"].get("source", ""))
    # Per-epoch repr probes (default on); LRB also emits feature_align.
    assert metrics["repr_analysis"] is True
    for key in (
        "repr/weight_eff_rank",
        "repr/weight_rank_ratio",
        "repr/act_participation_ratio",
        "repr/dead_frac",
        "repr/feature_align",
    ):
        assert key in metrics["history"][0]
        assert key in metrics["final"]
        assert isinstance(metrics["history"][0][key], float)
    assert "W_true" not in metrics["meta"]
    assert metrics["meta"].get("W_true_shape") == [16, 24]

    out = tmp_path / "metrics.json"
    write_metrics(metrics, out)
    loaded = json.loads(out.read_text())
    assert "primary_value" in loaded
    assert loaded["final"]["mse"] == loaded["primary_value"]
    assert "train_mse" in loaded["history"][0]
    assert "epoch_sec" in loaded["history"][0]
    assert "repr/weight_eff_rank" in loaded["history"][0]
    assert "repr/feature_align" in loaded["final"]


def test_epoch_sec_is_cumulative():
    """``epoch_sec`` / W&B ``time/epoch_sec`` is wall from train start, not per-epoch Δ."""
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
        "model": {"name": "linear"},
        "epochs": 3,
        "batch_size": 16,
        "lr": 0.05,
        "seed": 0,
        "device": "cpu",
        "repr_analysis": False,
    }
    metrics = train_one(cfg)
    secs = [row["epoch_sec"] for row in metrics["history"]]
    assert len(secs) == 3
    assert secs[0] > 0
    assert secs[1] >= secs[0]
    assert secs[2] >= secs[1]
    assert abs(secs[-1] - metrics["elapsed_sec"]) < 0.5


def test_wandb_resolve_default_off_and_cli_overrides():
    assert resolve_wandb_enabled({}, None, False) is False
    assert resolve_wandb_enabled({"wandb": True}, None, False) is True
    assert resolve_wandb_enabled({"wandb": True}, None, True) is False
    assert resolve_wandb_enabled({}, True, False) is True
    assert resolve_wandb_enabled({"wandb": False}, True, False) is True


def test_wandb_run_name_and_tags():
    assert run_name("mnist", "linear") == "mnist-linear"
    assert run_name("cifar10", "mlp", "relu") == "cifar10-mlp-relu"
    assert run_name("mnist", "mlp", "gelu") == "mnist-mlp-gelu"
    assert run_tags(dataset="mnist", model="linear", task="classification", seed=0) == [
        "mnist",
        "linear",
        "classification",
        "seed-0",
    ]
    assert run_tags(
        dataset="cifar10",
        model="mlp",
        task="classification",
        seed=0,
        activation="silu",
    ) == ["cifar10", "mlp", "classification", "seed-0", "silu"]


def test_mlp_activation_from_cfg():
    assert mlp_activation({"model": {"name": "linear"}}, "linear") is None
    assert mlp_activation({"model": {"name": "mlp", "activation": "GELU"}}, "mlp") == "gelu"
    assert mlp_activation({"model": {"name": "mlp"}}, "mlp") == "relu"


def test_epoch_metric_payload_classification_keys():
    payload = epoch_metric_payload(
        train_loss=0.5,
        train_metrics={"loss": 0.4, "acc": 0.9},
        val_metrics={"loss": 0.6, "acc": 0.8},
        primary="acc",
        epoch_sec=1.25,
        extra={"repr/weight_eff_rank": 2.5, "repr/dead_frac": 0.0},
    )
    assert payload == {
        "train/loss": 0.5,
        "train/acc": 0.9,
        "val/loss": 0.6,
        "val/acc": 0.8,
        "time/epoch_sec": 1.25,
        "repr/weight_eff_rank": 2.5,
        "repr/dead_frac": 0.0,
    }
    assert not any(k.startswith("summary/") for k in payload)


def test_repr_smoke_epoch_emits_keys():
    """One smoke epoch must emit core ``repr/`` keys (classification path, no LRB)."""
    cfg = {
        "dataset": "mnist1d",
        "data": {"source": "synthetic", "n_train": 64, "n_test": 32, "seed": 2},
        "model": {"name": "linear"},
        "epochs": 1,
        "batch_size": 32,
        "lr": 0.05,
        "seed": 2,
        "device": "cpu",
    }
    metrics = train_one(cfg)
    row = metrics["history"][0]
    for key in (
        "repr/weight_eff_rank",
        "repr/weight_rank_ratio",
        "repr/act_participation_ratio",
        "repr/dead_frac",
    ):
        assert key in row
        assert isinstance(row[key], float)
    # No ground-truth dictionary on mnist1d → feature_align omitted.
    assert "repr/feature_align" not in row
    assert 0.0 <= row["repr/dead_frac"] <= 1.0
    assert row["repr/weight_rank_ratio"] >= 0.0


def test_epoch_metric_payload_regression_keys():
    payload = epoch_metric_payload(
        train_loss=0.12,
        train_metrics={"loss": 0.11, "mse": 0.11},
        val_metrics={"loss": 0.15, "mse": 0.15},
        primary="mse",
        epoch_sec=0.5,
    )
    assert set(payload) == {
        "train/loss",
        "train/mse",
        "val/loss",
        "val/mse",
        "time/epoch_sec",
    }
    assert payload["train/mse"] == 0.11
    assert payload["val/mse"] == 0.15


def test_mnist1d_synthetic_offline_smoke():
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
    assert metrics["meta"]["source"] == "synthetic"
    assert metrics["primary_metric"] == "acc"
    assert 0.0 <= float(metrics["primary_value"]) <= 1.0
    assert "train_acc" in metrics["final"]
    assert 0.0 <= float(metrics["final"]["train_acc"]) <= 1.0
    assert metrics["elapsed_sec"] > 0


def test_mnist1d_hf_default_smoke():
    """Default mnist1d source is Hub christopher/mnist1d."""
    cfg = {
        "dataset": "mnist1d",
        "data": {"max_train": 64, "max_test": 32},  # source defaults to hf
        "model": {"name": "linear"},
        "epochs": 1,
        "batch_size": 32,
        "lr": 0.05,
        "seed": 0,
        "device": "cpu",
    }
    metrics = train_one(cfg)
    assert metrics["meta"]["source"] == "huggingface"
    assert metrics["meta"]["hf_path"] == "christopher/mnist1d"
    assert metrics["meta"]["n_train"] == 64
    assert metrics["meta"]["in_dim"] == 40
    assert metrics["primary_metric"] == "acc"
    assert 0.0 <= float(metrics["primary_value"]) <= 1.0


def test_hf_presets_registered():
    for name in ("cifar10", "cifar100", "mnist", "svhn", "stl10", "fashion_mnist", "hf", "mnist1d"):
        assert name in REGISTRY
    assert set(HF_PRESETS) >= {"cifar10", "cifar100", "mnist", "svhn", "stl10"}


@pytest.mark.parametrize("dataset", ["mnist", "cifar10"])
def test_hf_domain_smoke(dataset: str):
    """Real HF download+train on ≥2 vision domains; tiny subsets keep it CPU-fast."""
    cfg = {
        "dataset": dataset,
        "data": {"max_train": 64, "max_test": 32},
        "model": {"name": "linear"},
        "epochs": 1,
        "batch_size": 32,
        "lr": 0.05,
        "seed": 0,
        "device": "cpu",
    }
    metrics = train_one(cfg)
    assert metrics["meta"]["source"] == "huggingface"
    assert metrics["meta"]["n_train"] == 64
    assert metrics["meta"]["n_test"] == 32
    assert metrics["primary_metric"] == "acc"
    assert 0.0 <= float(metrics["primary_value"]) <= 1.0
    assert metrics["n_params"] > 0
