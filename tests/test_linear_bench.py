"""Smoke: linear_bench train completes and writes a primary metric."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from linear_bench.data import HF_PRESETS, REGISTRY
from linear_bench.train import train_one, write_metrics
from linear_bench.wandb_util import (
    epoch_metric_payload,
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
    # synthetic_lrb is intentionally local (no Hub corpus)
    assert "huggingface" not in str(metrics["meta"].get("source", ""))

    out = tmp_path / "metrics.json"
    write_metrics(metrics, out)
    loaded = json.loads(out.read_text())
    assert "primary_value" in loaded
    assert loaded["final"]["mse"] == loaded["primary_value"]
    assert "train_mse" in loaded["history"][0]
    assert "epoch_sec" in loaded["history"][0]


def test_wandb_resolve_default_off_and_cli_overrides():
    assert resolve_wandb_enabled({}, None, False) is False
    assert resolve_wandb_enabled({"wandb": True}, None, False) is True
    assert resolve_wandb_enabled({"wandb": True}, None, True) is False
    assert resolve_wandb_enabled({}, True, False) is True
    assert resolve_wandb_enabled({"wandb": False}, True, False) is True


def test_wandb_run_name_and_tags():
    assert run_name("mnist", "linear") == "mnist-linear"
    assert run_name("cifar10", "mlp") == "cifar10-mlp"
    assert run_tags(dataset="mnist", model="linear", task="classification", seed=0) == [
        "mnist",
        "linear",
        "classification",
        "seed-0",
    ]


def test_epoch_metric_payload_classification_keys():
    payload = epoch_metric_payload(
        train_loss=0.5,
        train_metrics={"loss": 0.4, "acc": 0.9},
        val_metrics={"loss": 0.6, "acc": 0.8},
        primary="acc",
        epoch_sec=1.25,
    )
    assert payload == {
        "train/loss": 0.5,
        "train/acc": 0.9,
        "val/loss": 0.6,
        "val/acc": 0.8,
        "time/epoch_sec": 1.25,
    }
    assert not any(k.startswith("summary/") for k in payload)


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
