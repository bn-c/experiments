"""Config → train → metrics JSON/stdout."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import load_dataset
from .models import build_model
from .wandb_util import finish_wandb, log_epoch, maybe_init_wandb


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _loss_and_metric(task: str) -> tuple[nn.Module, str]:
    if task == "classification":
        return nn.CrossEntropyLoss(), "acc"
    if task == "regression":
        return nn.MSELoss(), "mse"
    raise ValueError(f"unknown task {task!r}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, task: str, device: torch.device) -> dict[str, float]:
    model.eval()
    if task == "classification":
        correct = 0
        total = 0
        loss_sum = 0.0
        crit = nn.CrossEntropyLoss(reduction="sum")
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += float(crit(logits, y).item())
            pred = logits.argmax(dim=-1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
        return {"loss": loss_sum / max(total, 1), "acc": correct / max(total, 1)}

    # regression
    sse = 0.0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        sse += float(((pred - y) ** 2).sum().item())
        total += int(y.numel())
    mse = sse / max(total, 1)
    return {"loss": mse, "mse": mse}


def train_one(cfg: dict[str, Any]) -> dict[str, Any]:
    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    device_s = str(cfg.get("device", "cpu"))
    if device_s.startswith("cuda") and not torch.cuda.is_available():
        device_s = "cpu"
    device = torch.device(device_s)

    data_cfg = dict(cfg.get("data", {}))
    data_cfg.setdefault("seed", seed)
    dataset_name = str(cfg["dataset"])
    train_ds, test_ds, meta = load_dataset(dataset_name, data_cfg)
    task = str(meta["task"])

    batch_size = int(cfg.get("batch_size", 64))
    epochs = int(cfg.get("epochs", 2))
    lr = float(cfg.get("lr", 1e-2))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model_cfg = dict(cfg.get("model", {}))
    model_name = str(model_cfg.pop("name", "linear"))
    model = build_model(model_name, int(meta["in_dim"]), int(meta["out_dim"]), model_cfg).to(device)

    criterion, primary = _loss_and_metric(task)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters())

    wandb_on = maybe_init_wandb(
        cfg,
        dataset=dataset_name,
        model_name=model_name,
        task=task,
        n_params=n_params,
        seed=seed,
    )

    history: list[dict[str, float]] = []
    t0 = time.perf_counter()
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            running = 0.0
            n_ex = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                opt.step()
                running += float(loss.item()) * int(x.shape[0])
                n_ex += int(x.shape[0])
            train_loss = running / max(n_ex, 1)
            ev = evaluate(model, test_loader, task, device)
            row = {"epoch": float(epoch), "train_loss": train_loss, **ev}
            history.append(row)
            if wandb_on:
                log_epoch(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_metrics=ev,
                    primary=primary,
                )
    finally:
        if wandb_on:
            finish_wandb()

    elapsed = time.perf_counter() - t0
    final = history[-1] if history else {}
    metrics: dict[str, Any] = {
        "dataset": dataset_name,
        "model": model_name,
        "task": task,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "device": str(device),
        "elapsed_sec": round(elapsed, 4),
        "n_params": n_params,
        "meta": meta,
        "final": {k: v for k, v in final.items() if k != "epoch"},
        "primary_metric": primary,
        "primary_value": final.get(primary),
        "history": history,
        "wandb": bool(wandb_on),
    }
    return metrics


def write_metrics(metrics: dict[str, Any], out_path: Path | None) -> None:
    text = json.dumps(metrics, indent=2, default=str)
    print(text)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n")


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config must be a JSON object")
    return cfg
