"""Optional Weights & Biases logging (SFPRM-style soft import).

Enable with ``--wandb`` or config ``"wandb": true``. Default is off so smoke/CI
never require the package or an API key. Without ``WANDB_API_KEY``, init uses
offline mode (local ``wandb/`` dir). ``--no-wandb`` / ``"wandb": false`` disable.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

try:
    import wandb

    _WANDB = True
except ImportError:  # pragma: no cover - optional dep
    wandb = None  # type: ignore[assignment]
    _WANDB = False

DEFAULT_PROJECT = "linear-bench"


def resolve_wandb_enabled(cfg: dict[str, Any], cli_wandb: bool | None, cli_no_wandb: bool) -> bool:
    """CLI overrides config; default off (smoke-safe)."""
    if cli_no_wandb:
        return False
    if cli_wandb is True:
        return True
    if "wandb" in cfg:
        return bool(cfg["wandb"])
    return False


def maybe_init_wandb(
    cfg: dict[str, Any],
    *,
    dataset: str,
    model_name: str,
    task: str,
    n_params: int,
) -> bool:
    """Start a W&B run when ``cfg['wandb']`` is true. Returns True if logging is active."""
    if not bool(cfg.get("wandb")):
        return False
    if not _WANDB:
        print("wandb not installed; continuing without logging", flush=True)
        return False

    proj = str(cfg.get("wandb_project") or DEFAULT_PROJECT)
    ent = cfg.get("wandb_entity")
    mode = os.environ.get("WANDB_MODE")
    if mode is None:
        mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"

    logged = {
        k: v
        for k, v in cfg.items()
        if k not in ("wandb", "wandb_project", "wandb_entity", "out")
    }
    logged.update(
        {
            "dataset": dataset,
            "model_type": model_name,
            "task": task,
            "num_params": n_params,
        }
    )

    name = f"linear_bench-{dataset}-{model_name}-{datetime.now(tz=UTC).strftime('%Y%m%d-%H%M%S')}"
    wandb.init(
        project=proj,
        entity=ent,
        name=name,
        mode=mode,
        config=logged,
    )
    wandb.log({"num_params": n_params, "dataset": dataset, "model_type": model_name}, step=0)
    return True


def log_epoch(
    *,
    epoch: int,
    train_loss: float,
    val_metrics: dict[str, float],
    dataset: str,
    model_type: str,
    primary: str,
) -> None:
    """Log one epoch: train/val loss + val acc (classification) or mse (LRB)."""
    if not _WANDB or wandb.run is None:
        return
    payload: dict[str, Any] = {
        "epoch": epoch,
        "train/loss": train_loss,
        "val/loss": float(val_metrics["loss"]),
        "dataset": dataset,
        "model_type": model_type,
    }
    if primary == "acc" and "acc" in val_metrics:
        payload["val/acc"] = float(val_metrics["acc"])
    if primary == "mse" and "mse" in val_metrics:
        payload["val/mse"] = float(val_metrics["mse"])
    wandb.log(payload, step=epoch)


def finish_wandb() -> None:
    if not _WANDB or wandb.run is None:
        return
    wandb.finish()
