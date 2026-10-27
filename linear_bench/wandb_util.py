"""Optional Weights & Biases logging (SFPRM-style soft import).

Enable with ``--wandb`` or config ``"wandb": true``. Default is off so smoke/CI
never require the package or an API key. Without ``WANDB_API_KEY``, init uses
offline mode (local ``wandb/`` dir). ``--no-wandb`` / ``"wandb": false`` disable.

Dashboard UX (one project ``linear-bench``):

- **name** ``{dataset}-{model}`` (readable run list)
- **group** = dataset (compare models within a domain)
- **job_type** = model (filter linear vs mlp)
- **tags** = dataset, model, task, ``seed-{n}``
- **config** holds dataset / model / seed / task / hyperparams
  (incl. ``optimizer``, ``weight_decay``)
- **metrics** stay schema-only: ``train/loss``, ``train/acc``|``train/mse``,
  ``val/loss``, ``val/acc``|``val/mse``, ``time/epoch_sec`` (cumulative wall
  from train start), plus optional ``repr/*`` DoF probes — no domain/model
  keys in the chart series
- **summary** once at end: ``summary/train_acc``|``summary/train_mse``,
  matching ``summary/val_*``, plus ``summary/wall_sec`` for table compare
"""

from __future__ import annotations

import os
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


def run_name(dataset: str, model: str) -> str:
    """Display name: ``{dataset}-{model}``."""
    return f"{dataset}-{model}"


def run_tags(*, dataset: str, model: str, task: str, seed: int) -> list[str]:
    """Stable filter tags (domain, model, task, seed)."""
    return [dataset, model, task, f"seed-{seed}"]


def maybe_init_wandb(
    cfg: dict[str, Any],
    *,
    dataset: str,
    model_name: str,
    task: str,
    n_params: int,
    seed: int,
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
    # Keep nested model knobs; add flat scalars for Config / filters (not metrics).
    model_cfg = logged.get("model")
    if isinstance(model_cfg, dict):
        logged["model"] = {**model_cfg, "name": model_name}
    else:
        logged["model"] = {"name": model_name}
    logged.update(
        {
            "dataset": dataset,
            "model_name": model_name,
            "model_type": model_name,  # alias for older baseline runs
            "task": task,
            "seed": int(seed),
            "num_params": n_params,
        }
    )

    wandb.init(
        project=proj,
        entity=ent,
        name=run_name(dataset, model_name),
        group=dataset,
        job_type=model_name,
        tags=run_tags(dataset=dataset, model=model_name, task=task, seed=seed),
        mode=mode,
        config=logged,
    )
    return True


def epoch_metric_payload(
    *,
    train_loss: float,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    primary: str,
    epoch_sec: float | None = None,
    extra: dict[str, float] | None = None,
) -> dict[str, float]:
    """Flat ``train/*`` / ``val/*`` / ``time/*`` / ``repr/*`` keys (no domain/model in names)."""
    payload: dict[str, float] = {
        "train/loss": float(train_loss),
        "val/loss": float(val_metrics["loss"]),
    }
    if primary == "acc":
        if "acc" in train_metrics:
            payload["train/acc"] = float(train_metrics["acc"])
        if "acc" in val_metrics:
            payload["val/acc"] = float(val_metrics["acc"])
    if primary == "mse":
        # train/mse from eval pass (exact SSE/n); may differ slightly from train/loss
        if "mse" in train_metrics:
            payload["train/mse"] = float(train_metrics["mse"])
        if "mse" in val_metrics:
            payload["val/mse"] = float(val_metrics["mse"])
    if epoch_sec is not None:
        # Caller passes cumulative wall seconds since train start.
        payload["time/epoch_sec"] = float(epoch_sec)
    if extra:
        for k, v in extra.items():
            payload[str(k)] = float(v)
    return payload


def log_epoch(
    *,
    epoch: int,
    train_loss: float,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    primary: str,
    epoch_sec: float | None = None,
    extra: dict[str, float] | None = None,
) -> None:
    """Log one epoch: train/val loss + acc|mse, cumulative ``time/epoch_sec`` + ``repr/*``."""
    if not _WANDB or wandb.run is None:
        return
    wandb.log(
        epoch_metric_payload(
            train_loss=train_loss,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            primary=primary,
            epoch_sec=epoch_sec,
            extra=extra,
        ),
        step=epoch,
    )


def set_run_summary(
    *,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    primary: str,
    wall_sec: float,
) -> None:
    """Final summary scalars once for W&B table compare (metrics + ``summary/wall_sec``)."""
    if not _WANDB or wandb.run is None:
        return
    if primary == "acc":
        if "acc" in train_metrics:
            wandb.run.summary["summary/train_acc"] = float(train_metrics["acc"])
        if "acc" in val_metrics:
            wandb.run.summary["summary/val_acc"] = float(val_metrics["acc"])
    elif primary == "mse":
        if "mse" in train_metrics:
            wandb.run.summary["summary/train_mse"] = float(train_metrics["mse"])
        if "mse" in val_metrics:
            wandb.run.summary["summary/val_mse"] = float(val_metrics["mse"])
    wandb.run.summary["summary/wall_sec"] = float(wall_sec)


def finish_wandb() -> None:
    if not _WANDB or wandb.run is None:
        return
    wandb.finish()
