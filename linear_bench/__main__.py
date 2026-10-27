"""CLI: ``python -m linear_bench --config path/to.json``."""

from __future__ import annotations

import argparse
from pathlib import Path

from .train import load_config, train_one, write_metrics
from .wandb_util import DEFAULT_PROJECT, resolve_wandb_enabled


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="linear_bench: cheap linear/MLP train+eval")
    p.add_argument("--config", type=Path, required=True, help="JSON config path")
    p.add_argument("--out", type=Path, default=None, help="optional metrics JSON path")
    p.add_argument("--seed", type=int, default=None, help="override config seed")
    p.add_argument("--epochs", type=int, default=None, help="override config epochs")
    p.add_argument("--device", type=str, default=None, help="override device (cpu/cuda)")
    p.add_argument("--wandb", action="store_true", default=None, help="enable W&B logging")
    p.add_argument("--no-wandb", action="store_true", help="disable W&B (overrides config)")
    p.add_argument("--wandb-project", type=str, default=None, help=f"W&B project (default {DEFAULT_PROJECT})")
    p.add_argument("--wandb-entity", type=str, default=None, help="W&B entity/team")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.device is not None:
        cfg["device"] = args.device

    cfg["wandb"] = resolve_wandb_enabled(cfg, args.wandb, args.no_wandb)
    if args.wandb_project is not None:
        cfg["wandb_project"] = args.wandb_project
    else:
        cfg.setdefault("wandb_project", DEFAULT_PROJECT)
    if args.wandb_entity is not None:
        cfg["wandb_entity"] = args.wandb_entity

    metrics = train_one(cfg)
    out = args.out
    if out is None and "out" in cfg:
        out = Path(cfg["out"])
    write_metrics(metrics, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
