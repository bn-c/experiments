"""CLI: ``python -m linear_bench --config path/to.json``."""

from __future__ import annotations

import argparse
from pathlib import Path

from .train import load_config, train_one, write_metrics


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="linear_bench: cheap linear/MLP train+eval")
    p.add_argument("--config", type=Path, required=True, help="JSON config path")
    p.add_argument("--out", type=Path, default=None, help="optional metrics JSON path")
    p.add_argument("--seed", type=int, default=None, help="override config seed")
    p.add_argument("--epochs", type=int, default=None, help="override config epochs")
    p.add_argument("--device", type=str, default=None, help="override device (cpu/cuda)")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.device is not None:
        cfg["device"] = args.device

    metrics = train_one(cfg)
    out = args.out
    if out is None and "out" in cfg:
        out = Path(cfg["out"])
    write_metrics(metrics, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
