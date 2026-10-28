#!/usr/bin/env python3
"""Torus SFPRM train entry — thin CLI wrapper over sphere train.py.

CLI-only: monkey-patches ``train.SFPRM`` / ``project_to_sphere`` / W&B init when
this file is executed as ``__main__``. Do not ``import train_torus`` from another
process that still needs the sphere bindings — import ``models.sfprm_torus_core``
directly instead. Sphere argparse defaults in ``train.py`` are untouched.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _patch_sphere_train_for_torus() -> None:
    """Bind torus model + retract + W&B group/tags onto the sphere train module."""
    import train as sphere_train
    from models.sfprm_torus_core import SFPRMTorus
    from models.toroidal_ops import project_to_torus

    try:
        import wandb

        _WANDB = True
    except ImportError:
        wandb = None  # type: ignore
        _WANDB = False

    # CarryBank.load / model ctor resolve these names at call time.
    sphere_train.SFPRM = SFPRMTorus  # type: ignore[misc, assignment]
    sphere_train.project_to_sphere = project_to_torus  # type: ignore[misc]

    def maybe_init_wandb_torus(args, num_params: int) -> bool:
        use = args.wandb
        if args.no_wandb:
            use = False
        elif use is None:
            use = bool(os.environ.get("WANDB_API_KEY"))

        if not use:
            return False
        if not _WANDB:
            print("wandb not installed; continuing without logging")
            return False

        mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"
        name = f"sfprm-torus-{datetime.now(tz=UTC).strftime('%Y%m%d-%H%M%S')}"
        group = os.environ.get("WANDB_RUN_GROUP", "manifold-ab-12k")
        tags = ["manifold:torus", "baseline:dinw2si9"]
        extra = os.environ.get("WANDB_TAGS")
        if extra:
            tags.extend(t.strip() for t in extra.split(",") if t.strip())
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=name,
            group=group,
            tags=tags,
            mode=mode,
            config={
                "manifold": "torus",
                "baseline_run": "dinw2si9",
                "steps": args.steps,
                "bptt_k": args.bptt_k,
                "batch_size": args.batch_size,
                "lr_muon": args.lr_muon,
                "lr_adam": args.lr_adam,
                "lr_puzzle": args.lr_puzzle,
                "weight_decay": args.weight_decay,
                "d_model": args.d_model,
                "d_ff": args.d_ff,
                "n_layers": args.n_layers,
                "tau": args.tau,
                "max_iter_train": args.max_iter_train,
                "max_iter_eval": args.max_loops,
                "num_params": num_params,
                "device": str(args.device),
            },
        )
        wandb.log({"num_params": num_params}, step=0)
        return True

    sphere_train.maybe_init_wandb = maybe_init_wandb_torus  # type: ignore[misc]
    sphere_train._WANDB = _WANDB  # type: ignore[misc]
    if _WANDB:
        sphere_train.wandb = wandb  # type: ignore[misc]


if __name__ == "__main__":
    _patch_sphere_train_for_torus()
    import train as sphere_train

    sphere_train.train(sphere_train.parse_args())
