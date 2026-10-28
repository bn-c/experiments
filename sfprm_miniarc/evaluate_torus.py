#!/usr/bin/env python3
"""Evaluate a torus SFPRM checkpoint — sphere evaluate.py untouched.

CLI-only monkey-patch (same caveat as ``train_torus.py``): patches ``evaluate.SFPRM``
and W&B init when run as ``__main__``.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _patch_sphere_eval_for_torus() -> None:
    import evaluate as sphere_eval
    from models.sfprm_torus_core import SFPRMTorus

    try:
        import wandb

        _WANDB = True
    except ImportError:
        wandb = None  # type: ignore
        _WANDB = False

    sphere_eval.SFPRM = SFPRMTorus  # type: ignore[misc, assignment]

    def maybe_init_wandb_torus(args) -> bool:
        use = args.wandb
        if args.no_wandb:
            use = False
        elif use is None:
            use = bool(os.environ.get("WANDB_API_KEY"))
        if not use or not _WANDB:
            return False
        mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"
        group = os.environ.get("WANDB_RUN_GROUP", "manifold-ab-12k")
        tags = ["manifold:torus", "baseline:dinw2si9", "job:eval"]
        extra = os.environ.get("WANDB_TAGS")
        if extra:
            tags.extend(t.strip() for t in extra.split(",") if t.strip())
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"sfprm-torus-eval-{datetime.now(tz=UTC).strftime('%Y%m%d-%H%M%S')}",
            group=group,
            tags=tags,
            mode=mode,
            config={
                "manifold": "torus",
                "baseline_run": "dinw2si9",
                "checkpoint": args.checkpoint,
                "tau": args.tau,
                "max_loops": args.max_loops,
            },
            job_type="eval",
        )
        return True

    sphere_eval.maybe_init_wandb = maybe_init_wandb_torus  # type: ignore[misc]
    sphere_eval._WANDB = _WANDB  # type: ignore[misc]
    if _WANDB:
        sphere_eval.wandb = wandb  # type: ignore[misc]


if __name__ == "__main__":
    _patch_sphere_eval_for_torus()
    from evaluate import main

    raise SystemExit(main())
