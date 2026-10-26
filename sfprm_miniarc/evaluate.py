#!/usr/bin/env python3
"""Adaptive chordal-halting evaluation (Pass@1) for SFPRM on Mini-ARC."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import build_cache  # noqa: E402
from models.sfprm_core import SFPRM  # noqa: E402

try:
    import wandb

    _WANDB = True
except ImportError:
    wandb = None  # type: ignore
    _WANDB = False


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def evaluate_task(
    model: SFPRM,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    query_mask: torch.Tensor,
    epsilon: float = 1e-3,
    max_loops: int = 100,
    use_amp: bool = False,
) -> tuple[bool, int]:
    model.eval()
    device_type = "cuda" if x_test.device.type == "cuda" else "cpu"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
        z_input = model.get_input_embedding(x_test)
        z = z_input.clone()
        used = max_loops
        for step in range(max_loops):
            z_next = model.step(z, z_input)
            chordal_delta = torch.norm(z_next - z, dim=-1).max().item()
            z = z_next
            if chordal_delta < epsilon:
                used = step + 1
                break
        pred_logits = model.head(z[query_mask])
        pred_grid = pred_logits.argmax(dim=-1)
        exact_match = bool((pred_grid == y_test[query_mask]).all().item())
        return exact_match, used


def maybe_init_wandb(args: argparse.Namespace) -> bool:
    use = args.wandb
    if args.no_wandb:
        use = False
    elif use is None:
        use = bool(os.environ.get("WANDB_API_KEY"))
    if not use or not _WANDB:
        return False
    mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"sfprm-eval-{datetime.now(tz=UTC).strftime('%Y%m%d-%H%M%S')}",
        mode=mode,
        config={
            "checkpoint": args.checkpoint,
            "epsilon": args.epsilon,
            "max_loops": args.max_loops,
        },
        job_type="eval",
    )
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate SFPRM with chordal halting")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_dir", type=str, default=str(ROOT / "data" / "mini_arc"))
    p.add_argument("--epsilon", type=float, default=1e-3)
    p.add_argument("--max_loops", type=int, default=100)
    p.add_argument("--max_seq_len", type=int, default=128)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max_tasks", type=int, default=None)
    p.add_argument("--wandb-project", type=str, default="sfprm-miniarc")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb", action="store_true", default=None)
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args(argv)

    device = resolve_device(args.device)
    use_amp = device.type == "cuda"
    cache = build_cache(
        Path(args.data_dir),
        device=device,
        max_seq_len=args.max_seq_len,
        augment=False,
        max_tasks=args.max_tasks,
    )

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model = SFPRM(vocab_size=11, d_model=args.d_model, max_seq_len=args.max_seq_len).to(device)
    model.load_state_dict(state)

    wandb_on = maybe_init_wandb(args)
    hits = 0
    halt_sum = 0
    for i in tqdm(range(len(cache)), desc="eval", dynamic_ncols=True):
        exact, used = evaluate_task(
            model,
            cache.tokens[i : i + 1],
            cache.targets[i : i + 1],
            cache.query_mask[i : i + 1],
            epsilon=args.epsilon,
            max_loops=args.max_loops,
            use_amp=use_amp,
        )
        hits += int(exact)
        halt_sum += used

    n = len(cache)
    pas = hits / max(n, 1)
    mean_halt = halt_sum / max(n, 1)
    print(f"Pass@1: {pas:.4f} ({hits}/{n})")
    print(f"Mean halt steps: {mean_halt:.2f}")

    if wandb_on:
        wandb.log({"eval/pass_at_1": pas, "eval/mean_halt_steps": mean_halt})
        wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
