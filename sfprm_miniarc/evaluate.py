#!/usr/bin/env python3
"""Adaptive chordal-halting evaluation (Pass@1) for SFPRM on Mini-ARC.

Halting uses paper residual max_t(1−⟨z,ẑ⟩) with τ=0.01 and eval max_iter=200.
Vote-over-augs (paper B.1 / B.4) is not implemented here — unaugmented Pass@1 only.
"""

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

from data.dataset import (  # noqa: E402
    DEFAULT_PACK_MODE,
    PACK_MODE_PACKED,
    PAPER_SEQ_LEN,
    VOCAB_SIZE,
    build_cache,
    default_max_seq_len,
)
from models.sfprm_core import (  # noqa: E402
    DEFAULT_D_FF,
    DEFAULT_D_MODEL,
    DEFAULT_N_HEADS,
    DEFAULT_N_LAYERS,
    SFPRM,
)
from models.solver import SolverConfig, SolverState, solver_step  # noqa: E402

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
    puzzle_id: torch.Tensor,
    *,
    tau: float = 0.01,
    max_loops: int = 200,
    gamma: float = 0.9,
    eta_min: float = 0.001,
    patience: int = 5,
    use_amp: bool = False,
) -> tuple[bool, int]:
    model.eval()
    cfg = SolverConfig(
        patience=patience,
        gamma=gamma,
        eta_min=eta_min,
        tau=tau,
        max_iter=max_loops,
    )
    device_type = "cuda" if x_test.device.type == "cuda" else "cpu"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
        z_input = model.get_input_embedding(x_test, puzzle_id)
        pad_mask = model.token_pad_mask(x_test)
        z = z_input.clone()
        state = SolverState.init(1, x_test.device, cfg)
        for _ in range(max_loops):
            if bool(state.halted.all()):
                break
            z, _residual, state = solver_step(
                model, z, z_input, state, cfg, pad_mask=pad_mask
            )
        qmask = model.expand_query_mask(query_mask)
        pred_logits = model.decode_logits(z)[qmask]
        pred_grid = pred_logits.argmax(dim=-1)
        exact_match = bool((pred_grid == y_test[query_mask]).all().item())
        return exact_match, int(state.steps_used.item())


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
            "tau": args.tau,
            "max_loops": args.max_loops,
        },
        job_type="eval",
    )
    return True


def _build_model_from_ckpt(ckpt: dict, args: argparse.Namespace, device: torch.device) -> SFPRM:
    a = ckpt.get("args") or {}
    d_model = int(a.get("d_model", args.d_model))
    n_heads = int(a.get("n_heads", args.n_heads))
    d_ff = int(a.get("d_ff", args.d_ff))
    n_layers = int(a.get("n_layers", args.n_layers))
    max_seq_len = int(a.get("max_seq_len", args.max_seq_len))
    puzzle_emb_len = int(a.get("puzzle_emb_len", args.puzzle_emb_len))
    # Resolve packing from ckpt (old EX-33 packs had max_seq_len=128, no pack_mode key).
    args.max_seq_len = max_seq_len
    if "pack_mode" in a:
        args.pack_mode = a["pack_mode"]
    elif max_seq_len > PAPER_SEQ_LEN:
        args.pack_mode = PACK_MODE_PACKED
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    # Infer n_puzzles from puzzle_emb weight if present
    n_puzzles = 149
    if isinstance(state, dict) and "puzzle_emb.weight" in state:
        n_puzzles = int(state["puzzle_emb.weight"].shape[0])
    model = SFPRM(
        vocab_size=VOCAB_SIZE,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        n_puzzles=n_puzzles,
        puzzle_emb_len=puzzle_emb_len,
    ).to(device)
    # Prefer EMA shadow for eval when present
    if args.use_ema and isinstance(ckpt, dict) and ckpt.get("ema"):
        model.load_state_dict(ckpt["ema"]["shadow"])
    else:
        model.load_state_dict(state)
    return model


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate SFPRM with paper chordal halting")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_dir", type=str, default=str(ROOT / "data" / "mini_arc"))
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--epsilon", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("--max_loops", type=int, default=200)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument(
        "--pack_mode",
        type=str,
        choices=["demo_pair", "packed"],
        default=DEFAULT_PACK_MODE,
        help="Must match train packing. Legacy ckpts: --pack_mode packed --max_seq_len 128",
    )
    p.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
        help="Token length (default: 25 for demo_pair, 128 for packed; overridden by ckpt args)",
    )
    p.add_argument("--d_model", type=int, default=DEFAULT_D_MODEL)
    p.add_argument("--n_heads", type=int, default=DEFAULT_N_HEADS)
    p.add_argument("--d_ff", type=int, default=DEFAULT_D_FF)
    p.add_argument("--n_layers", type=int, default=DEFAULT_N_LAYERS)
    p.add_argument("--puzzle_emb_len", type=int, default=1)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max_tasks", type=int, default=None)
    p.add_argument("--use_ema", action="store_true", default=True)
    p.add_argument("--no_ema", action="store_true")
    p.add_argument("--wandb-project", type=str, default="sfprm-miniarc")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb", action="store_true", default=None)
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args(argv)
    if args.epsilon is not None:
        args.tau = args.epsilon
    if args.no_ema:
        args.use_ema = False
    if args.max_seq_len is None:
        args.max_seq_len = default_max_seq_len(args.pack_mode)

    device = resolve_device(args.device)
    use_amp = device.type == "cuda"

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = _build_model_from_ckpt(ckpt, args, device)
    # Prefer packing / seq len saved with the checkpoint (overrides CLI defaults).
    cache = build_cache(
        Path(args.data_dir),
        device=device,
        max_seq_len=args.max_seq_len,
        pack_mode=args.pack_mode,
        augment=False,
        max_tasks=args.max_tasks,
    )

    wandb_on = maybe_init_wandb(args)
    hits = 0
    halt_sum = 0
    for i in tqdm(range(len(cache)), desc="eval", dynamic_ncols=True):
        exact, used = evaluate_task(
            model,
            cache.tokens[i : i + 1],
            cache.targets[i : i + 1],
            cache.query_mask[i : i + 1],
            cache.puzzle_ids[i : i + 1],
            tau=args.tau,
            max_loops=args.max_loops,
            gamma=args.gamma,
            use_amp=use_amp,
        )
        hits += int(exact)
        halt_sum += used

    n = len(cache)
    pas = hits / max(n, 1)
    mean_halt = halt_sum / max(n, 1)
    print(f"Pass@1: {pas:.4f} ({hits}/{n})")
    print(f"Mean halt steps: {mean_halt:.2f}")
    print("Note: vote-over-augs (paper) not applied — unaugmented task Pass@1 only.")

    if wandb_on:
        wandb.log({"eval/pass_at_1": pas, "eval/mean_halt_steps": mean_halt})
        wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
