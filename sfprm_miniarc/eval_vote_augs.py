#!/usr/bin/env python3
"""Unaugmented + D₄×color vote-over-augs Pass@1 (paper B.1) with chordal halt.

For each task, all D₄×color augs (default 8×8=64) are packed into one GPU batch
and solved in a single chordal-halt loop. Predictions are inverse-transformed to
canonical space, then per-cell majority vote. The (d4=0, identity-perm) sample
in that batch is also scored as unaugmented Pass@1 (no extra forward).

Also reports paper-halt diagnostics (τ / η-floor / max_iter cap).
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.dataset import (  # noqa: E402
    CELLS,
    GRID,
    NUM_COLORS,
    PAPER_SEQ_LEN,
    _transform_task,
    list_task_files,
    load_task_json,
    pack_demo_pair,
    random_color_perm,
)
from evaluate import _build_model_from_ckpt, resolve_device  # noqa: E402
from models.solver import SolverConfig, SolverState, solver_step  # noqa: E402


def inv_dihedral(grid: np.ndarray, k: int) -> np.ndarray:
    """Inverse of ``dihedral`` (rot90 then optional fliplr)."""
    g = np.asarray(grid, dtype=np.int64)
    if k >= 4:
        g = np.fliplr(g)
    return np.ascontiguousarray(np.rot90(g, -(k % 4)))


def inv_color_perm(grid: np.ndarray, perm: np.ndarray) -> np.ndarray:
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return inv[grid]


def canonicalize_pred(pred_flat: np.ndarray, d4_k: int, perm: np.ndarray) -> np.ndarray:
    g = pred_flat.reshape(GRID, GRID)
    g = inv_color_perm(g, perm)
    g = inv_dihedral(g, d4_k)
    return g.reshape(-1)


def majority_vote(preds: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(preds, axis=0)  # [A, 25]
    out = np.empty(CELLS, dtype=np.int64)
    for i in range(CELLS):
        out[i] = Counter(stack[:, i].tolist()).most_common(1)[0][0]
    return out


@torch.no_grad()
def run_batch(
    model,
    tokens: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    puzzle_ids: np.ndarray,
    *,
    device: torch.device,
    tau: float,
    max_loops: int,
    gamma: float,
    use_amp: bool,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Batched chordal-halt decode.

    Returns (preds_flat, steps_used, residual, eta, exact_match).
    """
    x = torch.tensor(tokens, device=device)
    y = torch.tensor(targets, device=device)
    m = torch.tensor(mask, device=device)
    pid = torch.tensor(puzzle_ids, device=device, dtype=torch.long)
    b = x.shape[0]
    cfg = SolverConfig(patience=5, gamma=gamma, eta_min=0.001, tau=tau, max_iter=max_loops)
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    device_type = "cuda" if device.type == "cuda" else "cpu"
    last_residual = torch.full((b,), float("inf"), device=device)
    with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
        z_input = model.get_input_embedding(x, pid)
        pad_mask = model.token_pad_mask(x)
        z = z_input.clone()
        state = SolverState.init(b, device, cfg)
        for _ in range(max_loops):
            if bool(state.halted.all()):
                break
            z, residual, state = solver_step(model, z, z_input, state, cfg, pad_mask=pad_mask)
            last_residual = residual
        qmask = model.expand_query_mask(m)  # accounts for puzzle-emb prefix
        logits = model.decode_logits(z)
        preds: list[np.ndarray] = []
        exact = np.zeros(b, dtype=bool)
        for i in range(b):
            pred_i = logits[i][qmask[i]].argmax(dim=-1).detach().cpu().numpy()
            tgt_i = y[i][m[i]].detach().cpu().numpy()
            preds.append(pred_i)
            exact[i] = bool(np.array_equal(pred_i, tgt_i))
    return (
        preds,
        state.steps_used.detach().cpu().numpy(),
        last_residual.detach().cpu().numpy(),
        state.eta.detach().cpu().numpy(),
        exact,
    )


def classify_halt(
    steps: np.ndarray,
    residual: np.ndarray,
    eta: np.ndarray,
    *,
    tau: float,
    eta_min: float,
    max_loops: int,
) -> Counter:
    """Approximate halt reason (priority: τ, then η-floor, else cap)."""
    c: Counter = Counter()
    for s, r, e in zip(steps.tolist(), residual.tolist(), eta.tolist(), strict=True):
        if r < tau:
            c["tau"] += 1
        elif e < eta_min:
            c["eta_floor"] += 1
        elif s >= max_loops:
            c["cap"] += 1
        else:
            c["other"] += 1
    return c


@dataclass(frozen=True)
class VoteEvalResult:
    """1-shot (unaug) + vote-over-augs Pass@1 and halt stats."""

    n: int
    unaug_pass_at_1: float
    vote_pass_at_1: float
    unaug_mean_halt: float
    vote_mean_halt_per_aug: float
    halt_reasons: dict[str, int]
    wall_s: float


@torch.no_grad()
def eval_vote_augs(
    model,
    data_dir: Path | str,
    *,
    device: torch.device,
    max_seq_len: int = PAPER_SEQ_LEN,
    n_color_perms: int = 8,
    aug_batch: int = 64,
    seed: int = 0,
    tau: float = 0.01,
    max_loops: int = 200,
    gamma: float = 0.9,
    max_tasks: int | None = None,
    log_every: int = 10,
    show_progress: bool = True,
) -> VoteEvalResult:
    """Run unaugmented + D₄×color vote Pass@1 (paper B.1).

    Index 0 of each task's aug batch is always (d4=0, identity) → 1-shot.
    """
    model.eval()
    use_amp = device.type == "cuda"
    files = list_task_files(Path(data_dir))
    if max_tasks is not None:
        files = files[:max_tasks]

    rng = random.Random(seed)
    identity = np.arange(NUM_COLORS, dtype=np.int64)

    unaug_hits = 0
    vote_hits = 0
    unaug_halt = 0
    vote_halt_sum = 0
    vote_halt_n = 0
    halt_reasons: Counter = Counter()

    t0 = time.perf_counter()
    iterator = enumerate(files)
    if show_progress:
        iterator = enumerate(tqdm(files, desc="vote-eval", dynamic_ncols=True))

    for puzzle_id, path in iterator:
        task = load_task_json(path)

        # D₄ × color perms; index 0 is always (d4=0, identity) → unaugmented / 1-shot
        perms = [identity] + [random_color_perm(rng) for _ in range(n_color_perms - 1)]
        aug_meta: list[tuple[int, np.ndarray]] = []
        tok_list: list[np.ndarray] = []
        tgt_list: list[np.ndarray] = []
        mask_list: list[np.ndarray] = []
        for d4_k in range(8):
            for perm in perms:
                _demos_a, q_in_a, q_out_a = _transform_task(task, d4_k, perm)
                tok_a, tgt_a, mask_a = pack_demo_pair(
                    q_in_a, q_out_a, max_seq_len=max_seq_len
                )
                tok_list.append(tok_a)
                tgt_list.append(tgt_a)
                mask_list.append(mask_a)
                aug_meta.append((d4_k, perm))

        canon_gt = _transform_task(task, 0, identity)[2].reshape(-1)
        canon_preds: list[np.ndarray] = []
        unaug_exact = False
        unaug_steps = 0

        for start in range(0, len(tok_list), aug_batch):
            end = min(start + aug_batch, len(tok_list))
            n_b = end - start
            preds_b, steps_b, residual_b, eta_b, exact_b = run_batch(
                model,
                np.stack(tok_list[start:end]),
                np.stack(tgt_list[start:end]),
                np.stack(mask_list[start:end]),
                np.full(n_b, puzzle_id, dtype=np.int64),
                device=device,
                tau=tau,
                max_loops=max_loops,
                gamma=gamma,
                use_amp=use_amp,
            )
            halt_reasons.update(
                classify_halt(
                    steps_b, residual_b, eta_b, tau=tau, eta_min=0.001, max_loops=max_loops
                )
            )
            for j, pred_a in enumerate(preds_b):
                idx = start + j
                d4_k, perm = aug_meta[idx]
                if idx == 0:
                    unaug_exact = bool(exact_b[j])
                    unaug_steps = int(steps_b[j])
                canon_preds.append(canonicalize_pred(pred_a, d4_k, perm))
                vote_halt_sum += int(steps_b[j])
                vote_halt_n += 1

        unaug_hits += int(unaug_exact)
        unaug_halt += unaug_steps
        voted = majority_vote(canon_preds)
        vote_hits += int(np.array_equal(voted, canon_gt))

        done = puzzle_id + 1
        if show_progress and (done % log_every == 0 or done == len(files)):
            print(
                f"\n[{done}/{len(files)}] "
                f"unaug={unaug_hits / done:.4f} vote={vote_hits / done:.4f} "
                f"mean_halt_unaug={unaug_halt / done:.1f} "
                f"mean_halt_per_aug={vote_halt_sum / max(vote_halt_n, 1):.1f} "
                f"halt_reasons={dict(halt_reasons)}",
                flush=True,
            )

    n = len(files)
    return VoteEvalResult(
        n=n,
        unaug_pass_at_1=unaug_hits / max(n, 1),
        vote_pass_at_1=vote_hits / max(n, 1),
        unaug_mean_halt=unaug_halt / max(n, 1),
        vote_mean_halt_per_aug=vote_halt_sum / max(vote_halt_n, 1),
        halt_reasons=dict(halt_reasons),
        wall_s=time.perf_counter() - t0,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_dir", type=str, default=str(ROOT / "data" / "mini_arc"))
    p.add_argument("--n_color_perms", type=int, default=8, help="D₄×this augs (desk train=8→64)")
    p.add_argument(
        "--aug_batch",
        type=int,
        default=64,
        help="Augs per GPU chordal-halt loop (default 64 = all D4×8 color in one batch)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--max_loops", type=int, default=200)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument(
        "--max_seq_len",
        type=int,
        default=PAPER_SEQ_LEN,
        help="Token length (paper Mini-ARC = 25; overridden by ckpt args)",
    )
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max_tasks", type=int, default=None)
    p.add_argument("--no_ema", action="store_true")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--wandb-project", type=str, default="sfprm-miniarc")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb", action="store_true", default=None)
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    device = resolve_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ns = argparse.Namespace(
        d_model=384,
        n_heads=8,
        d_ff=1536,
        n_layers=2,
        max_seq_len=args.max_seq_len,
        puzzle_emb_len=1,
        use_ema=not args.no_ema,
    )
    model = _build_model_from_ckpt(ckpt, ns, device)
    args.max_seq_len = ns.max_seq_len

    result = eval_vote_augs(
        model,
        args.data_dir,
        device=device,
        max_seq_len=args.max_seq_len,
        n_color_perms=args.n_color_perms,
        aug_batch=args.aug_batch,
        seed=args.seed,
        tau=args.tau,
        max_loops=args.max_loops,
        gamma=args.gamma,
        max_tasks=args.max_tasks,
        log_every=args.log_every,
    )

    n_augs = 8 * args.n_color_perms
    print(f"checkpoint: {Path(args.checkpoint).resolve()}")
    print(f"ckpt_step={ckpt.get('step')} ckpt_pass_at_1={ckpt.get('pass_at_1')} ema={bool(ckpt.get('ema'))}")
    print(f"script={Path(__file__).resolve()}")
    print(
        f"tasks={result.n} augs_per_task={n_augs} "
        f"(8 D4 × {args.n_color_perms} color) aug_batch={args.aug_batch}"
    )
    print(
        f"unaug Pass@1: {result.unaug_pass_at_1:.4f}  mean_halt={result.unaug_mean_halt:.2f}"
    )
    print(
        f"vote  Pass@1: {result.vote_pass_at_1:.4f}  "
        f"mean_halt_per_aug={result.vote_mean_halt_per_aug:.2f}"
    )
    print(f"paper_halt_reasons (all aug forwards): {result.halt_reasons}")
    print(f"wall_s={result.wall_s:.1f}")

    use_wandb = args.wandb
    if args.no_wandb:
        use_wandb = False
    elif use_wandb is None:
        use_wandb = bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        try:
            import wandb
        except ImportError:
            print("wandb not installed; skipping W&B log")
        else:
            mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=f"sfprm-vote-eval-{Path(args.checkpoint).stem}",
                mode=mode,
                config={"checkpoint": args.checkpoint, **vars(args)},
            )
            wandb.log(
                {
                    "eval/pass_at_1": result.unaug_pass_at_1,
                    "eval/vote_pass_at_1": result.vote_pass_at_1,
                    "eval/mean_halt_steps": result.unaug_mean_halt,
                    "eval/vote_mean_halt_steps": result.vote_mean_halt_per_aug,
                }
            )
            wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
