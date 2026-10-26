#!/usr/bin/env python3
"""Truncated BPTT training for SFPRM on Mini-ARC (paper-faithful solver + nGPT).

Defaults follow arXiv:2610.25646 Appendix B / Table 3 (Mini-ARC), except desk
keeps steps=12_000 and aug count D₄×8 (=64). Sequences default to paper
demo-pair seq-25 (``--pack_mode demo_pair``). Legacy packed demos+query ≤128:
``--pack_mode packed --max_seq_len 128``. Dims scaled for 1×16GB GPU.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import (  # noqa: E402
    DEFAULT_PACK_MODE,
    VOCAB_SIZE,
    MiniARCCache,
    build_cache,
    default_max_seq_len,
    make_train_indices,
)
from models.ema import EMAModel  # noqa: E402
from models.sfprm_core import (  # noqa: E402
    DEFAULT_D_FF,
    DEFAULT_D_MODEL,
    DEFAULT_N_HEADS,
    DEFAULT_N_LAYERS,
    SFPRM,
)
from models.solver import SolverConfig, SolverState, run_segment, solver_step  # noqa: E402
from models.spherical_ops import project_to_sphere  # noqa: E402
from optim.muon import Muon  # noqa: E402

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


def build_optimizers(
    model: SFPRM,
    lr_muon: float,
    lr_adam: float,
    weight_decay: float,
    lr_puzzle: float,
    wd_puzzle: float,
) -> tuple[Muon, torch.optim.AdamW, torch.optim.AdamW]:
    """
    Param split (paper):
      Muon  — 2D weight matrices in ``core_block`` (Linear + depthwise conv kernels).
      AdamW — embeddings (non-puzzle), head, Q-head, scalars / eigen-LR / s_qk / s_z / λ.
      AdamW (puzzle) — puzzle embeddings at lr_puzzle / wd_puzzle (paper 1e-2 / 0).
    """
    # Muon on 2-D Linear weights inside core_block only (conv kernels → AdamW).
    muon_params = [p for p in model.core_block.parameters() if p.requires_grad and p.ndim == 2]
    muon_ids = {id(p) for p in muon_params}
    puzzle_ids = {id(p) for p in model.puzzle_emb.parameters()}
    adam_params = [
        p
        for p in model.parameters()
        if p.requires_grad and id(p) not in muon_ids and id(p) not in puzzle_ids
    ]
    puzzle_params = [p for p in model.puzzle_emb.parameters() if p.requires_grad]

    opt_muon = Muon(muon_params, lr=lr_muon, momentum=0.95, weight_decay=weight_decay)
    opt_adam = torch.optim.AdamW(adam_params, lr=lr_adam, weight_decay=weight_decay)
    opt_puzzle = torch.optim.AdamW(puzzle_params, lr=lr_puzzle, weight_decay=wd_puzzle)
    return opt_muon, opt_adam, opt_puzzle


class CarryBank:
    """Per-sample latent + solver state carried across optimizer steps (detached)."""

    def __init__(self, n: int, seq_with_puzzle: int, d_model: int, device: torch.device, cfg: SolverConfig):
        self.cfg = cfg
        self.z = torch.zeros(n, seq_with_puzzle, d_model, device=device)
        self.valid = torch.zeros(n, dtype=torch.bool, device=device)
        self.eta = torch.ones(n, device=device)
        self.patience_left = torch.full((n,), cfg.patience, device=device, dtype=torch.long)
        self.best_residual = torch.full((n,), float("inf"), device=device)
        self.steps_used = torch.zeros(n, dtype=torch.long, device=device)
        self.halted = torch.zeros(n, dtype=torch.bool, device=device)

    def load(self, idx: torch.Tensor, z_input: torch.Tensor) -> tuple[torch.Tensor, SolverState]:
        b = idx.shape[0]
        z = z_input.clone()
        state = SolverState.init(b, z_input.device, self.cfg)
        live = self.valid[idx]
        if live.any():
            z = torch.where(live[:, None, None], self.z[idx], z)
            state.eta = torch.where(live, self.eta[idx], state.eta)
            state.patience_left = torch.where(live, self.patience_left[idx], state.patience_left)
            state.best_residual = torch.where(live, self.best_residual[idx], state.best_residual)
            state.steps_used = torch.where(live, self.steps_used[idx], state.steps_used)
            state.halted = torch.where(live, self.halted[idx], state.halted)
            # Re-init samples that already halted or hit max_iter
            rein = state.halted | (state.steps_used >= self.cfg.max_iter)
            if rein.any():
                z = torch.where(rein[:, None, None], z_input, z)
                state.eta = torch.where(rein, torch.ones_like(state.eta), state.eta)
                state.patience_left = torch.where(
                    rein, torch.full_like(state.patience_left, self.cfg.patience), state.patience_left
                )
                state.best_residual = torch.where(
                    rein, torch.full_like(state.best_residual, float("inf")), state.best_residual
                )
                state.steps_used = torch.where(rein, torch.zeros_like(state.steps_used), state.steps_used)
                state.halted = torch.where(rein, torch.zeros_like(state.halted), state.halted)
        return project_to_sphere(z), state

    @torch.no_grad()
    def store(self, idx: torch.Tensor, z: torch.Tensor, state: SolverState) -> None:
        self.z[idx] = z.detach()
        self.eta[idx] = state.eta.detach()
        self.patience_left[idx] = state.patience_left.detach()
        self.best_residual[idx] = state.best_residual.detach()
        self.steps_used[idx] = state.steps_used.detach()
        self.halted[idx] = state.halted.detach()
        self.valid[idx] = True


def maybe_init_wandb(args: argparse.Namespace, num_params: int) -> bool:
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
    name = f"sfprm-{datetime.now(tz=UTC).strftime('%Y%m%d-%H%M%S')}"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=name,
        mode=mode,
        config={
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


@torch.no_grad()
def eval_pass_at_1(
    model: SFPRM,
    cache: MiniARCCache,
    *,
    tau: float,
    max_loops: int,
    gamma_eval: float,
    device: torch.device,
) -> tuple[float, float]:
    """Chordal-halting Pass@1 (paper τ=0.01, max_iter=200). No vote-over-augs yet."""
    model.eval()
    cfg = SolverConfig(
        patience=5,
        gamma=gamma_eval,
        eta_min=0.001,
        tau=tau,
        max_iter=max_loops,
    )
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    hits = 0
    halt_sum = 0
    n = len(cache)
    for i in range(n):
        x = cache.tokens[i : i + 1]
        y = cache.targets[i : i + 1]
        mask = cache.query_mask[i : i + 1]
        pid = cache.puzzle_ids[i : i + 1]
        with torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=device.type == "cuda"):
            z_input = model.get_input_embedding(x, pid)
            pad_mask = model.token_pad_mask(x)
            z = z_input.clone()
            state = SolverState.init(1, device, cfg)
            for _ in range(max_loops):
                if bool(state.halted.all()):
                    break
                z, _residual, state = solver_step(
                    model, z, z_input, state, cfg, pad_mask=pad_mask
                )
            qmask = model.expand_query_mask(mask)
            logits = model.decode_logits(z)[qmask]
            pred = logits.argmax(dim=-1)
            exact = bool((pred == y[mask]).all().item())
        hits += int(exact)
        halt_sum += int(state.steps_used.item())
    return hits / max(n, 1), halt_sum / max(n, 1)


def train(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    args.device = device
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_cache = build_cache(
        data_dir,
        device=device,
        max_seq_len=args.max_seq_len,
        pack_mode=args.pack_mode,
        augment=True,
        n_color_perms=args.n_color_perms,
        seed=args.seed,
        max_tasks=args.max_tasks,
    )
    eval_cache = build_cache(
        data_dir,
        device=device,
        max_seq_len=args.max_seq_len,
        pack_mode=args.pack_mode,
        augment=False,
        seed=args.seed,
        max_tasks=args.max_tasks,
    )
    n_puzzles = int(train_cache.puzzle_ids.max().item()) + 1 if len(train_cache) else 1
    print(
        f"Train cache: {len(train_cache)} samples on {device}; eval tasks: {len(eval_cache)}; "
        f"puzzles={n_puzzles}; pack_mode={args.pack_mode}; max_seq_len={args.max_seq_len}"
    )

    model = SFPRM(
        vocab_size=VOCAB_SIZE,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq_len,
        n_puzzles=n_puzzles,
        puzzle_emb_len=args.puzzle_emb_len,
        weight_dropout=args.weight_dropout,
        variational_dropout=args.variational_dropout,
        lambda_init=args.lambda_init,
    ).to(device)
    n_params = model.count_parameters()
    print(
        f"Parameters: {n_params / 1e6:.3f}M "
        f"(d_model={args.d_model}, d_ff={args.d_ff}, layers={args.n_layers}, heads={args.n_heads})"
    )

    opt_muon, opt_adam, opt_puzzle = build_optimizers(
        model,
        args.lr_muon,
        args.lr_adam,
        args.weight_decay,
        args.lr_puzzle,
        args.wd_puzzle,
    )
    ema = EMAModel(model, decay=args.ema_decay) if args.ema_decay > 0 else None
    # Compile after EMA deepcopy so the shadow stays eager. Same hot path as
    # scripts/bench_compile.py (paper SFPRM.step); solver_step / run_segment call it.
    if args.compile:
        model.step = torch.compile(model.step, mode="reduce-overhead")  # type: ignore[method-assign]
        print("torch.compile: model.step (mode=reduce-overhead)")

    solver_cfg = SolverConfig(
        patience=args.patience,
        gamma=args.gamma_train,
        eta_min=args.eta_min,
        tau=args.tau,
        max_iter=args.max_iter_train,
    )
    seq_with_puzzle = args.max_seq_len + args.puzzle_emb_len
    carry = CarryBank(len(train_cache), seq_with_puzzle, args.d_model, device, solver_cfg)

    wandb_on = maybe_init_wandb(args, n_params)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.seed)

    amp_device = "cuda" if device.type == "cuda" else "cpu"
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    use_amp = device.type == "cuda"

    best_pass = -1.0
    best_path = out_dir / "model_best.pt"
    last_path = out_dir / "model_last.pt"

    model.train()
    pbar = tqdm(range(1, args.steps + 1), desc="train", dynamic_ncols=True)
    running = 0.0
    for step in pbar:
        idx = make_train_indices(len(train_cache), args.batch_size, step, gen).to(device)
        x, y, mask, pid = train_cache.batch(idx)

        opt_muon.zero_grad(set_to_none=True)
        opt_adam.zero_grad(set_to_none=True)
        opt_puzzle.zero_grad(set_to_none=True)

        with torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=use_amp):
            z_input = model.get_input_embedding(x, pid)
            pad_mask = model.token_pad_mask(x)
            z, state = carry.load(idx, z_input)
            # Locked variational mask sampled once inside run_segment; pad_mask ignores PAD.
            z, state, _residual = run_segment(
                model,
                z,
                z_input,
                state,
                solver_cfg,
                k=args.bptt_k,
                pad_mask=pad_mask,
            )

            qmask = model.expand_query_mask(mask)
            logits = model.decode_logits(z)[qmask]
            ce = F.cross_entropy(logits, y[mask])

            # Q-head on detached state: BCE with "all query cells correct" soft target
            with torch.no_grad():
                pred = logits.detach().argmax(dim=-1)
                # rebuild per-sample correctness
                correct = torch.zeros(x.shape[0], device=device)
                # ponytail: O(B) scatter via loop over batch — fine at B≤512
                offset = 0
                for b in range(x.shape[0]):
                    m = mask[b]
                    npos = int(m.sum().item())
                    if npos == 0:
                        continue
                    exact = bool((pred[offset : offset + npos] == y[b][m]).all().item())
                    correct[b] = float(exact)
                    offset += npos
            q_logit = model.q_logits(z.detach())
            q_loss = F.binary_cross_entropy_with_logits(q_logit, correct)
            loss = ce + args.q_loss_weight * q_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt_muon.step()
        opt_adam.step()
        opt_puzzle.step()
        model.renorm_weights()
        if ema is not None:
            ema.update(model)

        carry.store(idx, z, state)

        loss_v = float(loss.item())
        if not math.isfinite(loss_v):
            raise RuntimeError(f"non-finite loss at step {step}")
        running = 0.9 * running + 0.1 * loss_v if step > 1 else loss_v
        pbar.set_postfix(loss=f"{running:.4f}", lam=f"{float(model.lambda_param.item()):.3f}")

        if wandb_on and step % args.log_every == 0:
            wandb.log(
                {
                    "train/loss": loss_v,
                    "train/ce": float(ce.item()),
                    "train/q_loss": float(q_loss.item()),
                    "train/lr_muon": args.lr_muon,
                    "train/lr_adam": args.lr_adam,
                    "train/lambda": float(model.lambda_param.item()),
                    "train/mean_eta": float(state.eta.mean().item()),
                    "train/halt_frac": float(state.halted.float().mean().item()),
                },
                step=step,
            )

        do_eval = args.eval_every > 0 and (step % args.eval_every == 0 or step == args.steps)
        if do_eval:
            eval_model = ema.shadow if ema is not None else model
            pas, mean_halt = eval_pass_at_1(
                eval_model,
                eval_cache,
                tau=args.tau,
                max_loops=args.max_loops,
                gamma_eval=args.gamma_eval,
                device=device,
            )
            print(f"\n[step {step}] eval Pass@1={pas:.4f} mean_halt={mean_halt:.1f}")
            if wandb_on:
                wandb.log({"eval/pass_at_1": pas, "eval/mean_halt_steps": mean_halt}, step=step)
            if pas >= best_pass:
                best_pass = pas
                torch.save(
                    {
                        "model": model.state_dict(),
                        "ema": ema.state_dict() if ema is not None else None,
                        "step": step,
                        "pass_at_1": pas,
                        "args": vars(args),
                        "num_params": n_params,
                    },
                    best_path,
                )
            model.train()

        if step % args.ckpt_every == 0 or step == args.steps:
            torch.save(
                {
                    "model": model.state_dict(),
                    "ema": ema.state_dict() if ema is not None else None,
                    "step": step,
                    "args": vars(args),
                    "num_params": n_params,
                },
                last_path,
            )

    if best_pass < 0:
        torch.save(
            {
                "model": model.state_dict(),
                "ema": ema.state_dict() if ema is not None else None,
                "step": args.steps,
                "pass_at_1": None,
                "args": vars(args),
                "num_params": n_params,
            },
            best_path,
        )

    if wandb_on:
        if best_path.exists():
            art = wandb.Artifact(name=f"model-{wandb.run.id}", type="model")
            art.add_file(str(best_path))
            wandb.log_artifact(art)
        wandb.finish()
    print(f"Done. best Pass@1={best_pass if best_pass >= 0 else 'n/a'}; ckpt dir={out_dir}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SFPRM on Mini-ARC (paper-faithful)")
    # Desk keeps 12k steps + 149×64 augs; do not bump to paper ~69k / 100×.
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--bptt_k", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=512)
    # Paper Mini-ARC: Muon 5e-3, AdamW 1e-3, wd 0.1, puzzle lr/wd 1e-2/0
    p.add_argument("--lr_muon", type=float, default=5e-3)
    p.add_argument("--lr_adam", type=float, default=1e-3)
    p.add_argument("--lr_puzzle", type=float, default=1e-2)
    p.add_argument("--wd_puzzle", type=float, default=0.0)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--weight_dropout", type=float, default=0.05)
    p.add_argument("--variational_dropout", type=float, default=0.2)
    p.add_argument("--q_loss_weight", type=float, default=0.5)
    p.add_argument("--lambda_init", type=float, default=0.1)
    # 16GB desk dims (paper ratios 512/8/×4/L=2 → scaled)
    p.add_argument("--d_model", type=int, default=DEFAULT_D_MODEL)
    p.add_argument("--n_heads", type=int, default=DEFAULT_N_HEADS)
    p.add_argument("--d_ff", type=int, default=DEFAULT_D_FF)
    p.add_argument("--n_layers", type=int, default=DEFAULT_N_LAYERS)
    p.add_argument("--puzzle_emb_len", type=int, default=1)
    p.add_argument(
        "--pack_mode",
        type=str,
        choices=["demo_pair", "packed"],
        default=DEFAULT_PACK_MODE,
        help=(
            "demo_pair (paper): 25-token I→O grids, train on demos / eval on test. "
            "packed (legacy EX-33): demos+query ≤max_seq_len (use --max_seq_len 128)."
        ),
    )
    p.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
        help="Token length (default: 25 for demo_pair, 128 for packed)",
    )
    p.add_argument("--n_color_perms", type=int, default=8, help="D₄×this = 64 augs/task (desk)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data_dir", type=str, default=str(ROOT / "data" / "mini_arc"))
    p.add_argument("--out_dir", type=str, default=str(ROOT / "checkpoints"))
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--compile",
        action="store_true",
        help="wrap model.step with torch.compile(mode=reduce-overhead); default off",
    )
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--ckpt_every", type=int, default=1000)
    # Solver / halt (paper)
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--eta_min", type=float, default=0.001)
    p.add_argument("--gamma_train", type=float, default=0.9)
    p.add_argument("--gamma_eval", type=float, default=0.9)
    p.add_argument("--max_iter_train", type=int, default=8)
    p.add_argument("--max_loops", type=int, default=200, help="Eval max_iter (paper 200)")
    p.add_argument("--epsilon", type=float, default=None, help=argparse.SUPPRESS)  # legacy alias
    p.add_argument("--max_tasks", type=int, default=None, help="Limit tasks (smoke tests)")
    p.add_argument("--wandb-project", type=str, default="sfprm-miniarc")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb", action="store_true", default=None)
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args(argv)
    if args.epsilon is not None:
        args.tau = args.epsilon
    if args.max_seq_len is None:
        args.max_seq_len = default_max_seq_len(args.pack_mode)
    return args


if __name__ == "__main__":
    train(parse_args())
