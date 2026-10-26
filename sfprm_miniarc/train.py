#!/usr/bin/env python3
"""Truncated BPTT training for SFPRM on Mini-ARC (dual Muon + AdamW, W&B)."""

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

# Allow `python train.py` from sfprm_miniarc/
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import MiniARCCache, build_cache, make_train_indices  # noqa: E402
from models.sfprm_core import SFPRM  # noqa: E402
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
) -> tuple[Muon, torch.optim.AdamW]:
    """
    Param split (documented):
      Muon  — 2D Linear weights inside ``core_block`` only.
      AdamW — token/pos embeddings, classification head, raw_lambda, raw_eta.
    Embeddings and head are 2D but stay on AdamW per Muon usage guidance.
    """
    muon_params = [p for p in model.core_block.parameters() if p.ndim == 2 and p.requires_grad]
    muon_ids = {id(p) for p in muon_params}
    adam_params = [p for p in model.parameters() if p.requires_grad and id(p) not in muon_ids]
    opt_muon = Muon(muon_params, lr=lr_muon, momentum=0.95)
    opt_adam = torch.optim.AdamW(adam_params, lr=lr_adam, weight_decay=weight_decay)
    return opt_muon, opt_adam


def maybe_init_wandb(args: argparse.Namespace, num_params: int) -> bool:
    """Init W&B when enabled. Default: on if WANDB_API_KEY set; else off unless --wandb."""
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
            "compile": args.compile,
            "max_seq_len": args.max_seq_len,
            "n_color_perms": args.n_color_perms,
            "eval_every": args.eval_every,
            "epsilon": args.epsilon,
            "max_loops": args.max_loops,
            "num_params": num_params,
            "device": str(args.device),
        },
    )
    wandb.log({"num_params": num_params}, step=0)
    return True


@torch.no_grad()
def eval_metrics(
    model: SFPRM,
    cache: MiniARCCache,
    *,
    epsilon: float,
    max_loops: int,
    device: torch.device,
    bptt_k: int,
) -> dict[str, float]:
    """Unaugmented Mini-ARC eval: chordal-halting Pass@1 + fixed-k CE loss.

    Returns keys suitable for ``val/*`` (and aliased ``eval/*``) W&B logging.
    """
    model.eval()
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    use_amp = device.type == "cuda"
    hits = 0
    halt_sum = 0
    cell_correct = 0
    cell_total = 0
    loss_sum = 0.0
    n = len(cache)
    for i in range(n):
        x = cache.tokens[i : i + 1]
        y = cache.targets[i : i + 1]
        mask = cache.query_mask[i : i + 1]
        with torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=use_amp):
            z_input = model.get_input_embedding(x)

            # Fixed-k loss (same unroll depth as train BPTT).
            z_loss = z_input.clone()
            for _ in range(bptt_k):
                z_loss = model.step(z_loss, z_input)
            loss_sum += float(F.cross_entropy(model.head(z_loss[mask]), y[mask]).item())

            # Adaptive chordal halt → Pass@1 / cell acc / halt depth.
            z = z_input.clone()
            used = max_loops
            for step in range(max_loops):
                z_next = model.step(z, z_input)
                chordal_delta = torch.norm(z_next - z, dim=-1).max().item()
                z = z_next
                if chordal_delta < epsilon:
                    used = step + 1
                    break
            logits = model.head(z[mask])
            pred = logits.argmax(dim=-1)
            tgt = y[mask]
            exact = bool((pred == tgt).all().item())
            cell_correct += int((pred == tgt).sum().item())
            cell_total += int(tgt.numel())
        hits += int(exact)
        halt_sum += used
    denom = max(n, 1)
    return {
        "pass_at_1": hits / denom,
        "mean_halt_steps": halt_sum / denom,
        "cell_accuracy": cell_correct / max(cell_total, 1),
        "loss": loss_sum / denom,
    }


@torch.no_grad()
def eval_pass_at_1(
    model: SFPRM,
    cache: MiniARCCache,
    *,
    epsilon: float,
    max_loops: int,
    device: torch.device,
) -> tuple[float, float]:
    """Backward-compatible wrapper -> (pass_at_1, mean_halt_steps)."""
    m = eval_metrics(
        model,
        cache,
        epsilon=epsilon,
        max_loops=max_loops,
        device=device,
        bptt_k=6,
    )
    return m["pass_at_1"], m["mean_halt_steps"]


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
        augment=True,
        n_color_perms=args.n_color_perms,
        seed=args.seed,
        max_tasks=args.max_tasks,
    )
    eval_cache = build_cache(
        data_dir,
        device=device,
        max_seq_len=args.max_seq_len,
        augment=False,
        seed=args.seed,
        max_tasks=args.max_tasks,
    )
    print(f"Train cache: {len(train_cache)} samples on {device}; eval tasks: {len(eval_cache)}")

    model = SFPRM(vocab_size=11, d_model=args.d_model, max_seq_len=args.max_seq_len).to(device)
    n_params = model.count_parameters()
    print(f"Parameters: {n_params / 1e6:.3f}M")

    opt_muon, opt_adam = build_optimizers(model, args.lr_muon, args.lr_adam, args.weight_decay)

    step_fn = model.step
    if args.compile:
        try:
            step_fn = torch.compile(model.step, mode="reduce-overhead")
            print("torch.compile enabled (reduce-overhead)")
        except Exception as exc:  # noqa: BLE001
            print(f"torch.compile failed ({exc}); using eager step")

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
        x, y, mask = train_cache.batch(idx)

        opt_muon.zero_grad(set_to_none=True)
        opt_adam.zero_grad(set_to_none=True)

        with torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=use_amp):
            z_input = model.get_input_embedding(x)
            z = z_input.clone()
            for _ in range(args.bptt_k):
                z = step_fn(z, z_input)
            logits = model.head(z[mask])
            loss = F.cross_entropy(logits, y[mask])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt_muon.step()
        opt_adam.step()

        loss_v = float(loss.item())
        if not math.isfinite(loss_v):
            raise RuntimeError(f"non-finite loss at step {step}")
        running = 0.9 * running + 0.1 * loss_v if step > 1 else loss_v
        pbar.set_postfix(loss=f"{running:.4f}")

        if wandb_on and step % args.log_every == 0:
            wandb.log(
                {
                    "train/loss": loss_v,
                    "train/lr_muon": args.lr_muon,
                    "train/lr_adam": args.lr_adam,
                    "train/lambda": float(torch.sigmoid(model.raw_lambda).item()),
                    "train/eta": float(torch.sigmoid(model.raw_eta).item()),
                },
                step=step,
            )

        do_eval = args.eval_every > 0 and (step % args.eval_every == 0 or step == args.steps)
        if do_eval:
            metrics = eval_metrics(
                model,
                eval_cache,
                epsilon=args.epsilon,
                max_loops=args.max_loops,
                device=device,
                bptt_k=args.bptt_k,
            )
            pas = metrics["pass_at_1"]
            mean_halt = metrics["mean_halt_steps"]
            print(
                f"\n[step {step}] val Pass@1={pas:.4f} cell={metrics['cell_accuracy']:.4f} "
                f"loss={metrics['loss']:.4f} mean_halt={mean_halt:.1f}"
            )
            if wandb_on:
                # Primary TRM-style val/* panel; keep eval/* aliases for older dashboards.
                wandb.log(
                    {
                        "val/pass_at_1": pas,
                        "val/cell_accuracy": metrics["cell_accuracy"],
                        "val/loss": metrics["loss"],
                        "val/mean_halt_steps": mean_halt,
                        "eval/pass_at_1": pas,
                        "eval/mean_halt_steps": mean_halt,
                    },
                    step=step,
                )
            if pas >= best_pass:
                best_pass = pas
                torch.save(
                    {
                        "model": model.state_dict(),
                        "step": step,
                        "pass_at_1": pas,
                        "args": vars(args),
                    },
                    best_path,
                )
            model.train()

        if step % args.ckpt_every == 0 or step == args.steps:
            torch.save(
                {"model": model.state_dict(), "step": step, "args": vars(args)},
                last_path,
            )

    if best_pass < 0:
        # No eval during short smoke runs — still write a best checkpoint.
        torch.save(
            {"model": model.state_dict(), "step": args.steps, "pass_at_1": None, "args": vars(args)},
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
    p = argparse.ArgumentParser(description="Train SFPRM on Mini-ARC")
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--bptt_k", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr_muon", type=float, default=0.02)
    p.add_argument("--lr_adam", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--max_seq_len", type=int, default=128)
    p.add_argument("--n_color_perms", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data_dir", type=str, default=str(ROOT / "data" / "mini_arc"))
    p.add_argument("--out_dir", type=str, default=str(ROOT / "checkpoints"))
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--ckpt_every", type=int, default=1000)
    p.add_argument("--epsilon", type=float, default=1e-3)
    p.add_argument("--max_loops", type=int, default=100)
    p.add_argument("--max_tasks", type=int, default=None, help="Limit tasks (smoke tests)")
    p.add_argument("--wandb-project", type=str, default="sfprm-miniarc")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb", action="store_true", default=None, help="Force-enable W&B (offline if no key)")
    p.add_argument("--no-wandb", action="store_true", help="Disable W&B")
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
