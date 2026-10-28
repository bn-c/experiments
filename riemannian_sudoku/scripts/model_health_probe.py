#!/usr/bin/env python3
"""Read-only model-health probes for Riemannian Sudoku best.pt.

Reuses SFPRM activation effective-rank metrics (centered SVD):
  ER = exp(entropy of normalized σ²)
  PR = (∑λ)² / ∑λ² with λ = σ²

Probes: s / proj / axial-h ranks; MLP+velocity weight SVD; eval ablate
row/col/box/MLP; sphere cell–cell cosine vs axial mixing. No training.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from riemannian_sudoku.axial import (  # noqa: E402
    _attend_boxes,
    _attend_cols,
    _attend_rows,
)
from riemannian_sudoku.data import SudokuDataset, SudokuTorchDataset, split_train_val  # noqa: E402
from riemannian_sudoku.geometry import (  # noqa: E402
    project_tangent,
    riemannian_exp_map_orthant,
    sudoku_group_indices,
)
from riemannian_sudoku.model import RiemannianSudokuSolver, init_sphere_state  # noqa: E402
from riemannian_sudoku.train import evaluate  # noqa: E402


def spectrum_metrics(X: Tensor) -> dict:
    """Centered SVD metrics on rows of X [N, D]. Same as SFPRM EX-82."""
    X = X.detach().float()
    n, d = X.shape
    if n < 2:
        return {
            "n_rows": n,
            "d": d,
            "numeric_rank": 0,
            "eff_rank": 0.0,
            "participation_ratio": 0.0,
            "top1_var": float("nan"),
            "k90": 0,
            "dead_cols": int((X.std(0) < 1e-6).sum().item()) if n else d,
        }
    Xc = X - X.mean(0, keepdim=True)
    s = torch.linalg.svdvals(Xc).clamp_min(0.0)
    lam = s * s
    total = float(lam.sum().item())
    if total <= 0:
        er = pr = 0.0
        top1 = float("nan")
        k90 = 0
        p = torch.zeros_like(lam)
    else:
        p = lam / total
        p_nz = p[p > 0]
        er = float(torch.exp(-(p_nz * p_nz.log()).sum()).item())
        pr = float((lam.sum() ** 2 / (lam * lam).sum()).item())
        top1 = float(p[0].item())
        k90 = int((torch.cumsum(p, 0) < 0.9).sum().item()) + 1
    dead = int((X.std(0) < 1e-6).sum().item())
    tol = max(n, d) * torch.finfo(torch.float32).eps * float(s[0].item() if s.numel() else 0.0)
    numeric_rank = int((s > tol).sum().item()) if s.numel() else 0
    return {
        "n_rows": n,
        "d": d,
        "numeric_rank": numeric_rank,
        "max_sample_rank": min(n - 1, d),
        "eff_rank": er,
        "participation_ratio": pr,
        "top1_var": top1,
        "k90": k90,
        "dead_cols": dead,
        "cond_s1_smin": float((s[0] / s[numeric_rank - 1]).item()) if numeric_rank else float("nan"),
    }


def weight_svd_metrics(W: Tensor, name: str) -> dict:
    """SVD decay / norms for a weight matrix [out, in]."""
    W = W.detach().float().cpu()
    s = torch.linalg.svdvals(W).clamp_min(0.0)
    lam = s * s
    total = float(lam.sum().item())
    p = lam / total if total > 0 else torch.zeros_like(lam)
    p_nz = p[p > 0]
    er = float(torch.exp(-(p_nz * p_nz.log()).sum()).item()) if p_nz.numel() else 0.0
    pr = float((lam.sum() ** 2 / (lam * lam).sum()).item()) if total > 0 else 0.0
    topk = {f"frac_top{k}": float(p[:k].sum().item()) for k in (1, 2, 4, 8, 16) if k <= p.numel()}
    row_n = W.norm(dim=1)
    col_n = W.norm(dim=0)
    return {
        "name": name,
        "shape": list(W.shape),
        "eff_rank": er,
        "participation_ratio": pr,
        "k90": int((torch.cumsum(p, 0) < 0.9).sum().item()) + 1 if total > 0 else 0,
        "top1_var": float(p[0].item()) if p.numel() else float("nan"),
        "s_max": float(s[0].item()) if s.numel() else 0.0,
        "s_min": float(s[-1].item()) if s.numel() else 0.0,
        "cond": float((s[0] / s[-1].clamp_min(1e-12)).item()) if s.numel() else float("nan"),
        "row_norm_mean": float(row_n.mean().item()),
        "row_norm_std": float(row_n.std().item()),
        "col_norm_mean": float(col_n.mean().item()),
        "col_norm_std": float(col_n.std().item()),
        "fro": float(W.norm().item()),
        **topk,
    }


def load_model(ckpt_path: Path, device: torch.device) -> tuple[RiemannianSudokuSolver, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt.get("args") or {}
    model = RiemannianSudokuSolver(
        d_model=int(a.get("d_model", 128)),
        n_heads=int(a.get("n_heads", 4)),
        num_steps=int(a.get("num_steps", 8)),
        dt=float(a.get("dt", 0.5)),
        kinetic_weight=float(a.get("kinetic_weight", 0.05)),
        constraint_weight=float(a.get("constraint_weight", 0.0)),
        supervise_all_steps=bool(a.get("supervise_all_steps", False)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    meta = {
        "ckpt": str(ckpt_path),
        "step": ckpt.get("step"),
        "best_blank": ckpt.get("best_blank"),
        "n_params": ckpt.get("n_params"),
        "args": a,
    }
    return model, meta


def _group_mean_pairwise_cosine(s: Tensor, groups: Tensor) -> dict[str, float]:
    """Mean off-diagonal cell–cell cosine within each unit type. s: (B,81,9)."""
    # groups: (27, 9) → rows 0:9, cols 9:18, boxes 18:27
    out: dict[str, float] = {}
    names = ("row", "col", "box")
    for i, name in enumerate(names):
        g = groups[i * 9 : (i + 1) * 9]  # (9, 9)
        cells = s[:, g]  # (B, 9_units, 9_cells, 9)
        # normalize last dim
        u = cells / cells.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        # pairwise: (B, U, C, C)
        sim = torch.einsum("buci,budi->bucd", u, u)
        # off-diagonal mean
        c = sim.shape[-1]
        eye = torch.eye(c, device=sim.device, dtype=torch.bool)
        off = sim[..., ~eye]
        out[f"{name}_mean_cos"] = float(off.mean().item())
        out[f"{name}_std_cos"] = float(off.std().item())
    return out


def _attention_entropy(attn: nn.MultiheadAttention, x_ln: Tensor, layout: str) -> dict:
    """Mean attention entropy + head diversity for one axial layout."""
    b, _, d = x_ln.shape
    if layout == "row":
        h = x_ln.view(b, 9, 9, d).reshape(b * 9, 9, d)
    elif layout == "col":
        h = x_ln.view(b, 9, 9, d).transpose(1, 2).reshape(b * 9, 9, d)
    else:
        h = x_ln.view(b, 3, 3, 3, 3, d).permute(0, 1, 3, 2, 4, 5).reshape(b * 9, 9, d)
    # need_weights=True, average_attn_weights=False → (B*9, n_heads, 9, 9)
    _, w = attn(h, h, h, need_weights=True, average_attn_weights=False)
    # w may be (N, H, L, L) or (N, L, L) depending on torch version
    if w.dim() == 3:
        w = w.unsqueeze(1)
    # entropy over keys
    p = w.clamp_min(1e-12)
    ent = -(p * p.log()).sum(dim=-1)  # (N, H, L)
    max_ent = math.log(9.0)
    # head diversity: mean pairwise cosine of flattened attn maps across heads
    n, nh, l, _ = w.shape
    flat = w.reshape(n, nh, l * l)
    flat = flat / flat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    # (N, H, H)
    hh = torch.einsum("nhd,nid->nhi", flat, flat)
    # off-diag
    eye = torch.eye(nh, device=hh.device, dtype=torch.bool)
    off = hh[:, ~eye]
    return {
        "mean_entropy": float(ent.mean().item()),
        "entropy_frac_max": float((ent.mean() / max_ent).item()),
        "head_mean_cos": float(off.mean().item()) if off.numel() else float("nan"),
        "n_heads": nh,
    }


class AxialAblator:
    """Monkey-patch axial forward to zero one residual branch."""

    def __init__(self, axial: nn.Module, skip: str | None):
        self.axial = axial
        self.skip = skip
        self._orig = axial.forward

    def __enter__(self):
        skip = self.skip
        axial = self.axial

        def fwd(x: Tensor) -> Tensor:
            if skip != "row":
                x = x + _attend_rows(axial.row_attn, axial.ln_row(x))
            if skip != "col":
                x = x + _attend_cols(axial.col_attn, axial.ln_col(x))
            if skip != "box":
                x = x + _attend_boxes(axial.box_attn, axial.ln_box(x))
            if skip != "mlp":
                x = x + axial.mlp(axial.ln_mlp(x))
            return x

        axial.forward = fwd  # type: ignore[method-assign]
        return self

    def __exit__(self, *exc):
        self.axial.forward = self._orig  # type: ignore[method-assign]
        return False


@torch.no_grad()
def collect_activations(
    model: RiemannianSudokuSolver,
    loader: DataLoader,
    device: torch.device,
    max_boards: int,
) -> dict:
    """Collect s0, s_final, proj, h, and per-branch residual energies."""
    s0_rows: list[Tensor] = []
    s_final_rows: list[Tensor] = []
    s_blank_rows: list[Tensor] = []
    proj_rows: list[Tensor] = []
    h_rows: list[Tensor] = []
    # digit orthants of final s (blank cells only): per digit argmax bucket
    digit_buckets: dict[int, list[Tensor]] = {d: [] for d in range(9)}
    branch_energy = {"row": 0.0, "col": 0.0, "box": 0.0, "mlp": 0.0, "n": 0}
    mix_before: list[dict] = []
    mix_after: list[dict] = []
    attn_stats: dict[str, list[dict]] = {"row": [], "col": [], "box": []}
    groups = sudoku_group_indices().to(device)

    n_seen = 0
    for batch in loader:
        if n_seen >= max_boards:
            break
        puzzle = batch["puzzle"].to(device)
        bsz = puzzle.shape[0]
        take = min(bsz, max_boards - n_seen)
        puzzle = puzzle[:take]
        mask = (puzzle == 0)

        s0, blank_mask = init_sphere_state(puzzle)
        pe = model.t3_pe.unsqueeze(0).expand(puzzle.shape[0], -1, -1)
        x_in = torch.cat([s0, pe], dim=-1)
        proj = model.proj(x_in)

        # one axial step with branch energy + attn entropy (first ODE step only)
        axial = model.axial
        x = proj
        x_ln = axial.ln_row(x)
        d_row = _attend_rows(axial.row_attn, x_ln)
        attn_stats["row"].append(_attention_entropy(axial.row_attn, x_ln, "row"))
        x1 = x + d_row
        x_ln = axial.ln_col(x1)
        d_col = _attend_cols(axial.col_attn, x_ln)
        attn_stats["col"].append(_attention_entropy(axial.col_attn, x_ln, "col"))
        x2 = x1 + d_col
        x_ln = axial.ln_box(x2)
        d_box = _attend_boxes(axial.box_attn, x_ln)
        attn_stats["box"].append(_attention_entropy(axial.box_attn, x_ln, "box"))
        x3 = x2 + d_box
        d_mlp = axial.mlp(axial.ln_mlp(x3))
        h = x3 + d_mlp

        for name, delta in (("row", d_row), ("col", d_col), ("box", d_box), ("mlp", d_mlp)):
            branch_energy[name] += float(delta.pow(2).mean().item()) * take
        branch_energy["n"] += take

        mix_before.append(_group_mean_pairwise_cosine(s0, groups))

        # one Exp step from first velocity
        v = model.velocity(h)
        v = project_tangent(s0, v) * blank_mask.unsqueeze(-1)
        s1 = riemannian_exp_map_orthant(s0, v, dt=model.dt)
        # re-clamp clues
        clues = blank_mask == 0
        digit_idx = (puzzle.clamp(min=1) - 1).clamp(0, 8)
        s_clue = torch.zeros_like(s1)
        s_clue.scatter_(-1, digit_idx.unsqueeze(-1), 1.0)
        s1 = torch.where(clues.unsqueeze(-1), s_clue, s1)
        mix_after.append(_group_mean_pairwise_cosine(s1, groups))

        # full integrate for final s
        out = model(puzzle)
        sf = out["s"]

        # flatten cells as rows
        s0_rows.append(s0.reshape(-1, 9).cpu())
        s_final_rows.append(sf.reshape(-1, 9).cpu())
        proj_rows.append(proj.reshape(-1, proj.shape[-1]).cpu())
        h_rows.append(h.reshape(-1, h.shape[-1]).cpu())
        blank_flat = mask.reshape(-1)
        s_blank_rows.append(sf.reshape(-1, 9)[blank_flat].cpu())
        # digit buckets on blank cells by predicted digit
        pred_d = sf.reshape(-1, 9).argmax(-1)
        sf_flat = sf.reshape(-1, 9)
        for d in range(9):
            sel = blank_flat & (pred_d == d)
            if sel.any():
                digit_buckets[d].append(sf_flat[sel].cpu())

        n_seen += take

    def cat(xs: list[Tensor]) -> Tensor:
        return torch.cat(xs, dim=0)

    def mean_mix(items: list[dict]) -> dict:
        keys = items[0].keys()
        return {k: float(sum(d[k] for d in items) / len(items)) for k in keys}

    def mean_attn(items: list[dict]) -> dict:
        keys = [k for k in items[0] if k != "n_heads"]
        out = {k: float(sum(d[k] for d in items) / len(items)) for k in keys}
        out["n_heads"] = items[0]["n_heads"]
        return out

    n = max(branch_energy["n"], 1)
    digit_rank = {}
    for d, parts in digit_buckets.items():
        if not parts:
            digit_rank[str(d)] = {"n_rows": 0, "eff_rank": 0.0, "participation_ratio": 0.0}
            continue
        digit_rank[str(d)] = spectrum_metrics(cat(parts))

    return {
        "n_boards": n_seen,
        "ranks": {
            "s0_all_cells": spectrum_metrics(cat(s0_rows)),
            "s_final_all_cells": spectrum_metrics(cat(s_final_rows)),
            "s_final_blank": spectrum_metrics(cat(s_blank_rows)),
            "proj_first_step": spectrum_metrics(cat(proj_rows)),
            "h_axial_first_step": spectrum_metrics(cat(h_rows)),
            "s_final_by_pred_digit_blank": digit_rank,
        },
        "branch_activation_energy": {
            k: branch_energy[k] / n for k in ("row", "col", "box", "mlp")
        },
        "sphere_mixing": {
            "before_first_step": mean_mix(mix_before),
            "after_first_exp": mean_mix(mix_after),
        },
        "attention": {k: mean_attn(v) for k, v in attn_stats.items()},
    }


def branch_grad_energy(
    model: RiemannianSudokuSolver,
    loader: DataLoader,
    device: torch.device,
    max_boards: int = 128,
) -> dict[str, float]:
    """Mean ||grad||² into each axial sub-module over a few batches."""
    model.train()  # need grads through dropout-less path; modules have no dropout
    keys = {
        "row": list(model.axial.row_attn.parameters()),
        "col": list(model.axial.col_attn.parameters()),
        "box": list(model.axial.box_attn.parameters()),
        "mlp": list(model.axial.mlp.parameters()),
        "proj": list(model.proj.parameters()),
        "velocity": list(model.velocity.parameters()),
    }
    acc = {k: 0.0 for k in keys}
    n = 0
    for batch in loader:
        if n >= max_boards:
            break
        puzzle = batch["puzzle"].to(device)
        solution = batch["solution"].to(device)
        mask = batch["mask"].to(device)
        take = min(puzzle.shape[0], max_boards - n)
        puzzle, solution, mask = puzzle[:take], solution[:take], mask[:take]
        model.zero_grad(set_to_none=True)
        out = model(puzzle, solution, mask)
        out["loss"].backward()
        for name, params in keys.items():
            e = 0.0
            for p in params:
                if p.grad is not None:
                    e += float(p.grad.detach().pow(2).sum().item())
            acc[name] += e
        n += take
    model.eval()
    return {k: v / max(n, 1) for k, v in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=ROOT / "checkpoints/scrutiny-confirm/best.pt",
    )
    ap.add_argument("--val-size", type=int, default=2048)
    ap.add_argument("--train-size", type=int, default=100000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-act-boards", type=int, default=512)
    ap.add_argument("--max-grad-boards", type=int, default=128)
    ap.add_argument("--out", type=Path, default=Path("/tmp/riemannian_sudoku_model_health.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_model(args.ckpt, device)
    print(f"loaded step={meta['step']} best_blank={meta['best_blank']} device={device}", flush=True)

    ds = SudokuDataset()
    df = ds.load_split("train", n_samples=args.train_size + args.val_size)
    _, val_df = split_train_val(df, val_size=args.val_size, seed=args.seed)
    val_loader = DataLoader(
        SudokuTorchDataset(val_df, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
    )

    # --- weight SVD ---
    weight_svds = [
        weight_svd_metrics(model.proj.weight, "proj"),
        weight_svd_metrics(model.axial.mlp[0].weight, "axial_mlp_fc1"),
        weight_svd_metrics(model.axial.mlp[2].weight, "axial_mlp_fc2"),
        weight_svd_metrics(model.velocity[0].weight, "velocity_fc1"),
        weight_svd_metrics(model.velocity[2].weight, "velocity_fc2"),
    ]

    # --- activations / mixing ---
    print("collecting activations…", flush=True)
    act = collect_activations(model, val_loader, device, args.max_act_boards)

    # --- grads ---
    print("branch grad energy…", flush=True)
    grads = branch_grad_energy(model, val_loader, device, args.max_grad_boards)

    # --- ablations ---
    print("eval ablations…", flush=True)
    baseline = evaluate(model, val_loader, device)
    ablations = {"full": baseline}
    for skip in ("row", "col", "box", "mlp"):
        with AxialAblator(model.axial, skip):
            m = evaluate(model, val_loader, device)
        ablations[f"skip_{skip}"] = m
        print(
            f"  skip_{skip}: blank={m['blank_acc']:.4f} row={m['row_acc']:.4f} "
            f"(Δblank={m['blank_acc']-baseline['blank_acc']:+.4f})",
            flush=True,
        )

    report = {
        "meta": meta,
        "device": str(device),
        "weight_svd": weight_svds,
        "activations": act,
        "branch_grad_energy": grads,
        "eval_ablate": ablations,
        "eval_ablate_delta_blank": {
            k: float(v["blank_acc"] - baseline["blank_acc"])
            for k, v in ablations.items()
            if k != "full"
        },
        "eval_ablate_delta_row": {
            k: float(v["row_acc"] - baseline["row_acc"])
            for k, v in ablations.items()
            if k != "full"
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}", flush=True)

    # compact stdout summary
    ranks = act["ranks"]
    for name in ("s0_all_cells", "proj_first_step", "h_axial_first_step", "s_final_blank"):
        r = ranks[name]
        print(
            f"rank {name}: ER={r['eff_rank']:.2f} PR={r['participation_ratio']:.2f} "
            f"k90={r['k90']} dead={r['dead_cols']} d={r['d']}",
            flush=True,
        )
    print("weight SVD ER:", {w["name"]: round(w["eff_rank"], 2) for w in weight_svds}, flush=True)
    print("branch act energy:", act["branch_activation_energy"], flush=True)
    print("branch grad energy:", grads, flush=True)
    print("sphere mix before:", act["sphere_mixing"]["before_first_step"], flush=True)
    print("sphere mix after:", act["sphere_mixing"]["after_first_exp"], flush=True)


if __name__ == "__main__":
    main()
