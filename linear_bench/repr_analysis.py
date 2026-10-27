"""Cheap per-epoch representation / DoF probes (``repr/*`` metrics).

Weight matrix choice
--------------------
- ``linear``: the single ``nn.Linear.weight`` (shape ``out × in``).
- ``mlp`` (``nn.Sequential``): **last** ``nn.Linear`` before logits/outputs
  (hidden→out). Matches the class-head / recovery head people usually probe.

Activation features
-------------------
- ``linear``: model outputs (logits / regression preds) on a fixed probe batch.
- ``mlp``: post-activation penultimate hidden (input to the last Linear).

Effective rank uses the participation ratio of singular values::

    eff_rank = (Σ σ²)² / Σ σ⁴

``weight_rank_ratio = eff_rank / min(rows, cols)``.

LRB ``feature_align`` runs only when meta carries ``W_true`` (or ``A``): mean
absolute row-cosine of the primary weight vs ``W_true.T`` when shapes match;
otherwise linear CKA between probe hidden features and continuous targets
``y`` (synthetic codes). Skipped cleanly when neither applies.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def participation_ratio(singular_values: torch.Tensor) -> float:
    """Participation ratio / soft rank of a singular-value spectrum."""
    s = singular_values.detach().float().flatten().clamp_min(0.0)
    s2 = s * s
    num = float(s2.sum().item() ** 2)
    den = float((s2 * s2).sum().item())
    if den <= 0.0:
        return 0.0
    return num / den


def primary_weight(model: nn.Module) -> nn.Parameter:
    """Primary weight matrix: Linear.weight, or last Linear in a Sequential MLP."""
    if isinstance(model, nn.Linear):
        return model.weight
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not linears:
        raise ValueError("no nn.Linear found for weight probe")
    return linears[-1].weight


def _penultimate_hidden(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Post-activation features into the last Linear (MLP); else model(x)."""
    if isinstance(model, nn.Linear):
        return model(x)
    if isinstance(model, nn.Sequential):
        linears = [(i, m) for i, m in enumerate(model) if isinstance(m, nn.Linear)]
        if len(linears) >= 2:
            last_i = linears[-1][0]
            h = x
            for layer in model[:last_i]:
                h = layer(h)
            return h
        return model(x)
    return model(x)


def fixed_probe_batch(
    loader: DataLoader,
    *,
    device: torch.device,
    max_samples: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """First ``max_samples`` from ``loader`` (fixed across epochs)."""
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    n = 0
    for x, y in loader:
        xs.append(x)
        ys.append(y)
        n += int(x.shape[0])
        if n >= max_samples:
            break
    if not xs:
        raise ValueError("empty loader for repr probe batch")
    x = torch.cat(xs, dim=0)[:max_samples].to(device)
    y = torch.cat(ys, dim=0)[:max_samples].to(device)
    return x, y


def _mean_row_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = F.normalize(a.float(), dim=1, eps=1e-8)
    b = F.normalize(b.float(), dim=1, eps=1e-8)
    return float((a * b).sum(dim=1).abs().mean().item())


def _linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear CKA between two (n, d) matrices (centered)."""
    x = x.float() - x.float().mean(dim=0, keepdim=True)
    y = y.float() - y.float().mean(dim=0, keepdim=True)
    xtx = x.T @ x
    yty = y.T @ y
    xty = x.T @ y
    hsic_xy = float((xty * xty).sum().item())
    hsic_xx = float((xtx * xtx).sum().item())
    hsic_yy = float((yty * yty).sum().item())
    den = (hsic_xx * hsic_yy) ** 0.5
    if den <= 0.0:
        return 0.0
    return hsic_xy / den


def _as_weight_matrix(obj: Any) -> torch.Tensor | None:
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.detach().float()
    if isinstance(obj, np.ndarray):
        return torch.as_tensor(obj, dtype=torch.float32)
    return None


@torch.no_grad()
def compute_repr_metrics(
    model: nn.Module,
    *,
    probe_x: torch.Tensor,
    probe_y: torch.Tensor | None = None,
    meta: dict[str, Any] | None = None,
    dead_eps: float = 1e-5,
) -> dict[str, float]:
    """Return flat ``repr/*`` scalars for one epoch (model in eval mode)."""
    model.eval()
    w = primary_weight(model).detach().float()
    # SVD on CPU — matrices are small (linear / last MLP layer).
    s = torch.linalg.svdvals(w.cpu())
    eff = participation_ratio(s)
    rank_cap = float(min(w.shape))
    out: dict[str, float] = {
        "repr/weight_eff_rank": eff,
        "repr/weight_rank_ratio": (eff / rank_cap) if rank_cap > 0 else 0.0,
    }

    feats = _penultimate_hidden(model, probe_x)
    if feats.ndim == 1:
        feats = feats.unsqueeze(0)
    # Activation DoF via Gram participation on the feature dim.
    # Center then SVD of (n × d) — cheap for probe batch sizes.
    f = feats.detach().float()
    f = f - f.mean(dim=0, keepdim=True)
    if f.shape[0] >= 2 and f.shape[1] >= 1:
        s_act = torch.linalg.svdvals(f.cpu())
        out["repr/act_participation_ratio"] = participation_ratio(s_act)
    else:
        out["repr/act_participation_ratio"] = 0.0

    # Dead units: mean |act| across batch below eps.
    mean_abs = feats.detach().float().abs().mean(dim=0)
    out["repr/dead_frac"] = float((mean_abs < dead_eps).float().mean().item())

    meta = meta or {}
    w_true = _as_weight_matrix(meta.get("W_true", meta.get("A")))
    align: float | None = None
    if w_true is not None:
        w_true = w_true.to(device=torch.device("cpu"))
        wc = w.cpu()
        if wc.shape == w_true.shape:
            align = _mean_row_cosine(wc, w_true)
        elif wc.shape == w_true.T.shape:
            align = _mean_row_cosine(wc, w_true.T)
    if align is None and probe_y is not None and probe_y.ndim == 2 and int(probe_y.shape[1]) > 1:
        # Continuous codes (synthetic LRB): CKA(hidden, z_true).
        align = _linear_cka(feats.cpu(), probe_y.detach().cpu())
    if align is not None:
        out["repr/feature_align"] = float(align)

    return out


def serializable_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Drop large ground-truth matrices from JSON-facing meta."""
    skip = {"W_true", "A"}
    out: dict[str, Any] = {}
    for k, v in meta.items():
        if k in skip:
            if isinstance(v, (np.ndarray, torch.Tensor)):
                out[f"{k}_shape"] = list(v.shape)
            continue
        out[k] = v
    return out
