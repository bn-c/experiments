#!/usr/bin/env python3
"""Torus-only unit/smoke checks (sphere test_smoke.py untouched)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.sfprm_torus_core import DEFAULT_D_FF, DEFAULT_D_MODEL, DEFAULT_N_LAYERS, SFPRMTorus
from models.solver import SolverConfig, SolverState, run_segment
from models.toroidal_ops import (
    _as_pairs,
    _pair_radius,
    project_to_torus,
    torus_interp,
    torus_residual,
)


def _pair_norms(x: torch.Tensor) -> torch.Tensor:
    """Half-split pair L2 norms (RoPE-aligned)."""
    return _as_pairs(x).norm(dim=-1)


def _expect_pair_radius(d: int) -> float:
    return _pair_radius(d // 2)


def test_toroidal_ops_on_manifold():
    u = project_to_torus(torch.randn(2, 5, 8))
    v = project_to_torus(torch.randn(2, 5, 8))
    r = _expect_pair_radius(8)
    assert torch.allclose(_pair_norms(u), torch.full_like(_pair_norms(u), r), atol=1e-5)
    assert torch.allclose(u.norm(dim=-1), torch.ones(2, 5), atol=1e-5)

    out = torus_interp(u, v, torch.tensor(0.3))
    assert out.shape == u.shape
    assert torch.allclose(_pair_norms(out), torch.full_like(_pair_norms(out), r), atol=1e-5)
    assert torch.allclose(out.norm(dim=-1), torch.ones(2, 5), atol=1e-5)

    near = torus_interp(u, u, torch.tensor(0.5))
    assert torch.allclose(near, u, atol=1e-4)

    r0 = torus_residual(u, u)
    assert r0.shape == (2,)
    assert torch.allclose(r0, torch.zeros_like(r0), atol=1e-5)

    res = torus_residual(u, v)
    assert res.shape == (2,)
    assert float(res.min()) >= -1e-5
    assert float(res.max()) <= 2.0 + 1e-5


def test_torus_residual_scale_matches_sphere_range():
    z = project_to_torus(torch.randn(3, 4, 16))
    assert float(torus_residual(z, z).max()) < 1e-5
    assert float(torus_residual(z, -z).min()) > 1.9


def test_torus_model_forward_and_renorm():
    model = SFPRMTorus(
        vocab_size=12, d_model=64, n_heads=4, d_ff=128, n_layers=2, max_seq_len=32, n_puzzles=4
    )
    x = torch.randint(0, 12, (2, 32))
    pid = torch.tensor([0, 1])
    z_in = model.get_input_embedding(x, pid)
    assert z_in.shape == (2, 33, 64)
    r = _expect_pair_radius(64)
    assert torch.allclose(_pair_norms(z_in), torch.full_like(_pair_norms(z_in), r), atol=1e-5)
    assert torch.allclose(z_in.norm(dim=-1), torch.ones(2, 33), atol=1e-5)

    eta = torch.ones(2)
    z, _prop, res = model.step(z_in, z_in, eta)
    assert z.shape == z_in.shape
    assert res.shape == (2,)
    assert float(res.detach().max()) <= 2.0 + 1e-5
    assert torch.allclose(_pair_norms(z), torch.full_like(_pair_norms(z), r), atol=1e-4)
    assert torch.allclose(z.norm(dim=-1), torch.ones_like(z.norm(dim=-1)), atol=1e-4)

    with torch.no_grad():
        model.token_emb.weight.add_(0.5)
        model.puzzle_emb.weight.add_(0.5)
    model.renorm_weights()
    w = model.token_emb.weight
    assert torch.allclose(_pair_norms(w), torch.full_like(_pair_norms(w), r), atol=1e-5)
    assert torch.allclose(w.norm(dim=-1), torch.ones(w.shape[0]), atol=1e-5)
    pe_norms = _pair_norms(model.puzzle_emb.weight)
    assert not torch.allclose(pe_norms, torch.full_like(pe_norms, r), atol=1e-2)
    assert abs(float(model.lambda_param.item()) - 0.1) < 1e-5


def test_torus_desk_width_param_ballpark():
    model = SFPRMTorus(
        vocab_size=12,
        d_model=DEFAULT_D_MODEL,
        d_ff=DEFAULT_D_FF,
        n_layers=DEFAULT_N_LAYERS,
        max_seq_len=25,
        n_puzzles=149,
    )
    n = model.count_parameters()
    assert 4.0e6 < n < 6.0e6, n


def test_torus_solver_segment():
    model = SFPRMTorus(
        vocab_size=12, d_model=32, n_heads=4, d_ff=64, n_layers=1, max_seq_len=16, n_puzzles=2
    )
    x = torch.randint(0, 12, (2, 16))
    z_in = model.get_input_embedding(x, torch.zeros(2, dtype=torch.long))
    cfg = SolverConfig(patience=2, gamma=0.5, eta_min=0.001, tau=0.01, max_iter=20)
    state = SolverState.init(2, z_in.device, cfg)
    z, state, _ = run_segment(model, z_in.clone(), z_in, state, cfg, k=2)
    assert z.shape == z_in.shape
    r = _expect_pair_radius(32)
    assert torch.allclose(_pair_norms(z), torch.full_like(_pair_norms(z), r), atol=1e-4)


def test_odd_d_model_rejected():
    try:
        SFPRMTorus(vocab_size=12, d_model=33, n_heads=3, d_ff=66, n_layers=1, max_seq_len=8)
    except ValueError:
        return
    raise AssertionError("expected ValueError for odd d_model")


def test_antipodal_interp():
    torch.manual_seed(0)
    u = project_to_torus(torch.randn(64, 8))
    r = _expect_pair_radius(8)
    mid = torus_interp(u, -u, torch.tensor(0.5))
    assert torch.allclose(_pair_norms(mid), torch.full_like(_pair_norms(mid), r), atol=1e-5)
    up = _as_pairs(u)
    mp = _as_pairs(mid)
    dots = (up * mp).sum(dim=-1)
    assert torch.allclose(dots, torch.zeros_like(dots), atol=1e-4), float(dots.abs().max())
    end = torus_interp(u, -u, torch.tensor(1.0))
    assert torch.allclose(end, -u, atol=1e-4)


def test_rope_preserves_halfsplit_pairs():
    """Per-head project + RoPE stays on-manifold (no reproject)."""
    from models.sfprm_torus_core import apply_rope

    torch.manual_seed(2)
    d, h, b, s = 64, 4, 2, 8
    dh = d // h
    model = SFPRMTorus(
        vocab_size=12, d_model=d, n_heads=h, d_ff=128, n_layers=1, max_seq_len=s, n_puzzles=2
    ).eval()
    x = torch.randint(0, 12, (b, s))
    z = model.get_input_embedding(x).detach()
    layer = model.core_block.layers[0]
    with torch.no_grad():
        seq = z.shape[1]
        qh = layer._linear(layer.q_proj, z).view(b, seq, h, dh).transpose(1, 2)
        qh = project_to_torus(qh)
        rh = _expect_pair_radius(dh)
        # Per-head pair norms + ‖q_h‖₂=1 before head_amb scale
        assert torch.allclose(_pair_norms(qh), torch.full_like(_pair_norms(qh), rh), atol=1e-5)
        assert torch.allclose(qh.norm(dim=-1), torch.ones(b, h, seq), atol=1e-5)
        cos, sin = layer.rope(seq)
        cos, sin = cos.to(qh.dtype), sin.to(qh.dtype)
        qh2, _ = apply_rope(qh, qh, cos, sin)
        assert torch.allclose(_pair_norms(qh2), torch.full_like(_pair_norms(qh2), rh), atol=1e-4)
        assert torch.allclose(qh2.norm(dim=-1), torch.ones(b, h, seq), atol=1e-4)



def test_sqk_clamp_min_one():
    model = SFPRMTorus(
        vocab_size=12, d_model=32, n_heads=4, d_ff=64, n_layers=1, max_seq_len=8, n_puzzles=2
    )
    with torch.no_grad():
        model.core_block.layers[0].s_qk.fill_(-0.5)
    x = torch.randint(0, 12, (2, 8))
    z = model.get_input_embedding(x)
    # Forward must not crash; clamp_min(1) used for logits (param itself may be negative).
    out = model.core_block.layers[0](z)
    assert out.shape == z.shape
    assert torch.isfinite(out).all()


def test_qk_ambient_unit():
    """Latent unit ambient; after per-head project + 1/√H, full ‖q‖₂≈1."""
    torch.manual_seed(1)
    d, h = 64, 4
    model = SFPRMTorus(
        vocab_size=12, d_model=d, n_heads=h, d_ff=128, n_layers=1, max_seq_len=8, n_puzzles=2
    ).eval()
    x = torch.randint(0, 12, (4, 8))
    z = model.get_input_embedding(x).detach()
    assert torch.allclose(z.norm(dim=-1), torch.ones(4, z.shape[1]), atol=1e-4)
    layer = model.core_block.layers[0]
    with torch.no_grad():
        b, seq = z.shape[0], z.shape[1]
        qh = layer._linear(layer.q_proj, z).view(b, seq, h, d // h).transpose(1, 2)
        qh = project_to_torus(qh) * (h**-0.5)
        q_flat = qh.transpose(1, 2).contiguous().view(b, seq, d)
        assert torch.allclose(q_flat.norm(dim=-1), torch.ones(b, seq), atol=1e-4)


def main() -> int:
    test_toroidal_ops_on_manifold()
    test_torus_residual_scale_matches_sphere_range()
    test_antipodal_interp()
    test_rope_preserves_halfsplit_pairs()
    test_sqk_clamp_min_one()
    test_torus_model_forward_and_renorm()
    test_qk_ambient_unit()
    test_torus_desk_width_param_ballpark()
    test_torus_solver_segment()
    test_odd_d_model_rejected()
    print("test_torus_smoke: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
