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
    project_to_torus,
    torus_interp,
    torus_residual,
)


def _pair_norms(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    return x.reshape(*x.shape[:-1], d // 2, 2).norm(dim=-1)


def test_toroidal_ops_on_manifold():
    u = project_to_torus(torch.randn(2, 5, 8))
    v = project_to_torus(torch.randn(2, 5, 8))
    assert torch.allclose(_pair_norms(u), torch.ones_like(_pair_norms(u)), atol=1e-5)

    out = torus_interp(u, v, torch.tensor(0.3))
    assert out.shape == u.shape
    assert torch.allclose(_pair_norms(out), torch.ones_like(_pair_norms(out)), atol=1e-5)

    near = torus_interp(u, u, torch.tensor(0.5))
    assert torch.allclose(near, u, atol=1e-4)

    r0 = torus_residual(u, u)
    assert r0.shape == (2,)
    assert torch.allclose(r0, torch.zeros_like(r0), atol=1e-5)

    r = torus_residual(u, v)
    assert r.shape == (2,)
    assert float(r.min()) >= -1e-5
    assert float(r.max()) <= 2.0 + 1e-5


def test_torus_residual_scale_matches_sphere_range():
    # Opposite pairs → mean chordal ≈ 2; identical → 0.
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
    assert torch.allclose(_pair_norms(z_in), torch.ones_like(_pair_norms(z_in)), atol=1e-5)

    eta = torch.ones(2)
    z, _prop, res = model.step(z_in, z_in, eta)
    assert z.shape == z_in.shape
    assert res.shape == (2,)
    assert float(res.detach().max()) <= 2.0 + 1e-5
    assert torch.allclose(_pair_norms(z), torch.ones_like(_pair_norms(z)), atol=1e-4)

    with torch.no_grad():
        model.token_emb.weight.add_(0.5)
        model.puzzle_emb.weight.add_(0.5)
    model.renorm_weights()
    w = model.token_emb.weight
    assert torch.allclose(_pair_norms(w), torch.ones_like(_pair_norms(w)), atol=1e-5)
    # Puzzle emb excluded from renorm (project-at-use only).
    pe_norms = _pair_norms(model.puzzle_emb.weight)
    assert not torch.allclose(pe_norms, torch.ones_like(pe_norms), atol=1e-2)
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
    assert torch.allclose(_pair_norms(z), torch.ones_like(_pair_norms(z)), atol=1e-4)


def test_odd_d_model_rejected():
    try:
        SFPRMTorus(vocab_size=12, d_model=33, n_heads=3, d_ff=66, n_layers=1, max_seq_len=8)
    except ValueError:
        return
    raise AssertionError("expected ValueError for odd d_model")


def test_antipodal_interp():
    """Antipodal guard: midpoint ⊥ u and on-torus; eta=1 → -u."""
    torch.manual_seed(0)
    u = project_to_torus(torch.randn(64, 8))
    mid = torus_interp(u, -u, torch.tensor(0.5))
    assert torch.allclose(_pair_norms(mid), torch.ones_like(_pair_norms(mid)), atol=1e-5)
    # Per-pair: mid · u ≈ 0
    up = u.reshape(*u.shape[:-1], u.shape[-1] // 2, 2)
    mp = mid.reshape(*mid.shape[:-1], mid.shape[-1] // 2, 2)
    dots = (up * mp).sum(dim=-1)
    assert torch.allclose(dots, torch.zeros_like(dots), atol=1e-4), float(dots.abs().max())
    end = torus_interp(u, -u, torch.tensor(1.0))
    assert torch.allclose(end, -u, atol=1e-4)


def test_qk_ambient_scale_matches_sphere():
    """After torus Q/K path, ambient ‖q‖₂ ≈ 1 (sphere cosine scale), not √(d/2)."""
    from models.sfprm_core import SFPRM
    from models.sfprm_torus_core import apply_rope
    from models.spherical_ops import project_to_sphere

    torch.manual_seed(1)
    d, h, b, s = 64, 4, 4, 8
    torus = SFPRMTorus(
        vocab_size=12, d_model=d, n_heads=h, d_ff=128, n_layers=1, max_seq_len=s, n_puzzles=2
    ).eval()
    sphere = SFPRM(
        vocab_size=12, d_model=d, n_heads=h, d_ff=128, n_layers=1, max_seq_len=s, n_puzzles=2
    ).eval()
    x = torch.randint(0, 12, (b, s))
    z_t = torus.get_input_embedding(x).detach()
    seq = z_t.shape[1]  # +puzzle emb
    z_s = project_to_sphere(z_t.clone())

    with torch.no_grad():
        layer_t = torus.core_block.layers[0]
        layer_s = sphere.core_block.layers[0]
        layer_s.load_state_dict(layer_t.state_dict())

        q_t = project_to_torus(layer_t._linear(layer_t.q_proj, z_t))
        q_t = q_t.view(b, seq, h, d // h).transpose(1, 2)
        cos, sin = layer_t.rope(seq)
        cos, sin = cos.to(q_t.dtype), sin.to(q_t.dtype)
        q_t, _ = apply_rope(q_t, q_t, cos, sin)
        q_t = project_to_torus(q_t) * ((d * 0.5) ** -0.5)
        q_flat = q_t.transpose(1, 2).contiguous().view(b, seq, d)
        assert torch.allclose(q_flat.norm(dim=-1), torch.ones(b, seq), atol=1e-4)
        # On interleaved torus after re-project+scale: pair norms = amb
        amb = (d * 0.5) ** -0.5
        assert torch.allclose(_pair_norms(q_flat), torch.full((b, seq, d // 2), amb), atol=1e-4)

        q_s = project_to_sphere(layer_s._linear(layer_s.q_proj, z_s))
        assert torch.allclose(q_s.norm(dim=-1), torch.ones(b, seq), atol=1e-4)


def main() -> int:
    test_toroidal_ops_on_manifold()
    test_torus_residual_scale_matches_sphere_range()
    test_antipodal_interp()
    test_torus_model_forward_and_renorm()
    test_qk_ambient_scale_matches_sphere()
    test_torus_desk_width_param_ballpark()
    test_torus_solver_segment()
    test_odd_d_model_rejected()
    print("test_torus_smoke: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
