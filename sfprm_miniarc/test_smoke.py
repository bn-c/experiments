#!/usr/bin/env python3
"""Cheap unit/smoke checks (no GPU, no full train)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.sfprm_core import DEFAULT_D_FF, DEFAULT_D_MODEL, DEFAULT_N_LAYERS, SFPRM
from models.solver import SolverConfig, SolverState, run_segment
from models.spherical_ops import chordal_residual, guarded_slerp, project_to_sphere, slerp
from optim.muon import Muon, zeropower_via_newtonschulz5


def test_spherical_ops_shapes():
    u = project_to_sphere(torch.randn(2, 5, 8))
    v = project_to_sphere(torch.randn(2, 5, 8))
    lam = torch.tensor(0.3)
    out = slerp(u, v, lam)
    assert out.shape == u.shape
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    # Guarded near-coincident / antipodal branches
    near = guarded_slerp(u, u, torch.tensor(0.5))
    assert torch.allclose(near, u, atol=1e-4)
    anti = guarded_slerp(u, -u, torch.tensor(0.25))
    assert anti.shape == u.shape
    r = chordal_residual(u, v)
    assert r.shape == (2,)


def test_model_forward_and_param_count():
    model = SFPRM(vocab_size=12, d_model=64, n_heads=4, d_ff=128, n_layers=2, max_seq_len=32, n_puzzles=4)
    x = torch.randint(0, 12, (2, 32))
    pid = torch.tensor([0, 1])
    z_in = model.get_input_embedding(x, pid)
    assert z_in.shape == (2, 33, 64)  # +1 puzzle emb
    eta = torch.ones(2)
    z, _prop, res = model.step(z_in, z_in, eta)
    assert z.shape == z_in.shape
    assert res.shape == (2,)
    logits = model.decode_logits(z)
    assert logits.shape[-1] == 10
    assert model.count_parameters() > 0
    # λ init ≈ 0.1
    assert abs(float(model.lambda_param.item()) - 0.1) < 1e-5


def test_desk_width_param_ballpark():
    model = SFPRM(
        vocab_size=12,
        d_model=DEFAULT_D_MODEL,
        d_ff=DEFAULT_D_FF,
        n_layers=DEFAULT_N_LAYERS,
        max_seq_len=128,
        n_puzzles=149,
    )
    n = model.count_parameters()
    # Desk 384/8/1536/L=2 SwiGLU ≈ 4.5–5.5M (paper 512/… is ~6.83–8.5M)
    assert 4.0e6 < n < 6.0e6, n


def test_solver_patience_decays_eta():
    model = SFPRM(vocab_size=12, d_model=32, n_heads=4, d_ff=64, n_layers=1, max_seq_len=16, n_puzzles=2)
    x = torch.randint(0, 12, (2, 16))
    z_in = model.get_input_embedding(x, torch.zeros(2, dtype=torch.long))
    z = z_in.clone()
    cfg = SolverConfig(patience=2, gamma=0.5, eta_min=0.001, tau=1e-12, max_iter=20)
    state = SolverState.init(2, z.device, cfg)
    z, state, _ = run_segment(model, z, z_in, state, cfg, k=6)
    # With absurdly tight tau, residual won't halt; patience should decay η
    assert float(state.eta.max().item()) <= 1.0
    assert float(state.eta.min().item()) < 1.0 or bool(state.halted.any())


def test_muon_step():
    p = torch.nn.Parameter(torch.randn(16, 8))
    opt = Muon([p], lr=0.02, momentum=0.95)
    loss = (p**2).sum()
    loss.backward()
    opt.step()
    g = torch.randn(8, 8)
    o = zeropower_via_newtonschulz5(g)
    assert o.shape == g.shape


def main() -> int:
    test_spherical_ops_shapes()
    test_model_forward_and_param_count()
    test_desk_width_param_ballpark()
    test_solver_patience_decays_eta()
    test_muon_step()
    print("test_smoke: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
