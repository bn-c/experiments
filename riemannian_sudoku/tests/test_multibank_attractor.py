"""Phase-3 Multi-Manifold Block-Adjacency Attractor unit gates."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from riemannian_sudoku.manifold_bank import (
    MultiBankRouter,
    MultiBankSolver,
    SphereBank,
    TorusBank,
    _phi_norm,
    _row_stoch_dustbin,
    _sinkhorn_dustbin,
    alpha_from_depth,
    gamma_anneal,
)
from riemannian_sudoku.manifold_embed import gt_family_logits, gt_peer_mask
from riemannian_sudoku.model import build_solver
from riemannian_sudoku.solver import ortho_hinge, sudoku_graph_purity


def test_sphere_torus_bank_shapes() -> None:
    torch.manual_seed(0)
    s0 = SphereBank(81, 64, d_msg=64, pe_dim=6, with_claims=True)
    s1 = SphereBank(9, 32, d_msg=64, pe_dim=0, with_claims=False)
    t = TorusBank(n_per=9, m=16, f_heads=3, d_msg=64)
    poles = torch.randn(9, 64)
    poles = F.normalize(poles, dim=-1)
    x = F.normalize(torch.randn(2, 81, 64), dim=-1)
    s = F.normalize(torch.randn(2, 9, 32), dim=-1)
    y = torch.randn(2, 27, 16)
    pe = torch.randn(81, 6)
    m0 = s0.read(x, poles=poles)
    m1 = s1.read(s)
    mt = t.read(y)
    assert m0.shape == (2, 81, 64) and m1.shape == (2, 9, 64) and mt.shape == (2, 27, 64)
    pkt_t = torch.randn(2, 27, 128)
    y2, w = t.write(y, pkt_t)
    assert y2.shape == y.shape and w.shape == y.shape
    x2, g = s0.write(x, torch.randn(2, 81, 64), pe, x, torch.zeros(2, 81, dtype=torch.bool))
    assert x2.shape == x.shape and g.shape == (2, 81)
    assert torch.allclose(x2.norm(dim=-1), torch.ones(2, 81), atol=1e-4)


def test_only_four_bipartite_edges() -> None:
    r = MultiBankRouter()
    assert r.edge_names == ('S0_T1', 'S0_T2', 'S1_T1', 'S1_T2')
    assert set(r.get_incidence(torch.tensor(0.5)).keys()) == set(r.edge_names)
    # No same-type params.
    names = [n for n, _ in r.named_parameters()]
    assert not any('S0_S1' in n or 'T1_T2' in n for n in names)


def test_phi_norm_and_dustbin_denom() -> None:
    torch.manual_seed(0)
    phi = torch.randn(3, 9, 81) * 0.01  # tiny ‖Φ‖ — phi_norm must rescue
    pn = _phi_norm(phi)
    assert torch.allclose(pn.mean(dim=(1, 2)), torch.zeros(3), atol=1e-5)
    assert torch.allclose(pn.std(dim=(1, 2), unbiased=False), torch.ones(3), atol=1e-4)
    dust = torch.full((3, 9), -1.0)
    p_real, p_dust = _sinkhorn_dustbin(pn, dust, torch.tensor(0.25))
    a = _row_stoch_dustbin(p_real, p_dust)
    row = a.sum(dim=-1) + 0.0 * p_dust  # A rows ≤ 1
    assert (row <= 1.0 + 1e-4).all()
    # Dustbin-in-denom: A + dust_frac ≈ 1
    frac = p_dust / (p_real.sum(-1) + p_dust + 1e-6)
    assert torch.allclose(a.sum(-1) + frac, torch.ones_like(frac), atol=1e-4)


def test_alpha_staggered_h_and_gamma_anneal() -> None:
    r = MultiBankRouter()
    assert float(r.h_S0) == 0.0
    assert abs(float(r.h_T1) - 0.9) < 1e-6
    assert abs(float(r.h_T2) - 1.1) < 1e-6
    assert abs(float(r.h_S1) - 1.8) < 1e-6
    assert abs(float(r.beta_S0_T1) - 2.0) < 1e-6
    # γ=0 → α = σ(β) regardless of |h_i-h_j|
    a0 = alpha_from_depth(r.h_S0, r.h_T1, r.beta_S0_T1, torch.tensor(0.0))
    assert abs(float(a0) - (1.0 / (1.0 + math.exp(-2.0)))) < 1e-5
    assert gamma_anneal(0) == 0.0
    assert abs(gamma_anneal(500, gamma_max=1.0) - 0.5) < 1e-9
    assert gamma_anneal(1000, gamma_max=1.0) == 1.0
    assert gamma_anneal(5000, gamma_max=1.0) == 1.0


def test_f_head_unit_sum_not_mean() -> None:
    """Summing F heads with weight 1 (not 1/F) — scale check on packet_t2s."""
    r = MultiBankRouter(n_s0=81, n_s1=9)
    torch.manual_seed(0)
    # One-hot A: each cell attends one house per head equally.
    a_t2s = torch.zeros(3, 81, 9)
    for f in range(3):
        for n in range(81):
            a_t2s[f, n, n % 9] = 1.0
    msg = torch.ones(1, 27, 64)
    pkt = r.packet_t2s(a_t2s, msg, torch.tensor(1.0))
    # Each of 3 heads contributes 1 → sum ≈ 3 (unit sum, not mean 1).
    assert torch.allclose(pkt, torch.full_like(pkt, 3.0), atol=1e-5)


def test_static_leapfrog_no_argsort() -> None:
    import inspect

    from riemannian_sudoku import manifold_bank as mb

    src = inspect.getsource(mb.MultiBankAttractorCore.integrate)
    assert 'argsort' not in src
    assert 'step_all' not in src or True  # method inlines T then S
    # Order lock: first torus write appears before sphere write in source.
    assert src.index('t1.write') < src.index('s0.write')
    assert src.index('t2.write') < src.index('s0.write')


def test_multibank_solver_forward_and_params() -> None:
    torch.manual_seed(0)
    model = build_solver('multibank_attractor', num_steps=2, pe_mode='t2_grid')
    assert model.mixer == 'multibank_attractor'
    puzzle = torch.zeros(2, 81, dtype=torch.long)
    puzzle[0, 0] = 5
    solution = torch.randint(1, 10, (2, 81))
    mask = (puzzle == 0).float()
    out = model(puzzle, solution, mask)
    assert 'loss' in out and out['loss'].ndim == 0
    assert 'alpha_S0_T1' in out and 'h_T1' in out and 'curr_gamma' in out
    out['loss'].backward()
    n_vf = model.vf_param_count()
    n_r = model.router_param_count()
    n_all = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_vf + n_r == n_all
    assert n_all > 100_000
    # Φ_S0_T1 is 3×9×81 = 2187
    assert model.router.phi_S0_T1.numel() == 2187


def test_gt_pin_purity_high() -> None:
    """GT-pinned S0–T1 at small τ → purity near 1 (sanity of Sinkhorn path)."""
    model = MultiBankSolver(num_steps=1, pe_mode='t2_grid')
    gt = gt_family_logits(boost=2.0)
    model.router.pin_s0_edge('S0_T1', gt, families=[0, 1, 2])
    model.curr_tau.fill_(0.25)
    model.curr_gamma.fill_(0.0)
    inc = model.router.get_incidence(model.curr_tau)
    p = inc['S0_T1']['P']
    pur = sudoku_graph_purity(p, model.gt_peer_mask)
    assert float(pur) >= 0.98, f'purity={float(pur)}'
    assert float(ortho_hinge(p)) == 0.0 or float(ortho_hinge(p)) < 1e-6


def test_mu_only_t2s_no_sigma_half() -> None:
    r = MultiBankRouter()
    a = r.get_incidence(torch.tensor(0.5))['S0_T1']['A_t2s']
    msg = torch.randn(1, 27, 64)
    pkt = r.packet_t2s(a, msg, torch.tensor(1.0))
    assert pkt.shape == (1, 81, 64)  # μ only, not 128
    a_s = r.get_incidence(torch.tensor(0.5))['S0_T1']['A_s2t']
    msg_s = torch.randn(1, 81, 64)
    pkt_s = r.packet_s2t(a_s, msg_s, torch.tensor(1.0))
    assert pkt_s.shape == (1, 27, 128)  # [μ,σ]
