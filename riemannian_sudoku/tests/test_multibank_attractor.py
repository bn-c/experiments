"""Phase-3 Multi-Manifold Block-Adjacency Attractor unit gates (v3.1)."""

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
    competition_loss,
    gamma_anneal,
    h_repel_loss,
    universal_clash_loss,
)
from riemannian_sudoku.manifold_embed import gt_family_logits, gt_peer_mask
from riemannian_sudoku.model import build_solver
from riemannian_sudoku.solver import ortho_hinge, sudoku_graph_purity

# Locked after v3.1 Read/Write reshape (router stays 4,975).
_VF_PARAMS_V31 = 958_435
_ROUTER_PARAMS = 4_975


def test_sphere_torus_bank_shapes() -> None:
    torch.manual_seed(0)
    s0 = SphereBank(81, 64, d_msg=64, pe_dim=6, ungated_sem=True)
    s1 = SphereBank(9, 32, d_msg=64, pe_dim=0, ungated_sem=False)
    t = TorusBank(n_per=9, m=16, f_heads=3, d_msg=64)
    x = F.normalize(torch.randn(2, 81, 64), dim=-1)
    s = F.normalize(torch.randn(2, 9, 32), dim=-1)
    y = torch.randn(2, 27, 16)
    pe = torch.randn(81, 6)
    m0 = s0.read(x)
    m1 = s1.read(s)
    mt = t.read(y)
    assert m0.shape == (2, 81, 64) and m1.shape == (2, 9, 64) and mt.shape == (2, 27, 64)
    assert abs(float(s0.read_proj.bias.detach().mean()) - (-0.2)) < 1e-5
    pkt_t = torch.randn(2, 27, 128)
    y2, w = t.write(y, pkt_t)
    assert y2.shape == y.shape and w.shape == y.shape
    house = torch.randn(2, 81, 128)  # cat[μ,σ]
    x2, g, v_c = s0.write(x, house, pe, x, torch.zeros(2, 81, dtype=torch.bool))
    assert x2.shape == x.shape and g.shape == (2, 81) and v_c.shape == x.shape
    assert torch.allclose(x2.norm(dim=-1), torch.ones(2, 81), atol=1e-4)


def test_read_pole_independence() -> None:
    """ETF poles must not enter SphereBank.read (adapter-only boundary)."""
    import inspect

    from riemannian_sudoku import manifold_bank as mb

    src = inspect.getsource(mb.SphereBank.read)
    body = src.split('"""', 2)[-1]  # skip docstring (may mention poles / LN)
    assert 'poles' not in body
    assert 'LayerNorm' not in body and 'layer_norm' not in body and 'F.normalize' not in body
    s0 = SphereBank(81, 64, d_msg=64, pe_dim=6, ungated_sem=True)
    x = F.normalize(torch.randn(2, 81, 64), dim=-1)
    assert s0.read(x).shape == (2, 81, 64)


def test_only_four_bipartite_edges() -> None:
    r = MultiBankRouter()
    assert r.edge_names == ('S0_T1', 'S0_T2', 'S1_T1', 'S1_T2')
    assert set(r.get_incidence(torch.tensor(0.5)).keys()) == set(r.edge_names)
    names = [n for n, _ in r.named_parameters()]
    assert not any('S0_S1' in n or 'T1_T2' in n for n in names)


def test_phi_norm_and_dustbin_denom() -> None:
    torch.manual_seed(0)
    phi = torch.randn(3, 9, 81) * 0.01
    pn = _phi_norm(phi)
    assert torch.allclose(pn.mean(dim=(1, 2)), torch.zeros(3), atol=1e-5)
    assert torch.allclose(pn.std(dim=(1, 2), unbiased=False), torch.ones(3), atol=1e-4)
    dust = torch.full((3, 9), -1.0)
    p_real, p_dust = _sinkhorn_dustbin(pn, dust, torch.tensor(0.25))
    a = _row_stoch_dustbin(p_real, p_dust)
    row = a.sum(dim=-1)
    assert (row <= 1.0 + 1e-4).all()
    frac = p_dust / (p_real.sum(-1) + p_dust + 1e-6)
    assert torch.allclose(a.sum(-1) + frac, torch.ones_like(frac), atol=1e-4)


def test_alpha_staggered_h_and_gamma_anneal() -> None:
    r = MultiBankRouter()
    assert float(r.h_S0) == 0.0
    assert abs(float(r.h_T1) - 0.95) < 1e-6
    assert abs(float(r.h_T2) - 1.65) < 1e-6
    assert abs(float(r.h_S1) - 1.35) < 1e-6
    assert abs(float(r.beta_S0_T1) - 2.0) < 1e-6
    a0 = alpha_from_depth(r.h_S0, r.h_T1, r.beta_S0_T1, torch.tensor(0.0))
    assert abs(float(a0) - (1.0 / (1.0 + math.exp(-2.0)))) < 1e-5
    assert gamma_anneal(0) == 0.0
    assert abs(gamma_anneal(500, gamma_max=1.0) - 0.5) < 1e-9
    assert gamma_anneal(1000, gamma_max=1.0) == 1.0


def test_active_mass_t2s_scale3_and_musigma() -> None:
    """Active-mass T→S: scale-3 when W≥3; packet is cat[μ,σ]; dustbin w→0 drops out."""
    r = MultiBankRouter(n_s0=81, n_s1=9)
    torch.manual_seed(0)
    # Synthetic hard P: each cell mass 1/9 on its house → N_per·Σ_h P = 1 per head.
    p = torch.zeros(3, 9, 81)
    for f in range(3):
        for n in range(81):
            p[f, n % 9, n] = 1.0 / 9.0
    msg = torch.ones(1, 27, 64)
    pkt = r.packet_t2s(p, msg, torch.tensor(1.0))
    assert pkt.shape == (1, 81, 128)
    mu, sigma = pkt[..., :64], pkt[..., 64:]
    # W=3, scale=1, Σ_f w_f m_f = 3 → μ≈3; identical heads ⇒ σ≈0.
    assert torch.allclose(mu, torch.full_like(mu, 3.0), atol=1e-4)
    # Identical heads → σ = sqrt(ε) floor.
    assert float(sigma.max()) <= 1e-3 + 1e-9
    # Dustbin drop: zero one family's P → that head's w=0, μ→2.
    p2 = p.clone()
    p2[0].zero_()
    pkt2 = r.packet_t2s(p2, msg, torch.tensor(1.0))
    assert torch.allclose(pkt2[..., :64], torch.full((1, 81, 64), 2.0), atol=1e-4)


def test_static_leapfrog_no_argsort() -> None:
    import inspect

    from riemannian_sudoku import manifold_bank as mb

    src = inspect.getsource(mb.MultiBankAttractorCore.integrate)
    assert 'argsort' not in src
    assert src.index('t1.write') < src.index('s0.write')
    assert src.index('t2.write') < src.index('s0.write')
    # Momentum is a local loop tensor, not an nn.Buffer.
    assert 'zeros_like' in src
    assert 'register_buffer' not in src


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
    assert 'pooled_top3_purity' in out and 'l_h_repel' in out and 'l_comp' in out
    out['loss'].backward()
    n_vf = model.vf_param_count()
    n_r = model.router_param_count()
    n_all = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_vf + n_r == n_all
    assert n_vf == _VF_PARAMS_V31, f'vf={n_vf}'
    assert n_r == _ROUTER_PARAMS, f'router={n_r}'
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


def test_h_repel_and_competition() -> None:
    h1 = torch.tensor(0.95)
    h2 = torch.tensor(1.65)
    # Far depths → small repel; identical → λ.
    assert float(h_repel_loss(h1, h2, lam=0.05)) < float(h_repel_loss(h1, h1, lam=0.05))
    assert abs(float(h_repel_loss(h1, h1, lam=0.05)) - 0.05) < 1e-6
    alphas = {
        'S0_T1': torch.tensor(0.8),
        'S0_T2': torch.tensor(0.8),
        'S1_T1': torch.tensor(0.5),
        'S1_T2': torch.tensor(0.5),
    }
    # γ=0 → L_comp=0; γ>0 → positive.
    assert float(competition_loss(alphas, torch.tensor(0.0))) == 0.0
    assert float(competition_loss(alphas, torch.tensor(1.0), lam=0.05)) > 0.0


def test_universal_clash_detached_board_mean() -> None:
    torch.manual_seed(0)
    x = F.normalize(torch.randn(2, 81, 64, requires_grad=True), dim=-1)
    x = x.detach().requires_grad_(True)
    a1 = torch.tensor(0.9, requires_grad=True)
    a2 = torch.tensor(0.1, requires_grad=True)
    loss = universal_clash_loss(x, a1, a2)
    loss.backward()
    # Detached x̃: clash must not train cell answers.
    assert x.grad is None or torch.allclose(x.grad, torch.zeros_like(x.grad))
    assert a1.grad is not None and float(a1.grad.abs()) > 0.0


def test_s2t_musigma_and_t2s_musigma() -> None:
    r = MultiBankRouter()
    a = r.get_incidence(torch.tensor(0.5))['S0_T1']
    msg_t = torch.randn(1, 27, 64)
    pkt = r.packet_t2s(a['P'], msg_t, torch.tensor(1.0))
    assert pkt.shape == (1, 81, 128)  # [μ,σ] not μ-only
    msg_s = torch.randn(1, 81, 64)
    pkt_s = r.packet_s2t(a['A_s2t'], msg_s, torch.tensor(1.0))
    assert pkt_s.shape == (1, 27, 128)
