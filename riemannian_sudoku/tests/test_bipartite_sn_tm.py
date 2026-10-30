"""Bipartite S^8×S^{54}↔T^{16}: Householder ETF, Clifford, geodesic, params, CE."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from riemannian_sudoku.geodesic import geodesic_step
from riemannian_sudoku.manifold_embed import (
    BLANK_COORD,
    COORD_DIM,
    COORD_REST,
    DECISION_DIM,
    M,
    N_DIGITS,
    cell_house_idx,
    clifford_embed,
    house_cell_idx,
    house_pair_triu_idx,
    init_bipartite_state,
    north_pole,
    simplex_etf_poles,
    torus_t3_pe,
)
from riemannian_sudoku.model import build_solver
from riemannian_sudoku.solver import BipartiteSnTmSolver


def test_static_index_buffers() -> None:
    h = house_cell_idx()
    c = cell_house_idx()
    assert h.shape == (27, 9) and c.shape == (81, 3)
    assert set(c[0].tolist()) == {0, 9, 18}
    assert 0 in h[0].tolist() and 0 in h[9].tolist() and 0 in h[18].tolist()
    ti, tj = house_pair_triu_idx()
    assert ti.shape == (36,) and tj.shape == (36,)


def test_householder_etf_orthogonal_to_blank() -> None:
    poles = simplex_etf_poles()
    assert poles.shape == (9, DECISION_DIM)
    assert torch.allclose(poles.norm(dim=-1), torch.ones(9), atol=1e-5)
    assert poles[:, BLANK_COORD].abs().max() < 1e-6
    e8 = north_pole()
    assert (poles @ e8).abs().max() < 1e-5
    grams = poles @ poles.T
    off = grams - torch.eye(N_DIGITS)
    assert torch.allclose(
        off, torch.full_like(off, -1.0 / 8.0) * (1.0 - torch.eye(N_DIGITS)), atol=1e-5
    )


def test_clifford_embed_unit_norm() -> None:
    torch.manual_seed(0)
    y = torch.randn(4, 27, M)
    emb = clifford_embed(y)
    assert emb.shape == (4, 27, 2 * M)
    assert torch.allclose(emb.norm(dim=-1), torch.ones(4, 27), atol=1e-5)


def test_simplex_etf_and_north_pole_init() -> None:
    poles = simplex_etf_poles()
    pe = torus_t3_pe()
    puzzle = torch.tensor([[0, 5] + [0] * 79])
    x, c, y, z, is_clue = init_bipartite_state(puzzle, poles, pe)
    assert z is None
    assert x.shape == (1, 81, DECISION_DIM) and c.shape == (1, 81, COORD_DIM)
    assert y.shape == (1, 27, M)
    assert torch.allclose(x.norm(dim=-1), torch.ones(1, 81), atol=1e-4)
    assert torch.allclose(c.norm(dim=-1), torch.ones(1, 81), atol=1e-4)
    assert bool(is_clue[0, 1]) and not bool(is_clue[0, 0])
    assert torch.allclose(x[0, 1], poles[4], atol=1e-5)
    assert x[0, 0, BLANK_COORD] > 0.5
    # Rest-axis mass on coordination sphere after PE renorm.
    assert c[0, 0, COORD_REST] > 0.5


def test_blank_init_logits_near_uniform() -> None:
    """A2: no digit-9 spike from e_8 / pole collinearity."""
    model = BipartiteSnTmSolver(num_steps=1)
    puzzle = torch.zeros(1, 81, dtype=torch.long)
    x, _, _, _, _ = init_bipartite_state(puzzle, model.poles, model.pe)
    logits = model._logits(x)
    probs = F.softmax(logits, dim=-1)
    blank = probs[0]
    assert blank.max(dim=-1).values.mean() < 0.25
    assert (blank.std(dim=-1) < 0.05).all()
    assert (blank[:, 8] - blank.mean(dim=-1)).abs().mean() < 0.05


def test_geodesic_order_projects_before_cap() -> None:
    """Ambient tanh-before-project is forbidden; large ambient along x must vanish."""
    torch.manual_seed(0)
    x = F.normalize(torch.randn(2, 81, DECISION_DIM), dim=-1)
    x0 = x.clone()
    is_clue = torch.zeros(2, 81, dtype=torch.bool)
    u = 100.0 * x
    y = geodesic_step(x, u, x0, is_clue)
    assert torch.allclose(y, x, atol=1e-4)
    is_clue[:, 0] = True
    u2 = torch.randn_like(x)
    y2 = geodesic_step(x, u2, x0, is_clue)
    assert torch.allclose(y2[:, 0], x0[:, 0], atol=1e-6)
    assert (y2[:, 1:].norm(dim=-1) - 1).abs().max() < 1e-4
    # Optional freeze: coordination sphere free path
    c = F.normalize(torch.randn(2, 81, COORD_DIM), dim=-1)
    c2 = geodesic_step(c, torch.randn_like(c))
    assert torch.allclose(c2.norm(dim=-1), torch.ones(2, 81), atol=1e-4)


def test_bipartite_forward_backward_and_diags() -> None:
    """Phase-1 ablation path (no z) still trains; clue freeze on x and c."""
    torch.manual_seed(2)
    model = BipartiteSnTmSolver(num_steps=2, omega_weight=0.0, use_global_torus=False)
    puzzle = torch.randint(0, 10, (4, 81))
    solution = torch.randint(1, 10, (4, 81))
    mask = (puzzle == 0).float()
    out = model(puzzle, solution, mask)
    assert torch.isfinite(out['loss'])
    assert out['logits'].shape == (4, 81, 9)
    assert out['x'].shape == (4, 81, DECISION_DIM)
    assert out['c'].shape == (4, 81, COORD_DIM)
    assert out['s'].shape == (4, 81, DECISION_DIM + COORD_DIM)
    assert 'omega_sq_1' in out and 'omega_sq_k' in out
    assert 'scratch_energy' in out and 'blank_np' in out
    assert 'omega_z_sq_1' not in out
    # Rest-axis displacement, not constant 1/55.
    assert abs(float(out['scratch_energy']) - 1.0 / COORD_DIM) > 1e-3
    out['loss'].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    # Clue cells stay at init on both S^8 and S^{54}.
    x0, c0, _, _, is_clue = init_bipartite_state(puzzle, model.poles, model.pe)
    assert torch.allclose(out['x'][is_clue], x0[is_clue], atol=1e-4)
    assert torch.allclose(out['c'][is_clue], c0[is_clue], atol=1e-4)
    assert torch.allclose(out['x'].norm(dim=-1), torch.ones(4, 81), atol=1e-4)
    assert torch.allclose(out['c'].norm(dim=-1), torch.ones(4, 81), atol=1e-4)
    assert model.vf.w_x_chart.weight.grad is not None
    assert model.vf.w_x_chart.weight.grad.abs().sum() > 0


def test_global_torus_forward_and_omega_z_diags() -> None:
    """Tip default: Phase-2 global T^{16} emits omega_z diags."""
    torch.manual_seed(4)
    model = BipartiteSnTmSolver(num_steps=2)  # use_global_torus=True tip default
    puzzle = torch.randint(0, 10, (2, 81))
    solution = torch.randint(1, 10, (2, 81))
    out = model(puzzle, solution)
    assert torch.isfinite(out['loss'])
    assert 'omega_z_sq_1' in out and 'omega_z_sq_k' in out
    assert torch.isfinite(out['omega_z_sq_1']) and torch.isfinite(out['omega_z_sq_k'])
    out['loss'].backward()


def test_constant_house_batch_no_nan() -> None:
    """Identical cells → finite std via eps; forward/backward finite."""
    torch.manual_seed(3)
    model = BipartiteSnTmSolver(num_steps=2)
    puzzle = torch.zeros(2, 81, dtype=torch.long)
    solution = torch.randint(1, 10, (2, 81))
    out = model(puzzle, solution)
    assert torch.isfinite(out['loss'])
    out['loss'].backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_bipartite_param_count_target() -> None:
    """Tip default = Phase-2 + chart (~681.9k)."""
    model = BipartiteSnTmSolver()
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert abs(n - 681_877) <= 2_000, f'n_params={n} expected ≈681877'
    assert 600_000 <= n <= 700_000, f'n_params={n} outside 0.60–0.70M'
    assert model.use_global_torus and model.vf.vf_z is not None
    assert model.logit_scale.item() == 4.0
    assert model.vf.w_x_chart.weight.shape == (18, 9)
    assert sum(p.numel() for p in model.vf.w_x_chart.parameters()) == 180
    # Poles never enter VF.
    assert not any('pole' in n.lower() for n, _ in model.vf.named_parameters())


def test_bipartite_param_count_phase1_ablation() -> None:
    model = BipartiteSnTmSolver(use_global_torus=False)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Phase-1 product (no z): H_Y=320, H_X=428 + w_x_chart(180) → ~680.6k
    assert abs(n - 680_565) <= 2_000, f'n_params={n} expected ≈680565'
    assert model.vf.vf_z is None
    assert model.vf.w_x_chart.weight.shape == (18, 9)


def test_build_solver_bipartite_and_resolve() -> None:
    from riemannian_sudoku.train import _resolve_mixer, _resolve_use_global_torus

    m = build_solver('bipartite_sn_tm', num_steps=3)
    assert isinstance(m, BipartiteSnTmSolver)
    assert m.mixer == 'bipartite_sn_tm'
    assert m.use_global_torus
    m1 = build_solver('bipartite_sn_tm', num_steps=3, use_global_torus=False)
    assert not m1.use_global_torus
    assert _resolve_mixer(None) == 'bipartite_sn_tm'
    assert _resolve_mixer(None, {'mixer': 'bipartite_sn_tm'}) == 'bipartite_sn_tm'
    assert _resolve_mixer('bipartite_sn_tm', {'mixer': 'geo_transformer'}) == 'bipartite_sn_tm'
    assert _resolve_use_global_torus(None) is True
    assert _resolve_use_global_torus(None, {'use_global_torus': False}) is False
    assert _resolve_use_global_torus(True, {'use_global_torus': False}) is True
    assert _resolve_use_global_torus(False, {'use_global_torus': True}) is False


def test_logit_scale_clamp() -> None:
    model = BipartiteSnTmSolver(num_steps=1)
    with torch.no_grad():
        model.logit_scale.fill_(100.0)
    puzzle = torch.zeros(1, 81, dtype=torch.long)
    logits = model(puzzle)['logits']
    assert logits.abs().max() <= 50.0 + 1e-4


def test_product_unit_norm_invariant_after_k() -> None:
    torch.manual_seed(5)
    model = BipartiteSnTmSolver(num_steps=8)
    puzzle = torch.randint(0, 10, (2, 81))
    out = model(puzzle)
    assert torch.allclose(out['x'].norm(dim=-1), torch.ones(2, 81), atol=1e-4)
    assert torch.allclose(out['c'].norm(dim=-1), torch.ones(2, 81), atol=1e-4)


def test_terminal_window_ce_ignores_early_steps() -> None:
    """CE window last-N only: early states do not change task_loss vs final-only when N=1."""
    torch.manual_seed(7)
    puzzle = torch.randint(0, 10, (2, 81))
    solution = torch.randint(1, 10, (2, 81))
    mask = (puzzle == 0).float()
    final = BipartiteSnTmSolver(num_steps=4, omega_weight=0.0, supervise_terminal_window=0)
    win1 = BipartiteSnTmSolver(num_steps=4, omega_weight=0.0, supervise_terminal_window=1)
    with torch.no_grad():
        win1.load_state_dict(final.state_dict())
    a = final(puzzle, solution, mask)
    b = win1(puzzle, solution, mask)
    assert torch.allclose(a['task_loss'], b['task_loss'], atol=1e-5)
    win4 = BipartiteSnTmSolver(num_steps=4, omega_weight=0.0, supervise_terminal_window=4)
    with torch.no_grad():
        win4.load_state_dict(final.state_dict())
    c = win4(puzzle, solution, mask)
    # Full-window mean CE generally differs from final-only (unless flat trajectory).
    assert 'brake_loss' in c and torch.isfinite(c['loss'])
    c['loss'].backward()


def test_velocity_brake_adds_lambda_times_four_energies() -> None:
    """λ_brake multiplies ‖ω_y‖²+‖ω_z‖²+‖v_x‖²+‖v_c‖² into loss."""
    torch.manual_seed(8)
    puzzle = torch.randint(0, 10, (2, 81))
    solution = torch.randint(1, 10, (2, 81))
    off = BipartiteSnTmSolver(num_steps=2, omega_weight=0.0)
    on = BipartiteSnTmSolver(num_steps=2, omega_weight=0.01)
    with torch.no_grad():
        on.load_state_dict(off.state_dict())
    a = off(puzzle, solution)
    b = on(puzzle, solution)
    assert torch.allclose(a['task_loss'], b['task_loss'], atol=1e-6)
    expected = a['task_loss'] + 0.01 * a['brake_loss']
    assert torch.allclose(b['loss'], expected, atol=1e-5)
    # Brake = sum of four mean-squares (ω_z present under tip default).
    parts = a['omega_sq_k'] + a['omega_z_sq_k'] + a['v_x_sq_k'] + a['v_c_sq_k']
    assert torch.allclose(a['brake_loss'], parts, atol=1e-6)
    assert float(a['brake_loss']) > 0.0
    b['loss'].backward()


def test_build_solver_terminal_window_and_brake() -> None:
    m = build_solver(
        'bipartite_sn_tm',
        num_steps=16,
        gauge_kinetic_weight=0.01,
        supervise_terminal_window=4,
    )
    assert isinstance(m, BipartiteSnTmSolver)
    assert m.supervise_terminal_window == 4
    assert m.omega_weight == 0.01
    assert not m.supervise_all_steps
