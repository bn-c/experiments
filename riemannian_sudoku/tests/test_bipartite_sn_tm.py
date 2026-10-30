"""Bipartite S^{63}↔T^{16}: Householder ETF, Clifford, geodesic, params, CE."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from riemannian_sudoku.geodesic import geodesic_step
from riemannian_sudoku.manifold_embed import (
    BLANK_COORD,
    D,
    M,
    N_DIGITS,
    cell_house_idx,
    clifford_embed,
    house_cell_idx,
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
    # Cell 0 ∈ row0, col0, box0
    assert set(c[0].tolist()) == {0, 9, 18}
    assert 0 in h[0].tolist() and 0 in h[9].tolist() and 0 in h[18].tolist()


def test_householder_etf_orthogonal_to_blank() -> None:
    poles = simplex_etf_poles()
    assert poles.shape == (9, D)
    assert torch.allclose(poles.norm(dim=-1), torch.ones(9), atol=1e-5)
    # Householder: blank axis cleared; ⟨e_8, poles[d]⟩ = 0
    assert poles[:, BLANK_COORD].abs().max() < 1e-6
    e8 = north_pole()
    assert (poles @ e8).abs().max() < 1e-5
    # Pairwise digit dots −1/8
    grams = poles @ poles.T
    off = grams - torch.eye(N_DIGITS)
    assert torch.allclose(off, torch.full_like(off, -1.0 / 8.0) * (1.0 - torch.eye(N_DIGITS)), atol=1e-5)


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
    x, y, is_clue = init_bipartite_state(puzzle, poles, pe)
    assert x.shape == (1, 81, D) and y.shape == (1, 27, M)
    assert torch.allclose(x.norm(dim=-1), torch.ones(1, 81), atol=1e-4)
    assert bool(is_clue[0, 1]) and not bool(is_clue[0, 0])
    # Clue digit 5 → pole index 4
    assert torch.allclose(x[0, 1], poles[4], atol=1e-5)
    # Blank has mass on North Pole coord (PE is scratch-only).
    assert x[0, 0, BLANK_COORD] > 0.5


def test_blank_init_logits_near_uniform() -> None:
    """A2: no digit-9 spike from e_8 / pole collinearity."""
    model = BipartiteSnTmSolver(num_steps=1)
    puzzle = torch.zeros(1, 81, dtype=torch.long)
    # Init-only readout: integrate still runs 1 step; check raw init logits path.
    x, _, _ = init_bipartite_state(puzzle, model.poles, model.pe)
    logits = model._logits(x)
    probs = F.softmax(logits, dim=-1)
    # Blank cells: near-uniform (max class mass not digit-9 dominated).
    blank = probs[0]  # all blank
    assert blank.max(dim=-1).values.mean() < 0.25
    assert (blank.std(dim=-1) < 0.05).all()
    # Class 8 (digit 9) not a spike vs others
    assert (blank[:, 8] - blank.mean(dim=-1)).abs().mean() < 0.05


def test_geodesic_order_projects_before_cap() -> None:
    """Ambient tanh-before-project is forbidden; large ambient along x must vanish."""
    torch.manual_seed(0)
    x = F.normalize(torch.randn(2, 81, D), dim=-1)
    x0 = x.clone()
    is_clue = torch.zeros(2, 81, dtype=torch.bool)
    # Pure normal ambient → tangent 0 → state unchanged (not ambient-capped).
    u = 100.0 * x
    y = geodesic_step(x, u, x0, is_clue)
    assert torch.allclose(y, x, atol=1e-4)
    # Clue freeze
    is_clue[:, 0] = True
    u2 = torch.randn_like(x)
    y2 = geodesic_step(x, u2, x0, is_clue)
    assert torch.allclose(y2[:, 0], x0[:, 0], atol=1e-6)
    assert (y2[:, 1:].norm(dim=-1) - 1).abs().max() < 1e-4


def test_bipartite_forward_backward_and_diags() -> None:
    torch.manual_seed(2)
    model = BipartiteSnTmSolver(num_steps=2, omega_weight=0.0)
    puzzle = torch.randint(0, 10, (4, 81))
    solution = torch.randint(1, 10, (4, 81))
    mask = (puzzle == 0).float()
    out = model(puzzle, solution, mask)
    assert torch.isfinite(out['loss'])
    assert out['logits'].shape == (4, 81, 9)
    assert 'omega_sq_1' in out and 'omega_sq_k' in out
    assert 'scratch_energy' in out and 'blank_np' in out
    out['loss'].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    # Clue cells stay at init poles
    x, _, is_clue = init_bipartite_state(puzzle, model.poles, model.pe)
    assert torch.allclose(out['x'][is_clue], x[is_clue], atol=1e-4)


def test_constant_house_batch_no_nan() -> None:
    """Identical cells → finite std via eps; forward/backward finite."""
    torch.manual_seed(3)
    model = BipartiteSnTmSolver(num_steps=2)
    # All blanks → nearly constant houses after init
    puzzle = torch.zeros(2, 81, dtype=torch.long)
    solution = torch.randint(1, 10, (2, 81))
    out = model(puzzle, solution)
    assert torch.isfinite(out['loss'])
    out['loss'].backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_bipartite_param_count_target() -> None:
    model = BipartiteSnTmSolver()
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert abs(n - 681_441) <= 1, f'n_params={n} expected ≈681441'
    assert 600_000 <= n <= 700_000, f'n_params={n} outside 0.60–0.70M'
    assert model.logit_scale.item() == 4.0


def test_build_solver_bipartite_and_resolve() -> None:
    from riemannian_sudoku.train import _resolve_mixer

    m = build_solver('bipartite_sn_tm', num_steps=3)
    assert isinstance(m, BipartiteSnTmSolver)
    assert m.mixer == 'bipartite_sn_tm'
    assert _resolve_mixer(None, {'mixer': 'bipartite_sn_tm'}) == 'bipartite_sn_tm'
    assert _resolve_mixer('bipartite_sn_tm', {'mixer': 'geo_transformer'}) == 'bipartite_sn_tm'


def test_logit_scale_clamp() -> None:
    model = BipartiteSnTmSolver(num_steps=1)
    with torch.no_grad():
        model.logit_scale.fill_(100.0)
    puzzle = torch.zeros(1, 81, dtype=torch.long)
    logits = model(puzzle)['logits']
    # scale clamped to 50; cosine ≤1 → |logit| ≤ 50
    assert logits.abs().max() <= 50.0 + 1e-4
