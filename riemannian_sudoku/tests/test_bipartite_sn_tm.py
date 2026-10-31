"""Bipartite S^{63}↔T^{16}: Householder ETF, Clifford, geodesic, params, CE."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from riemannian_sudoku.geodesic import geodesic_step
from riemannian_sudoku.manifold_embed import (
    BLANK_COORD,
    D,
    M,
    N_DIGITS,
    SCRATCH_START,
    cell_house_idx,
    clifford_embed,
    house_cell_idx,
    init_bipartite_state,
    north_pole,
    simplex_etf_poles,
    torus_t3_pe,
)
from riemannian_sudoku.model import build_solver
from riemannian_sudoku.solver import LOGIT_SCALE_MAX, BipartiteSnTmSolver


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
    # Householder H: n_hat = 1/√9 ↦ +e_8 (not −e_8).
    n_hat = torch.ones(N_DIGITS) / math.sqrt(float(N_DIGITS))
    e_blank9 = torch.zeros(N_DIGITS)
    e_blank9[BLANK_COORD] = 1.0
    v = n_hat - e_blank9
    H = torch.eye(N_DIGITS) - 2.0 * torch.outer(v, v) / (v @ v)
    assert torch.allclose(H @ n_hat, e_blank9, atol=1e-6)
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
    # PE on all cells: clue :9 matches ETF after renorm; scratch 9:15 nonzero.
    clue_sem = F.normalize(x[0, 1, :N_DIGITS], dim=-1, eps=1e-6)
    assert torch.allclose(clue_sem, poles[4, :N_DIGITS], atol=1e-5)
    assert x[0, 1, SCRATCH_START : SCRATCH_START + 6].abs().max() > 0
    assert not torch.allclose(x[0, 1], poles[4], atol=1e-3)
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
    assert n == 687_010, f'n_params={n} expected 687010'
    assert 600_000 <= n <= 700_000, f'n_params={n} outside 0.60–0.70M'
    assert model.logit_scale.item() == 4.0


def test_kuramoto_fixed_point_zero_omega() -> None:
    """y = atan2(c_sin, c_cos) ⇒ Kuramoto ω_raw ≈ 0 ⇒ soft-capped ω ≈ 0."""
    from riemannian_sudoku.coupled_vf import CoupledVectorField

    torch.manual_seed(5)
    vf = CoupledVectorField()
    poles = simplex_etf_poles()
    h_idx = house_cell_idx()
    x = F.normalize(torch.randn(2, 81, D), dim=-1)
    c_cos = torch.randn(2, 27, M)
    c_sin = torch.randn(2, 27, M)
    y = torch.atan2(c_sin, c_cos)

    class _FixedTargets(torch.nn.Module):
        def forward(self, _inp: torch.Tensor) -> torch.Tensor:
            return torch.cat([c_cos, c_sin], dim=-1)

    vf.vf_y = _FixedTargets()
    omega = vf.omega(x, y, poles, h_idx)
    assert omega.abs().max() < 1e-6


def test_closed_gate_idle_at_pole() -> None:
    """gate≈0 ⇒ u[...,9:]=0; u_8=0 at ETF pole ⇒ v_tan≈0 and x_next≈x."""
    from riemannian_sudoku.coupled_vf import CoupledVectorField

    torch.manual_seed(6)
    vf = CoupledVectorField().double()
    poles = F.normalize(simplex_etf_poles().double(), dim=-1, eps=1e-12)
    c_idx = cell_house_idx()
    pe = torus_t3_pe().double()
    digit = 3
    x = poles[digit].view(1, 1, D).expand(1, 81, D).contiguous()
    y = torch.zeros(1, 27, M, dtype=torch.float64)
    # Head: radial u_raw (=x at pole), force u_8=0, gate logit −100 → gate≈0.
    # Ungated NP e_8 would otherwise create tangent if u_8≠0 while x_8≈0.
    raw = torch.zeros(1, 81, D + 1, dtype=torch.float64)
    raw[..., :D] = x
    raw[..., BLANK_COORD] = 0.0
    raw[..., D] = -100.0

    class _FixedHead(torch.nn.Module):
        def forward(self, _h: torch.Tensor) -> torch.Tensor:
            return raw

    vf.vf_x_body = _FixedHead()
    u, gate = vf.ambient_u(x, y, pe, c_idx, poles)
    assert gate.max() < 1e-6
    assert u[..., SCRATCH_START:].abs().max() < 1e-6
    assert u[..., BLANK_COORD].abs().max() < 1e-6
    v_tan = u - (x * u).sum(dim=-1, keepdim=True) * x
    assert v_tan.norm(dim=-1).max() < 1e-6
    is_clue = torch.zeros(1, 81, dtype=torch.bool)
    x_next = geodesic_step(x, u, x, is_clue)
    assert (x_next - x).abs().max() < 1e-6


def test_build_solver_bipartite_and_resolve() -> None:
    from riemannian_sudoku.train import _resolve_mixer

    m = build_solver('bipartite_sn_tm', num_steps=3)
    assert isinstance(m, BipartiteSnTmSolver)
    assert m.mixer == 'bipartite_sn_tm'
    assert _resolve_mixer(None, {'mixer': 'bipartite_sn_tm'}) == 'bipartite_sn_tm'
    assert _resolve_mixer('bipartite_sn_tm', {'mixer': 'geo_transformer'}) == 'bipartite_sn_tm'


def test_logit_scale_clamp() -> None:
    assert LOGIT_SCALE_MAX == 10.0
    model = BipartiteSnTmSolver(num_steps=1)
    with torch.no_grad():
        model.logit_scale.fill_(100.0)
    puzzle = torch.zeros(1, 81, dtype=torch.long)
    logits = model(puzzle)['logits']
    # scale clamped to 10; on S^{63}, ||x[...,:9]||≤1 and ||P[:,:9]||=1 → |logit| ≤ 10
    assert logits.abs().max() <= LOGIT_SCALE_MAX + 1e-4


def test_logits_unnormalized_matches_raw_dot() -> None:
    """A2 unnorm: _logits is scale · (x[...,:9] @ P[:,:9].T), no slice-norm."""
    model = BipartiteSnTmSolver(num_steps=1)
    # Pole + scratch mass: ||x[...,:9]|| ≪ 1 while ||x||=1 (NP is inside [:9]).
    raw_x = 0.3 * model.poles[0].clone()
    raw_x[SCRATCH_START] = 0.7
    x = F.normalize(raw_x, dim=-1).view(1, 1, -1).expand(1, 81, -1)
    assert float(x[..., :N_DIGITS].norm(dim=-1).mean()) < 0.5
    expected = model.logit_scale.clamp(max=LOGIT_SCALE_MAX) * (
        x[..., :N_DIGITS] @ model.poles[:, :N_DIGITS].T
    )
    got = model._logits(x)
    assert torch.allclose(got, expected, atol=1e-6)
    x_dec = F.normalize(x[..., :N_DIGITS], dim=-1, eps=1e-6)
    sliced = model.logit_scale.clamp(max=LOGIT_SCALE_MAX) * (
        x_dec @ model.poles[:, :N_DIGITS].T
    )
    assert not torch.allclose(got, sliced, atol=1e-3)


def test_house_features_relu_mean_scale() -> None:
    """Option A: keep relu(q); claim moments are mean(q) / mean(q²), not sum."""
    from riemannian_sudoku.coupled_vf import CoupledVectorField

    torch.manual_seed(4)
    vf = CoupledVectorField()
    poles = simplex_etf_poles()
    h_idx = house_cell_idx()
    x = F.normalize(torch.randn(2, 81, D), dim=-1)
    feat = vf.house_features(x, poles, h_idx)
    assert feat.shape == (2, 27, 2 * D + 2 * N_DIGITS)
    q = F.relu(x[..., :N_DIGITS] @ poles[:, :N_DIGITS].T)
    q_h = q[:, h_idx]
    expected = torch.cat(
        [
            x[:, h_idx].mean(dim=2),
            (x[:, h_idx].var(dim=2, unbiased=False) + 1e-6).sqrt(),
            q_h.mean(dim=2),
            (q_h * q_h).mean(dim=2),
        ],
        dim=-1,
    )
    assert torch.allclose(feat, expected, atol=1e-6)
    # Mean ≠ sum when house not identically zero
    assert not torch.allclose(q_h.mean(dim=2), q_h.sum(dim=2), atol=1e-3)
    # in_y stays 178 (zero-param scale)
    emb_m = 2 * M
    in_y = 2 * D + 2 * N_DIGITS + emb_m
    assert in_y == 178
    assert vf.vf_y[0].in_features == 178


def _parity_pack(seed: int = 7, batch: int = 4):
    """Shared gather vs hard-A inputs + model with identical weights."""
    from riemannian_sudoku.manifold_embed import hard_incidence_from_indices

    torch.manual_seed(seed)
    model = BipartiteSnTmSolver(num_steps=2)
    puzzle = torch.randint(0, 10, (batch, 81))
    x0, y0, is_clue = init_bipartite_state(puzzle, model.poles, model.pe)
    a_s2t, a_t2s = hard_incidence_from_indices(model.house_cell_idx, model.cell_house_idx)
    return model, puzzle, x0, y0, is_clue, a_s2t, a_t2s


def test_hard_incidence_shapes_and_row_sum() -> None:
    from riemannian_sudoku.manifold_embed import hard_incidence_from_indices

    a_s2t, a_t2s = hard_incidence_from_indices()
    assert a_s2t.shape == (27, 81) and a_t2s.shape == (81, 3, 27)
    assert torch.allclose(a_s2t.sum(dim=1), torch.ones(27), atol=1e-6)
    assert torch.allclose(a_t2s.sum(dim=-1), torch.ones(81, 3), atol=1e-6)
    # Cell 0 → houses 0, 9, 18
    assert a_t2s[0, :, [0, 9, 18]].sum() == 3.0


def test_hard_a_forward_parity() -> None:
    """house_features / omega / ambient_u / 1-step / K-step x_K gather vs hard-A."""
    atol, rtol = 1e-5, 1e-4
    model, _puzzle, x0, y0, is_clue, a_s2t, a_t2s = _parity_pack()
    vf = model.vf
    poles, pe = model.poles, model.pe
    h_idx, c_idx = model.house_cell_idx, model.cell_house_idx

    feat_g = vf.house_features(x0, poles, h_idx)
    feat_a = vf.house_features_hard(x0, poles, a_s2t)
    assert torch.allclose(feat_g, feat_a, atol=atol, rtol=rtol)

    omega_g = vf.omega(x0, y0, poles, h_idx)
    omega_a = vf.omega_hard(x0, y0, poles, a_s2t)
    assert torch.allclose(omega_g, omega_a, atol=atol, rtol=rtol)

    u_g, gate_g = vf.ambient_u(x0, y0, pe, c_idx, poles)
    u_a, gate_a = vf.ambient_u_hard(x0, y0, pe, a_t2s, poles)
    assert torch.allclose(u_g, u_a, atol=atol, rtol=rtol)
    assert torch.allclose(gate_g, gate_a, atol=atol, rtol=rtol)

    # 1-step x
    from riemannian_sudoku.geodesic import geodesic_step, torus_step

    y1_g = torus_step(y0, omega_g)
    y1_a = torus_step(y0, omega_a)
    u1_g, _ = vf.ambient_u(x0, y1_g, pe, c_idx, poles)
    u1_a, _ = vf.ambient_u_hard(x0, y1_a, pe, a_t2s, poles)
    x1_g = geodesic_step(x0, u1_g, x0, is_clue)
    x1_a = geodesic_step(x0, u1_a, x0, is_clue)
    assert torch.allclose(x1_g, x1_a, atol=atol, rtol=rtol)

    # K-step via core
    model.core.num_steps = 2
    xk_a, _, _, _, _ = model.core.integrate(x0, y0, is_clue, pe, a_s2t, a_t2s, poles)
    xk_g, _, _, _, _ = model.core.integrate_gather(x0, y0, is_clue, pe, h_idx, c_idx, poles)
    assert torch.allclose(xk_g, xk_a, atol=atol, rtol=rtol)


def test_hard_a_backward_parity() -> None:
    """1-step and K-step grads on vf_y / vf_x* match gather vs hard-A."""
    atol, rtol = 1e-5, 1e-4
    model, puzzle, x0, y0, is_clue, a_s2t, a_t2s = _parity_pack(seed=11)
    solution = torch.randint(1, 10, puzzle.shape)
    mask = (puzzle == 0).float()
    poles, pe = model.poles, model.pe
    h_idx, c_idx = model.house_cell_idx, model.cell_house_idx
    label_smoothing = max(0.0, 1.0 - model.ce_gamma)

    def _grads(hard: bool, steps: int):
        model.zero_grad(set_to_none=True)
        model.core.num_steps = steps
        if hard:
            x, omegas, _, _, _ = model.core.integrate(
                x0, y0, is_clue, pe, a_s2t, a_t2s, poles
            )
        else:
            x, omegas, _, _, _ = model.core.integrate_gather(
                x0, y0, is_clue, pe, h_idx, c_idx, poles
            )
        logits = model._logits(x)
        loss = model._blank_ce(logits, solution, mask, label_smoothing)
        # Touch ω so vf_y is in the graph even if CE depends only on x.
        loss = loss + 0.0 * omegas[-1].sum()
        loss.backward()
        grads = {
            n: p.grad.detach().clone()
            for n, p in model.named_parameters()
            if p.grad is not None and (n.startswith('vf.vf_y') or n.startswith('vf.vf_x') or n.startswith('vf.x_in'))
        }
        return grads

    for steps in (1, 2):
        g_g = _grads(hard=False, steps=steps)
        g_a = _grads(hard=True, steps=steps)
        assert g_g.keys() == g_a.keys()
        for name in g_g:
            assert torch.allclose(g_g[name], g_a[name], atol=atol, rtol=rtol), (
                f'steps={steps} {name} max_abs={(g_g[name] - g_a[name]).abs().max().item()}'
            )


def test_sudoku_adapter_pack() -> None:
    from riemannian_sudoku.manifold_embed import SudokuProblemAdapter

    poles = simplex_etf_poles()
    pe = torus_t3_pe()
    adapter = SudokuProblemAdapter(poles, pe)
    puzzle = torch.randint(0, 10, (2, 81))
    x0, is_clue, pe_o, a_s2t, a_t2s, poles_o, y0 = adapter.encode(puzzle)
    assert x0.shape == (2, 81, D) and y0.shape == (2, 27, M)
    assert is_clue.shape == (2, 81) and pe_o.shape == (81, 6)
    assert a_s2t.shape == (27, 81) and a_t2s.shape == (81, 3, 27)
    assert poles_o.shape == (9, D)

