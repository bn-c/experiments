"""Cross-coupled mixer: norms, orthant, clue freeze, tangent, gauge, backward."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from riemannian_sudoku.geometry import project_tangent
from riemannian_sudoku.model import (
    RiemannianSudokuSolver,
    init_sphere_state,
    init_t2_gauge,
)
from riemannian_sudoku.train import _resolve_mixer


def _cc_model(**kw) -> RiemannianSudokuSolver:
    defaults = dict(num_steps=3, dt=0.3, mixer='cross_coupled')
    defaults.update(kw)
    return RiemannianSudokuSolver(**defaults)


def test_init_t2_gauge_unit_and_angles() -> None:
    g = init_t2_gauge(2)
    assert g.shape == (2, 18, 2)
    assert torch.allclose(g.norm(dim=-1), torch.ones(2, 18), atol=1e-5)
    # Row 0 / col 0 at θ=0 → (1, 0); idx=3 → θ=2π/3.
    assert torch.allclose(g[0, 0], torch.tensor([1.0, 0.0]), atol=1e-5)
    assert torch.allclose(g[0, 9], torch.tensor([1.0, 0.0]), atol=1e-5)
    th = 2.0 * math.pi * 3 / 9
    assert torch.allclose(g[0, 3], torch.tensor([math.cos(th), math.sin(th)]), atol=1e-5)


def test_cross_coupled_sphere_and_gauge_norms() -> None:
    torch.manual_seed(0)
    model = _cc_model()
    puzzle = torch.randint(0, 10, (2, 81))
    puzzle[:, 0] = 4
    out = model(puzzle)
    s, g = out['s'], out['g']
    assert (s >= 0).all()
    assert torch.allclose(s.norm(dim=-1), torch.ones(2, 81), atol=1e-4)
    assert torch.allclose(g.norm(dim=-1), torch.ones(2, 18), atol=1e-4)


def test_cross_coupled_clue_freeze() -> None:
    torch.manual_seed(1)
    model = _cc_model(num_steps=4, dt=0.5)
    puzzle = torch.randint(0, 10, (2, 81))
    puzzle[:, 0] = 3
    puzzle[:, 1] = 0
    s, _, _, g = model._integrate_cross_coupled(puzzle)
    expected = torch.zeros(9)
    expected[2] = 1.0
    assert torch.allclose(s[:, 0], expected.expand_as(s[:, 0]), atol=1e-5)
    assert (s >= 0).all()
    assert torch.allclose(g.norm(dim=-1), torch.ones_like(g.norm(dim=-1)), atol=1e-4)


def test_cross_coupled_tangent_orthogonality() -> None:
    torch.manual_seed(2)
    model = _cc_model(num_steps=1)
    puzzle = torch.randint(0, 10, (3, 81))
    puzzle[:, 0] = 1  # clue
    s0, blank = init_sphere_state(puzzle)
    g0 = init_t2_gauge(3)
    v_s, v_g = model._cross_velocities(s0, g0)
    v_s = project_tangent(s0, v_s)
    v_g = project_tangent(g0, v_g)
    # Own-tangent: ⟨v, state⟩ ≈ 0 before clue mask.
    assert (v_s * s0).sum(dim=-1).abs().max() < 1e-5
    assert (v_g * g0).sum(dim=-1).abs().max() < 1e-5
    v_s_masked = v_s * blank.unsqueeze(-1)
    assert torch.allclose(v_s_masked[:, 0], torch.zeros(3, 9), atol=1e-6)


def test_cross_coupled_forward_backward() -> None:
    torch.manual_seed(0)
    model = _cc_model(num_steps=2, kinetic_weight=0.05, constraint_weight=0.0, supervise_all_steps=True)
    puzzle = torch.randint(0, 10, (4, 81))
    solution = torch.randint(1, 10, (4, 81))
    mask = (puzzle == 0).float()
    out = model(puzzle, solution, mask)
    assert torch.isfinite(out['loss'])
    assert 'g' in out
    assert 'gauge_kinetic_loss' in out
    out['loss'].backward()
    grads = [p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None]
    assert sum(grads) > 0.0
    # Gauge should have moved from init (untrained net, almost surely).
    g0 = init_t2_gauge(4)
    assert not torch.allclose(out['g'], g0, atol=1e-4)


def test_gauge_kinetic_weight_affects_loss() -> None:
    """λ_g=0 leaves loss = task + λ_k·kin; λ_g>0 adds gauge term (cells still separate)."""
    torch.manual_seed(3)
    puzzle = torch.randint(0, 10, (2, 81))
    solution = torch.randint(1, 10, (2, 81))
    mask = (puzzle == 0).float()
    off = _cc_model(num_steps=2, kinetic_weight=0.05, gauge_kinetic_weight=0.0, constraint_weight=0.0)
    on = _cc_model(num_steps=2, kinetic_weight=0.05, gauge_kinetic_weight=0.5, constraint_weight=0.0)
    on.load_state_dict(off.state_dict())
    with torch.no_grad():
        a = off(puzzle, solution, mask)
        b = on(puzzle, solution, mask)
    assert torch.allclose(a['kinetic_loss'], b['kinetic_loss'])
    assert a['gauge_kinetic_loss'].item() > 0.0
    assert torch.allclose(a['gauge_kinetic_loss'], b['gauge_kinetic_loss'])
    expected = a['loss'] + 0.5 * a['gauge_kinetic_loss']
    assert torch.allclose(b['loss'], expected, atol=1e-5)
    # Axial path: gauge kinetic is zero and λ_g does not change loss.
    ax = RiemannianSudokuSolver(
        d_model=32, n_heads=4, num_steps=2, kinetic_weight=0.05, gauge_kinetic_weight=0.5, constraint_weight=0.0
    )
    ax_out = ax(puzzle, solution, mask)
    assert ax_out['gauge_kinetic_loss'].item() == 0.0


def test_euclid_axial_still_default() -> None:
    model = RiemannianSudokuSolver(d_model=32, n_heads=4)
    assert model.mixer == 'euclid_axial'
    assert model.cross_mlp is None
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Tiny d_model; just ensure axial path builds.
    assert n > 0


def test_cross_coupled_param_count_and_mlp_shape() -> None:
    model = RiemannianSudokuSolver(mixer='cross_coupled')
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n == 1_047_293, f'unexpected param count {n}'
    assert model.cross_mlp is not None
    assert model.use_gauge is True
    linears = [m for m in model.cross_mlp if isinstance(m, nn.Linear)]
    shapes = [(m.in_features, m.out_features) for m in linears]
    assert shapes == [(765, 512), (512, 512), (512, 765)]


def test_no_gauge_cells_only_mlp_and_forward() -> None:
    """``use_gauge=False``: 729→MLP→729, no g / gauge_kinetic, still trains."""
    torch.manual_seed(4)
    model = _cc_model(use_gauge=False, kinetic_weight=0.5, gauge_kinetic_weight=0.5, constraint_weight=0.0)
    assert model.use_gauge is False
    assert model.cross_mlp is not None
    linears = [m for m in model.cross_mlp if isinstance(m, nn.Linear)]
    shapes = [(m.in_features, m.out_features) for m in linears]
    assert shapes == [(729, 512), (512, 512), (512, 729)]
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n == 1_010_393, f'unexpected no-gauge param count {n}'

    puzzle = torch.randint(0, 10, (2, 81))
    puzzle[:, 0] = 5
    solution = torch.randint(1, 10, (2, 81))
    mask = (puzzle == 0).float()
    out = model(puzzle, solution, mask)
    assert 'g' not in out
    assert out['gauge_kinetic_loss'].item() == 0.0
    # λ_g must not affect loss when there is no gauge channel.
    expected = out['task_loss'] + 0.5 * out['kinetic_loss']
    assert torch.allclose(out['loss'], expected, atol=1e-5)
    assert (out['s'] >= 0).all()
    assert torch.allclose(out['s'].norm(dim=-1), torch.ones(2, 81), atol=1e-4)
    out['loss'].backward()
    grads = [p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None]
    assert sum(grads) > 0.0


def test_resolve_mixer_ckpt_cli_roundtrip(tmp_path: Path) -> None:
    """Train/eval resume: ckpt mixer wins when CLI omits --mixer; CLI overrides."""
    from riemannian_sudoku.train import _resolve_use_gauge

    assert _resolve_mixer(None) == 'euclid_axial'
    assert _resolve_mixer(None, {}) == 'euclid_axial'
    assert _resolve_mixer(None, {'mixer': 'cross_coupled'}) == 'cross_coupled'
    assert _resolve_mixer('euclid_axial', {'mixer': 'cross_coupled'}) == 'euclid_axial'
    assert _resolve_mixer('cross_coupled', {'mixer': 'euclid_axial'}) == 'cross_coupled'
    assert _resolve_use_gauge(False) is True
    assert _resolve_use_gauge(True) is False
    assert _resolve_use_gauge(False, {'use_gauge': False}) is False
    assert _resolve_use_gauge(True, {'use_gauge': True}) is False  # CLI --no-gauge wins

    # Smoke: build from ckpt-resolved mixer and load_state_dict (the resume path).
    src = RiemannianSudokuSolver(mixer='cross_coupled', num_steps=2)
    ckpt_path = tmp_path / 'cc.pt'
    torch.save({'model': src.state_dict(), 'args': {'mixer': 'cross_coupled', 'use_gauge': True}}, ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    mixer = _resolve_mixer(None, ckpt.get('args') or {})
    use_gauge = _resolve_use_gauge(False, ckpt.get('args') or {})
    dst = RiemannianSudokuSolver(mixer=mixer, num_steps=2, use_gauge=use_gauge)
    dst.load_state_dict(ckpt['model'])  # must not raise Missing/Unexpected key
    assert dst.mixer == 'cross_coupled'
    assert dst.use_gauge is True


def test_unknown_mixer_raises() -> None:
    try:
        RiemannianSudokuSolver(mixer='nope')
    except ValueError as e:
        assert 'nope' in str(e)
    else:
        raise AssertionError('expected ValueError')


def test_cross_mlp_silu_same_params_as_gelu() -> None:
    """``--mlp-activation silu``: same Linear shapes / param count as default GELU."""
    gelu = _cc_model(use_gauge=False, mlp_activation='gelu')
    silu = _cc_model(use_gauge=False, mlp_activation='silu')
    assert gelu.mlp_activation == 'gelu'
    assert silu.mlp_activation == 'silu'
    n_g = sum(p.numel() for p in gelu.parameters() if p.requires_grad)
    n_s = sum(p.numel() for p in silu.parameters() if p.requires_grad)
    assert n_g == n_s == 1_010_393
    assert isinstance(silu.cross_mlp[1], nn.SiLU)
    assert isinstance(silu.cross_mlp[3], nn.SiLU)


def test_cross_mlp_swiglu_fair_params_and_forward() -> None:
    """SwiGLU mid-block ≈ GELU param count; trains and stays on the cell sphere."""
    from riemannian_sudoku.model import _SwiGLU

    torch.manual_seed(5)
    gelu = _cc_model(use_gauge=False, mlp_activation='gelu')
    swig = _cc_model(use_gauge=False, mlp_activation='swiglu', kinetic_weight=0.5, constraint_weight=0.0)
    n_g = sum(p.numel() for p in gelu.parameters() if p.requires_grad)
    n_w = sum(p.numel() for p in swig.parameters() if p.requires_grad)
    # Fair-ish: within 0.1% of GELU (h≈170 mid width).
    assert abs(n_w - n_g) / n_g < 0.001, f'swiglu={n_w} gelu={n_g}'
    assert isinstance(swig.cross_mlp[1], _SwiGLU)
    assert swig.mlp_layers == 1
    puzzle = torch.randint(0, 10, (2, 81))
    puzzle[:, 0] = 2
    solution = torch.randint(1, 10, (2, 81))
    mask = (puzzle == 0).float()
    out = swig(puzzle, solution, mask)
    assert torch.isfinite(out['loss'])
    assert 'g' not in out
    assert (out['s'] >= 0).all()
    out['loss'].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in swig.parameters())


def test_cross_mlp_swiglu_two_layers() -> None:
    """``mlp_layers=2``: Linear → SwiGLU → SwiGLU → Linear; deeper than Arm W."""
    from riemannian_sudoku.model import _SwiGLU

    torch.manual_seed(6)
    deep = _cc_model(
        use_gauge=False,
        mlp_activation='swiglu',
        mlp_layers=2,
        kinetic_weight=0.5,
        constraint_weight=0.0,
    )
    shallow = _cc_model(use_gauge=False, mlp_activation='swiglu', mlp_layers=1)
    assert deep.mlp_layers == 2
    assert isinstance(deep.cross_mlp[1], _SwiGLU)
    assert isinstance(deep.cross_mlp[2], _SwiGLU)
    assert isinstance(deep.cross_mlp[3], nn.Linear)
    n_d = sum(p.numel() for p in deep.parameters() if p.requires_grad)
    n_s = sum(p.numel() for p in shallow.parameters() if p.requires_grad)
    # Second SwiGLU adds ≈ one Linear(512,512) worth of params (261972).
    assert n_d > n_s
    assert n_d - n_s == 261_972, f'delta={n_d - n_s}'
    assert n_d == 1_271_681
    puzzle = torch.randint(0, 10, (2, 81))
    puzzle[:, 0] = 3
    solution = torch.randint(1, 10, (2, 81))
    mask = (puzzle == 0).float()
    out = deep(puzzle, solution, mask)
    assert torch.isfinite(out['loss'])
    assert 'g' not in out
    out['loss'].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in deep.parameters())


def test_cross_mlp_silu_expand_topology() -> None:
    """``--mlp-width expand --no-gauge``: cells-only 729→1458→SiLU→1458→SiLU→729 (~4.25M)."""
    torch.manual_seed(7)
    m = _cc_model(
        use_gauge=False,
        mlp_activation='silu',
        mlp_width='expand',
        kinetic_weight=0.5,
        constraint_weight=0.0,
    )
    assert m.mlp_width == 'expand'
    assert m.use_gauge is False
    mlp = m.cross_mlp
    assert isinstance(mlp[0], nn.Linear) and mlp[0].out_features == 1458
    assert isinstance(mlp[1], nn.SiLU)
    assert isinstance(mlp[2], nn.Linear) and mlp[2].in_features == mlp[2].out_features == 1458
    assert isinstance(mlp[3], nn.SiLU)
    assert isinstance(mlp[4], nn.Linear) and mlp[4].out_features == 729
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    # 729·1458+1458 + 1458²+1458 + 1458·729+729
    assert n == 4_255_173, f'params={n}'
    # Default GELU/SwiGLU paths unchanged.
    gelu = _cc_model(use_gauge=False, mlp_activation='gelu')
    assert gelu.mlp_width == 'default'
    assert sum(p.numel() for p in gelu.parameters() if p.requires_grad) == 1_010_393
    puzzle = torch.randint(0, 10, (2, 81))
    puzzle[:, 0] = 5
    solution = torch.randint(1, 10, (2, 81))
    mask = (puzzle == 0).float()
    out = m(puzzle, solution, mask)
    assert torch.isfinite(out['loss'])
    assert 'g' not in out
    out['loss'].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())


def test_cross_mlp_silu_expand_live_gauge() -> None:
    """Default expand SiLU: live T² — 765→1530→SiLU→1530→SiLU→765; MLP updates g."""
    torch.manual_seed(8)
    m = _cc_model(
        mlp_activation='silu',
        mlp_width='expand',
        kinetic_weight=0.5,
        constraint_weight=0.0,
    )
    assert m.use_gauge is True
    mlp = m.cross_mlp
    assert isinstance(mlp[0], nn.Linear) and mlp[0].in_features == 765 and mlp[0].out_features == 1530
    assert isinstance(mlp[1], nn.SiLU)
    assert isinstance(mlp[2], nn.Linear) and mlp[2].in_features == mlp[2].out_features == 1530
    assert isinstance(mlp[3], nn.SiLU)
    assert isinstance(mlp[4], nn.Linear) and mlp[4].out_features == 765
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    # 765·1530+1530 + 1530²+1530 + 1530·765+765
    assert n == 4_685_625, f'params={n}'

    puzzle = torch.randint(0, 10, (2, 81))
    puzzle[:, 0] = 5
    solution = torch.randint(1, 10, (2, 81))
    mask = (puzzle == 0).float()
    out = m(puzzle, solution, mask)
    assert 'g' in out
    assert out['g'].shape == (2, 18, 2)
    assert torch.allclose(out['g'].norm(dim=-1), torch.ones(2, 18), atol=1e-4)
    assert torch.isfinite(out['loss'])
    # MLP produced gauge tangent velocities each step (live ODE, not fixed PE / omitted).
    gv = m._last_gauge_velocities
    assert gv is not None and len(gv) == m.num_steps
    assert all(v.shape == (2, 18, 2) and v.norm() > 0 for v in gv)
    out['loss'].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())


def test_resolve_mlp_activation() -> None:
    from riemannian_sudoku.train import _resolve_mlp_activation

    assert _resolve_mlp_activation(None) == 'gelu'
    assert _resolve_mlp_activation(None, {}) == 'gelu'
    assert _resolve_mlp_activation(None, {'mlp_activation': 'silu'}) == 'silu'
    assert _resolve_mlp_activation('swiglu', {'mlp_activation': 'silu'}) == 'swiglu'


def test_resolve_mlp_layers() -> None:
    from riemannian_sudoku.train import _resolve_mlp_layers

    assert _resolve_mlp_layers(None) == 1
    assert _resolve_mlp_layers(None, {}) == 1
    assert _resolve_mlp_layers(None, {'mlp_layers': 2}) == 2
    assert _resolve_mlp_layers(3, {'mlp_layers': 2}) == 3
    assert _resolve_mlp_layers(0, {'mlp_layers': 2}) == 2  # 0 = unset → ckpt


def test_resolve_mlp_width() -> None:
    from riemannian_sudoku.train import _resolve_mlp_width

    assert _resolve_mlp_width(None) == 'default'
    assert _resolve_mlp_width(None, {}) == 'default'
    assert _resolve_mlp_width(None, {'mlp_width': 'expand'}) == 'expand'
    assert _resolve_mlp_width('expand', {'mlp_width': 'default'}) == 'expand'


def test_unknown_mlp_activation_raises() -> None:
    try:
        RiemannianSudokuSolver(mixer='cross_coupled', mlp_activation='relu')
    except ValueError as e:
        assert 'relu' in str(e)
    else:
        raise AssertionError('expected ValueError')


def test_mlp_layers_lt_one_raises() -> None:
    try:
        RiemannianSudokuSolver(mixer='cross_coupled', mlp_layers=0)
    except ValueError as e:
        assert 'mlp_layers' in str(e)
    else:
        raise AssertionError('expected ValueError')


def test_unknown_mlp_width_raises() -> None:
    try:
        RiemannianSudokuSolver(mixer='cross_coupled', mlp_width='wide')
    except ValueError as e:
        assert 'wide' in str(e)
    else:
        raise AssertionError('expected ValueError')
