"""Manifold-token Transformer mixer: 1× residual MHA + L=2 FFN head."""

from __future__ import annotations

import torch

from riemannian_sudoku.geometry import project_tangent
from riemannian_sudoku.model import (
    ManifoldTokenTransformer,
    RiemannianSudokuSolver,
    init_sphere_state,
)


def _geo_tf(**kw) -> RiemannianSudokuSolver:
    defaults = dict(
        num_steps=3,
        dt=0.5,
        mixer='geo_transformer',
        d_model=128,
        n_heads=4,
        kinetic_weight=0.5,
        constraint_weight=0.0,
    )
    defaults.update(kw)
    return RiemannianSudokuSolver(**defaults)


def test_geo_transformer_sphere_norms_and_clue_freeze() -> None:
    torch.manual_seed(0)
    model = _geo_tf(num_steps=4)
    puzzle = torch.randint(0, 10, (2, 81))
    puzzle[:, 0] = 7
    puzzle[:, 1] = 0
    out = model(puzzle)
    s = out['s']
    assert (s >= 0).all()
    assert torch.allclose(s.norm(dim=-1), torch.ones(2, 81), atol=1e-4)
    expected = torch.zeros(9)
    expected[6] = 1.0
    assert torch.allclose(s[:, 0], expected.expand_as(s[:, 0]), atol=1e-5)
    assert 'g' not in out
    assert model.use_gauge is False


def test_geo_transformer_tangent_orthogonality() -> None:
    torch.manual_seed(1)
    model = _geo_tf(num_steps=1)
    puzzle = torch.randint(0, 10, (3, 81))
    puzzle[:, 0] = 1  # clue
    s0, blank = init_sphere_state(puzzle)
    assert model.geo_tf is not None
    v = project_tangent(s0, model.geo_tf(s0))
    assert (v * s0).sum(dim=-1).abs().max() < 1e-5
    v_masked = v * blank.unsqueeze(-1)
    assert torch.allclose(v_masked[:, 0], torch.zeros(3, 9), atol=1e-6)


def test_geo_transformer_forward_backward() -> None:
    torch.manual_seed(2)
    model = _geo_tf(num_steps=2, supervise_all_steps=True)
    puzzle = torch.randint(0, 10, (4, 81))
    solution = torch.randint(1, 10, (4, 81))
    mask = (puzzle == 0).float()
    out = model(puzzle, solution, mask)
    assert torch.isfinite(out['loss'])
    assert out['gauge_kinetic_loss'].item() == 0.0
    out['loss'].backward()
    assert model.geo_tf is not None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.geo_tf.parameters())


def test_geo_transformer_param_count_o1_in_m() -> None:
    """Shared encode/MHA/head → params ≪ ambient cross MLP (~1M); no encoder FFN."""
    model = RiemannianSudokuSolver(mixer='geo_transformer', d_model=128, n_heads=4)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n < 1_000_000
    assert n > 50_000  # real MHA + L=2 head, not the 2-scalar geo arms
    assert model.geo_tf is not None
    assert model.cross_mlp is None
    assert model.use_gauge is False
    # Position = e_row + e_col only (no box / type / live T²).
    assert model.geo_tf.row_embed.num_embeddings == 9
    assert model.geo_tf.col_embed.num_embeddings == 9
    assert not hasattr(model.geo_tf, 'box_embed')
    assert not hasattr(model.geo_tf, 'type_embed')
    # Locked: plain MHA, not TransformerEncoder / EncoderLayer.
    assert isinstance(model.geo_tf.mha, torch.nn.MultiheadAttention)
    assert isinstance(model.geo_tf.norm, torch.nn.LayerNorm)
    assert not hasattr(model.geo_tf, 'transformer')
    assert not any(isinstance(m, torch.nn.TransformerEncoderLayer) for m in model.modules())
    assert not any(isinstance(m, torch.nn.TransformerEncoder) for m in model.modules())
    # Head is L=2 SiLU expand with d_ff=4d: 128→512→512→9
    head = model.geo_tf.head
    assert isinstance(head[0], torch.nn.Linear) and head[0].out_features == 512
    assert isinstance(head[2], torch.nn.Linear) and head[2].out_features == 512
    assert isinstance(head[4], torch.nn.Linear) and head[4].out_features == 9


def test_manifold_token_transformer_shapes() -> None:
    torch.manual_seed(3)
    tf = ManifoldTokenTransformer(d_model=64, n_heads=4)
    s = torch.randn(3, 81, 9)
    v = tf(s)
    assert v.shape == (3, 81, 9)
    assert tf.cell_rows.shape == (81,) and tf.cell_cols.shape == (81,)
    # Cell 10 = row1 col1 → embeds indexed correctly.
    assert int(tf.cell_rows[10]) == 1 and int(tf.cell_cols[10]) == 1
    # Zero TransformerEncoderLayer FFN params (linear1/linear2).
    assert sum(1 for n, _ in tf.named_modules() if 'linear1' in n or 'linear2' in n) == 0


def test_resolve_mixer_geo_transformer() -> None:
    from riemannian_sudoku.train import _resolve_mixer

    assert _resolve_mixer(None, {'mixer': 'geo_transformer'}) == 'geo_transformer'
    assert _resolve_mixer('geo_transformer', {'mixer': 'geo_log'}) == 'geo_transformer'
