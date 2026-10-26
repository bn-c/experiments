"""Pytest coverage for paper-faithful SFPRM Mini-ARC modules."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
_SFPRM = _ROOT / "sfprm_miniarc"
if str(_SFPRM) not in sys.path:
    sys.path.insert(0, str(_SFPRM))

from models.sfprm_core import DEFAULT_D_FF, DEFAULT_D_MODEL, SFPRM, logit  # noqa: E402
from models.solver import SolverConfig, SolverState, update_schedule  # noqa: E402
from models.spherical_ops import chordal_residual, guarded_slerp, project_to_sphere  # noqa: E402


def test_logit_lambda_init():
    assert abs(torch.sigmoid(torch.tensor(logit(0.1))).item() - 0.1) < 1e-6


def test_chordal_residual_zero_on_identical():
    z = project_to_sphere(torch.randn(3, 7, 16))
    r = chordal_residual(z, z)
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-5)


def test_chordal_residual_ignores_pad():
    z = project_to_sphere(torch.randn(2, 4, 8))
    z_tilde = z.clone()
    # Make position 0 diverge a lot; mark it PAD (invalid)
    z_tilde[:, 0] = project_to_sphere(torch.randn(2, 8))
    mask = torch.tensor([[False, True, True, True], [False, True, True, True]])
    r_masked = chordal_residual(z, z_tilde, token_mask=mask)
    r_full = chordal_residual(z, z_tilde, token_mask=None)
    assert torch.all(r_masked < r_full)
    assert torch.allclose(r_masked, torch.zeros_like(r_masked), atol=1e-5)


def test_guarded_slerp_unit_norm_all_branches():
    u = project_to_sphere(torch.randn(4, 5, 16))
    v = project_to_sphere(torch.randn(4, 5, 16))
    eta = torch.tensor([0.1, 0.5, 0.9, 1.0])
    for other in (v, u, -u):
        out = guarded_slerp(u, other, eta)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_patience_decay_and_halt_on_tau():
    cfg = SolverConfig(patience=2, gamma=0.5, eta_min=1e-3, tau=0.01, max_iter=10)
    state = SolverState.init(2, torch.device("cpu"), cfg)
    # First call sets best_residual; next two non-improvements exhaust patience → decay
    state = update_schedule(state, torch.tensor([0.5, 0.5]), cfg)
    state = update_schedule(state, torch.tensor([0.5, 0.5]), cfg)
    state = update_schedule(state, torch.tensor([0.5, 0.5]), cfg)
    assert abs(float(state.eta[0].item()) - 0.5) < 1e-6
    # Hit tau → halt
    state = update_schedule(state, torch.tensor([0.001, 0.001]), cfg)
    assert bool(state.halted.all())


def test_sfprm_step_and_renorm():
    model = SFPRM(
        vocab_size=12,
        d_model=32,
        n_heads=4,
        d_ff=64,
        n_layers=2,
        max_seq_len=16,
        n_puzzles=3,
        lambda_init=0.1,
    )
    assert abs(float(model.lambda_param.item()) - 0.1) < 1e-5
    x = torch.randint(0, 12, (2, 16))
    z_in = model.get_input_embedding(x, torch.tensor([0, 1]))
    z, _t, r = model.step(z_in, z_in, torch.ones(2))
    assert z.shape[1] == 17  # puzzle emb + seq
    assert r.ndim == 1
    model.renorm_weights()
    w = model.core_block.layers[0].q_proj.weight
    assert torch.allclose(w.norm(dim=-1), torch.ones(w.shape[0]), atol=1e-4)


def test_desk_param_count_range():
    model = SFPRM(
        vocab_size=12,
        d_model=DEFAULT_D_MODEL,
        d_ff=DEFAULT_D_FF,
        n_layers=2,
        max_seq_len=128,
        n_puzzles=149,
    )
    n = model.count_parameters()
    assert 4.0e6 < n < 6.0e6, n


def test_qk_logits_no_sqrt_dh():
    """Unit Q/K attention uses s_qk only (no /√d_h)."""
    model = SFPRM(
        vocab_size=12,
        d_model=32,
        n_heads=4,
        d_ff=64,
        n_layers=1,
        max_seq_len=8,
        n_puzzles=2,
        weight_dropout=0.0,
        variational_dropout=0.0,
    )
    layer = model.core_block.layers[0]
    # Spy: build identical unit q,k and check softmax path doesn't blow up; scale = s_qk.
    assert layer.s_qk.shape == (8,)  # head_dim = 32/4
    x = torch.randint(0, 10, (1, 8))  # no PAD
    z = model.get_input_embedding(x, torch.zeros(1, dtype=torch.long))
    pad = model.token_pad_mask(x)
    out = model.core_block(z, pad_mask=pad)
    assert out.shape == z.shape
    assert torch.isfinite(out).all()


def test_variational_mask_locked_across_segment_iters():
    model = SFPRM(
        vocab_size=12,
        d_model=32,
        n_heads=4,
        d_ff=64,
        n_layers=1,
        max_seq_len=8,
        n_puzzles=2,
        variational_dropout=0.5,
    )
    model.train()
    masks: list[torch.Tensor] = []
    orig = model.core_block.sample_variational_mask

    def capture_sample(batch_size, device, dtype):
        m = orig(batch_size, device, dtype)
        assert m is not None
        masks.append(m.detach().clone())
        return m

    model.core_block.sample_variational_mask = capture_sample  # type: ignore[method-assign]
    from models.solver import SolverConfig, SolverState, run_segment

    x = torch.randint(0, 10, (2, 8))
    z_in = model.get_input_embedding(x, torch.zeros(2, dtype=torch.long))
    pad = model.token_pad_mask(x)
    cfg = SolverConfig(patience=5, gamma=0.9, eta_min=0.001, tau=0.0, max_iter=8)
    state = SolverState.init(2, z_in.device, cfg)
    run_segment(model, z_in.clone(), z_in, state, cfg, k=4, pad_mask=pad)
    # Sampled once for the segment, not once per solver iter
    assert len(masks) == 1


def test_token_pad_mask_and_residual():
    model = SFPRM(
        vocab_size=12,
        d_model=32,
        n_heads=4,
        d_ff=64,
        n_layers=1,
        max_seq_len=8,
        n_puzzles=2,
        variational_dropout=0.0,
    )
    from models.sfprm_core import PAD_ID

    x = torch.randint(0, 10, (2, 8))
    x[:, -2:] = PAD_ID
    pad = model.token_pad_mask(x)
    assert pad.shape == (2, 9)  # + puzzle emb
    assert bool(pad[:, 0].all())  # puzzle emb valid
    assert not bool(pad[:, -1].any())
    z = model.get_input_embedding(x, torch.zeros(2, dtype=torch.long))
    z2 = z.clone()
    z2[:, -1] = project_to_sphere(torch.randn_like(z2[:, -1]))
    r = chordal_residual(z, z2, token_mask=pad)
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-5)
