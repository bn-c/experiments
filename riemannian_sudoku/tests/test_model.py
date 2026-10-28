"""Model smoke: param count, sphere invariants, clue freeze, backward."""

from __future__ import annotations

import torch

from riemannian_sudoku.model import RiemannianSudokuSolver, init_sphere_state


def test_param_count_near_285k() -> None:
    model = RiemannianSudokuSolver(d_model=128, n_heads=4)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Plan target ~284_809; allow tiny drift from MHA impl details.
    assert 280_000 <= n <= 290_000, f'unexpected param count {n}'


def test_init_sphere_centroid_and_clues() -> None:
    puzzle = torch.zeros(1, 81, dtype=torch.long)
    puzzle[0, 0] = 5
    puzzle[0, 10] = 9
    s, mask = init_sphere_state(puzzle)
    assert torch.allclose(s[0, 1], torch.full((9,), 1.0 / 3.0), atol=1e-6)
    assert torch.allclose(s[0, 0], torch.tensor([0, 0, 0, 0, 1.0, 0, 0, 0, 0]))
    assert mask[0, 0] == 0 and mask[0, 1] == 1
    assert torch.allclose(s.norm(dim=-1), torch.ones(1, 81), atol=1e-5)


def test_integrate_clues_frozen_and_orthant() -> None:
    torch.manual_seed(1)
    model = RiemannianSudokuSolver(d_model=32, n_heads=4, num_steps=3, dt=0.3)
    puzzle = torch.randint(0, 10, (2, 81))
    # Ensure some clues.
    puzzle[:, 0] = 3
    puzzle[:, 1] = 0
    s, _, _ = model.integrate(puzzle)
    assert (s >= 0).all()
    assert torch.allclose(s.norm(dim=-1), torch.ones(2, 81), atol=1e-4)
    # Clue cell 0 stays at e_2
    expected = torch.zeros(9)
    expected[2] = 1.0
    assert torch.allclose(s[:, 0], expected.expand_as(s[:, 0]), atol=1e-5)


def test_loss_backward() -> None:
    torch.manual_seed(0)
    model = RiemannianSudokuSolver(d_model=32, n_heads=4, num_steps=2)
    puzzle = torch.randint(0, 10, (4, 81))
    solution = torch.randint(1, 10, (4, 81))
    mask = (puzzle == 0).float()
    out = model(puzzle, solution, mask)
    assert torch.isfinite(out['loss'])
    out['loss'].backward()
    grads = [p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None]
    assert sum(grads) > 0.0


def test_supervise_all_steps_averages_blank_nll() -> None:
    torch.manual_seed(0)
    puzzle = torch.randint(0, 10, (2, 81))
    solution = torch.randint(1, 10, (2, 81))
    mask = (puzzle == 0).float()
    final = RiemannianSudokuSolver(d_model=32, n_heads=4, num_steps=3, kinetic_weight=0.0, constraint_weight=0.0)
    multi = RiemannianSudokuSolver(
        d_model=32,
        n_heads=4,
        num_steps=3,
        kinetic_weight=0.0,
        constraint_weight=0.0,
        supervise_all_steps=True,
    )
    multi.load_state_dict(final.state_dict())
    out_f = final(puzzle, solution, mask)
    out_m = multi(puzzle, solution, mask)
    assert torch.isfinite(out_m['loss'])
    # Multi-step mean ≠ final-only for a random untrained net (almost surely).
    assert not torch.allclose(out_f['task_loss'], out_m['task_loss'], atol=1e-6)
    out_m['loss'].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in multi.parameters())


def test_buffers_follow_device() -> None:
    model = RiemannianSudokuSolver(d_model=32, n_heads=4)
    if not torch.cuda.is_available():
        assert model.t3_pe.device.type == 'cpu'
        return
    model = model.cuda()
    assert model.t3_pe.device.type == 'cuda'
    assert model.group_indices.device.type == 'cuda'
