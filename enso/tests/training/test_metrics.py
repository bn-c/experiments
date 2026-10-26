import torch

from ebm.training.losses import LossOutput
from ebm.training.metrics import (
    compute_cell_accuracy,
    compute_puzzle_accuracy,
    compute_z_variance,
    format_train_postfix,
)


def test_cell_accuracy_perfect():
    pred = torch.ones(2, 9, 9, dtype=torch.long)
    target = torch.zeros(2, 9, 9, 9)
    target[:, :, :, 0] = 1.0
    mask = torch.zeros(2, 9, 9)
    assert compute_cell_accuracy(pred, target, mask) == 1.0


def test_cell_accuracy_all_wrong():
    pred = torch.full((1, 9, 9), 9, dtype=torch.long)
    target = torch.zeros(1, 9, 9, 9)
    target[:, :, :, 0] = 1.0
    mask = torch.zeros(1, 9, 9)
    assert compute_cell_accuracy(pred, target, mask) == 0.0


def test_cell_accuracy_ignores_clues():
    pred = torch.full((1, 9, 9), 9, dtype=torch.long)
    target = torch.zeros(1, 9, 9, 9)
    target[:, :, :, 0] = 1.0
    mask = torch.ones(1, 9, 9)
    assert compute_cell_accuracy(pred, target, mask) == 1.0


def test_puzzle_accuracy_perfect():
    pred = torch.ones(4, 9, 9, dtype=torch.long)
    target = torch.zeros(4, 9, 9, 9)
    target[:, :, :, 0] = 1.0
    assert compute_puzzle_accuracy(pred, target) == 1.0


def test_puzzle_accuracy_partial():
    pred = torch.ones(2, 9, 9, dtype=torch.long)
    target = torch.zeros(2, 9, 9, 9)
    target[:, :, :, 0] = 1.0
    pred[1, 0, 0] = 5
    assert compute_puzzle_accuracy(pred, target) == 0.5


def test_z_variance_positive():
    z = torch.randn(32, 64)
    var = compute_z_variance(z)
    assert var > 0


def test_format_train_postfix():
    loss_out = LossOutput(
        total=torch.tensor(35.8327),
        energy=torch.tensor(1.2345),
        vicreg=torch.tensor(24.5),
        decode=torch.tensor(2.1),
        constraint=torch.tensor(0.0567),
    )
    assert format_train_postfix(loss_out) == {
        'loss': '35.8327',
        'energy': '1.2345',
        'vicreg': '24.5000',
        'decode': '2.1000',
        'constraint': '0.0567',
    }


def test_z_variance_collapsed():
    z = torch.ones(32, 64)
    var = compute_z_variance(z)
    assert var == 0.0
