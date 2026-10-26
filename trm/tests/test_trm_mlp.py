"""Unit tests for TRM-MLP."""

from __future__ import annotations

import logging
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, Dataset

from trm.config import ModelConfig, tiny_model_config, tiny_train_config
from trm.model.trm_mlp import TRMMLP
from trm.training.loss import masked_cross_entropy
from trm.training.trainer import Trainer


def _fake_batch(batch_size: int = 4) -> dict[str, torch.Tensor]:
    puzzle = torch.randint(0, 10, (batch_size, 81))
    solution = torch.randint(1, 10, (batch_size, 81))
    mask = (puzzle == 0).float()
    return {'puzzle': puzzle, 'solution': solution, 'mask': mask}


def test_forward_supervised_shapes() -> None:
    cfg = tiny_model_config()
    model = TRMMLP(cfg)
    batch = _fake_batch(2)
    _, step_logits = model.forward_supervised(batch['puzzle'])
    assert len(step_logits) == cfg.max_supervision_steps
    assert step_logits[0].shape == (2, 81, 9)


def test_supervision_step_returns_q_halt() -> None:
    cfg = tiny_model_config()
    model = TRMMLP(cfg)
    batch = _fake_batch(3)
    x = model.embed_puzzle(batch['puzzle'])
    y, z = model.initial_state(3, batch['puzzle'].device)
    y, z, logits, q_halt = model.supervision_step(x, y, z)
    assert logits.shape == (3, 81, 9)
    assert q_halt.shape == (3,)


class _ListDS(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        batch = _fake_batch(1)
        return {k: v[0] for k, v in batch.items()}


def test_one_training_step() -> None:
    cfg = tiny_model_config()
    model = TRMMLP(cfg)
    train_cfg = tiny_train_config()
    loader = DataLoader(_ListDS(), batch_size=4)
    trainer = Trainer(model, loader, cfg, train_cfg, torch.device('cpu'))
    batch = _fake_batch(4)
    metrics = trainer._train_batch(batch['puzzle'], batch['solution'], batch['mask'])
    assert metrics.lm_loss >= 0.0
    assert metrics.q_halt_loss >= 0.0
    assert 0.0 <= metrics.exact_accuracy <= 1.0
    assert 0.0 <= metrics.q_halt_accuracy <= 1.0
    assert 1.0 <= metrics.steps <= cfg.max_supervision_steps
    loss = trainer.train_epoch()
    assert loss >= 0.0


def test_evaluate_loader_returns_accuracies_in_range() -> None:
    cfg = tiny_model_config()
    model = TRMMLP(cfg)
    train_cfg = tiny_train_config()
    loader = DataLoader(_ListDS(), batch_size=4)
    trainer = Trainer(model, loader, cfg, train_cfg, torch.device('cpu'), validate=False)
    trainer.train_epoch()
    cell_acc, puzzle_acc = trainer._evaluate_loader(loader, desc='test')
    assert 0.0 <= cell_acc <= 1.0
    assert 0.0 <= puzzle_acc <= 1.0


def test_masked_loss_finite() -> None:
    batch = _fake_batch(2)
    cfg = ModelConfig(d_model=32, n_latent=1, T=1, max_supervision_steps=1)
    model = TRMMLP(cfg)
    _, logits = model.forward_supervised(batch['puzzle'])
    loss = masked_cross_entropy(logits[0], batch['solution'], batch['mask'])
    assert torch.isfinite(loss)


def test_train_saves_checkpoint_each_epoch(tmp_path, caplog) -> None:
    caplog.set_level(logging.INFO, logger='trm.training.trainer')
    cfg = tiny_model_config()
    model = TRMMLP(cfg)
    train_cfg = tiny_train_config().model_copy(update={'epochs': 2, 'checkpoint_dir': tmp_path})
    loader = DataLoader(_ListDS(), batch_size=4)
    trainer = Trainer(model, loader, cfg, train_cfg, torch.device('cpu'), validate=False)

    with (
        patch('trm.training.trainer.init_wandb', return_value=False),
        patch('trm.training.trainer.finish_wandb'),
    ):
        trainer.train()

    saves = [r.message for r in caplog.records if r.message.startswith('Saved checkpoint')]
    assert len(saves) == 2
    assert (tmp_path / 'checkpoint.last.pt').is_file()
