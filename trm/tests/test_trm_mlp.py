"""Unit tests for TRM-MLP."""

from __future__ import annotations

from unittest.mock import patch

import torch

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


def test_one_training_step() -> None:
    from torch.utils.data import DataLoader, Dataset

    class _ListDS(Dataset):
        def __len__(self) -> int:
            return 4

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            batch = _fake_batch(1)
            return {k: v[0] for k, v in batch.items()}

    cfg = tiny_model_config()
    model = TRMMLP(cfg)
    train_cfg = tiny_train_config()
    loader = DataLoader(_ListDS(), batch_size=4)
    trainer = Trainer(model, loader, cfg, train_cfg, torch.device('cpu'))
    loss = trainer.train_epoch()
    assert loss > 0.0


def test_masked_loss_finite() -> None:
    batch = _fake_batch(2)
    cfg = ModelConfig(d_model=32, n_latent=1, T=1, max_supervision_steps=1)
    model = TRMMLP(cfg)
    _, logits = model.forward_supervised(batch['puzzle'])
    loss = masked_cross_entropy(logits[0], batch['solution'], batch['mask'])
    assert torch.isfinite(loss)


def test_checkpoint_saved_after_each_epoch(tmp_path) -> None:
    from torch.utils.data import DataLoader, Dataset

    class _ListDS(Dataset):
        def __len__(self) -> int:
            return 4

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            batch = _fake_batch(1)
            return {k: v[0] for k, v in batch.items()}

    cfg = tiny_model_config()
    model = TRMMLP(cfg)
    train_cfg = tiny_train_config().model_copy(update={'epochs': 2, 'checkpoint_dir': tmp_path})
    loader = DataLoader(_ListDS(), batch_size=4)
    trainer = Trainer(model, loader, cfg, train_cfg, torch.device('cpu'))

    with (
        patch('trm.training.trainer.init_wandb', return_value=False),
        patch('trm.training.trainer.upload_checkpoint_to_wandb'),
        patch('trm.training.trainer.finish_wandb'),
        patch.object(trainer, 'save_checkpoint', wraps=trainer.save_checkpoint) as save,
    ):
        trainer.train()
        assert save.call_count == 2
        assert (tmp_path / 'checkpoint.last.pt').is_file()
