"""Training loop with deep supervision and EMA."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from trm.config import AppConfig, ModelConfig, TrainConfig
from trm.model.ema import EMAModel
from trm.model.trm_mlp import TRMMLP
from trm.training.loss import masked_cross_entropy
from trm.training.metrics import (
    finish_wandb,
    init_wandb,
    log_epoch,
    log_train_step,
    upload_checkpoint_to_wandb,
)

logger = logging.getLogger(__name__)


class Trainer:
    """Single-GPU trainer for TRM deep supervision."""

    def __init__(
        self,
        model: TRMMLP,
        train_loader: DataLoader,
        model_cfg: ModelConfig,
        train_cfg: TrainConfig,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.device = device
        self.app_cfg = AppConfig()
        self.global_step = 0
        self.ema = EMAModel(self.model, decay=train_cfg.ema_decay)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
            betas=(0.9, 0.95),
        )

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in tqdm(self.train_loader, desc='train', leave=False):
            puzzle = batch['puzzle'].to(self.device)
            solution = batch['solution'].to(self.device)
            mask = batch['mask'].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            _, step_logits = self.model.forward_supervised(puzzle)
            loss = sum(masked_cross_entropy(logits, solution, mask) for logits in step_logits)
            loss.backward()
            self.optimizer.step()
            self.ema.update(self.model)

            batch_loss = float(loss.item())
            log_train_step(batch_loss, self.global_step)
            self.global_step += 1
            total_loss += batch_loss
            n_batches += 1
        return total_loss / max(n_batches, 1)

    def train(self) -> None:
        init_wandb(
            self.app_cfg,
            run_name='trm-sudoku',
            extra_config={**self.model_cfg.model_dump(), **self.train_cfg.model_dump()},
        )
        try:
            for epoch in range(self.train_cfg.epochs):
                avg_loss = self.train_epoch()
                log_epoch(epoch + 1, avg_loss, self.global_step)
                logger.info('Epoch %d/%d — loss %.4f', epoch + 1, self.train_cfg.epochs, avg_loss)
            checkpoint_path = self.train_cfg.checkpoint_dir / 'checkpoint.last.pt'
            self.save_checkpoint(checkpoint_path)
            upload_checkpoint_to_wandb(checkpoint_path)
        finally:
            finish_wandb()

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                'model_cfg': self.model_cfg.model_dump(),
                'model': self.model.state_dict(),
                'ema': self.ema.state_dict(),
            },
            path,
        )
        logger.info('Saved checkpoint to %s', path)
