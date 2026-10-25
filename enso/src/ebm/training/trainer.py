"""Training loop for Sudoku JEPA."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from ebm.model.jepa import SudokuJEPA
from ebm.training.checkpoint import LAST_CHECKPOINT_NAME, CheckpointManager, _CheckpointData
from ebm.training.losses import compute_loss
from ebm.training.metrics import (
    compute_cell_accuracy,
    compute_puzzle_accuracy,
    compute_z_variance,
    finish_wandb,
    init_wandb,
    log_train_step,
    log_validation,
    upload_checkpoint_to_wandb,
)
from ebm.training.scheduler import create_lr_scheduler, get_ema_momentum
from ebm.utils.config import ArchitectureConfig, Config, TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Grouped configuration for the Trainer."""

    app: Config
    arch: ArchitectureConfig
    train: TrainingConfig


class Trainer:
    """
    Trains a SudokuJEPA model with AdamW, LR scheduling, EMA, and checkpointing.

    Handles the full training loop including validation, metric logging,
    and checkpoint management.
    """

    def __init__(
        self,
        model: SudokuJEPA,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: TrainerConfig,
        test_loader: DataLoader | None = None,
        resume_from: Path | None = None,
    ) -> None:
        """
        Initialize the trainer.

        Args:
            model: SudokuJEPA model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            cfg: Grouped configuration (app, arch, train).
            test_loader: Optional held-out test loader (evaluated once after training).
            resume_from: Path to ``checkpoint.last.pt`` (or other full training checkpoint).

        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_cfg = cfg.train

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )

        self.total_steps = len(train_loader) * cfg.train.epochs
        self.scheduler = create_lr_scheduler(self.optimizer, cfg.train, self.total_steps)
        self.checkpoint_mgr = CheckpointManager(cfg.train.checkpoint_dir, cfg.train.keep_top_k)

        self.start_epoch = 0
        self.global_step = 0
        self._last_val_energy = 0.0
        self._last_cell_acc = 0.0
        if resume_from is not None:
            self._load_resume(resume_from)

        init_wandb(
            cfg.app,
            run_name='sudoku-jepa',
            extra_config={**cfg.arch.model_dump(), **cfg.train.model_dump()},
        )

    def _load_resume(self, path: Path) -> None:
        if not path.exists():
            logger.warning('Resume checkpoint not found at %s; starting from scratch', path)
            return
        meta = CheckpointManager.load(path, self.model, self.optimizer, self.scheduler)
        self.start_epoch = int(meta['epoch']) + 1
        self.global_step = int(meta.get('step', 0))
        self._last_val_energy = float(meta.get('val_energy', 0.0))
        self._last_cell_acc = float(meta.get('cell_accuracy', 0.0))
        self.checkpoint_mgr.reindex_from_disk()
        logger.info(
            'Resumed from %s at epoch %d (global_step=%d, last cell_acc=%.4f)',
            path.name,
            self.start_epoch,
            self.global_step,
            float(meta.get('cell_accuracy', 0.0)),
        )
        if self.start_epoch >= self.train_cfg.epochs:
            logger.warning(
                'Checkpoint epoch %d >= configured epochs %d; nothing to train',
                self.start_epoch,
                self.train_cfg.epochs,
            )

    def train(self) -> None:
        """Run the full training loop for all epochs."""
        if self.start_epoch >= self.train_cfg.epochs:
            return

        self._last_ckpt_time = time.monotonic()

        with logging_redirect_tqdm():
            epoch_bar = tqdm(
                range(self.start_epoch, self.train_cfg.epochs),
                desc='epochs',
                initial=self.start_epoch,
                total=self.train_cfg.epochs,
            )
            for epoch in epoch_bar:
                train_loss = self._train_epoch(epoch)
                val_energy, cell_acc, puzzle_acc, z_var = self._validate(epoch)

                self._last_val_energy = val_energy
                self._last_cell_acc = cell_acc
                log_validation(val_energy, cell_acc, puzzle_acc, z_var, self.global_step)
                self._persist_full_checkpoint_if_due(epoch, val_energy, cell_acc)

                summary = (
                    f'train_loss={train_loss:.4f} val_energy={val_energy:.4f} '
                    f'cell_acc={cell_acc:.4f} puzzle_acc={puzzle_acc:.4f}'
                )
                epoch_bar.set_postfix_str(summary, refresh=False)
                logger.info(
                    'Epoch %d | train_loss=%.4f | val_energy=%.4f | cell_acc=%.4f | puzzle_acc=%.4f',
                    epoch,
                    train_loss,
                    val_energy,
                    cell_acc,
                    puzzle_acc,
                )

            if self.test_loader is not None:
                test_energy, cell_acc, puzzle_acc, z_var = self._evaluate_loader(self.test_loader, desc='held-out test')
                logger.info(
                    'Held-out test | energy=%.4f | cell_acc=%.4f | puzzle_acc=%.4f | z_var=%.4f',
                    test_energy,
                    cell_acc,
                    puzzle_acc,
                    z_var,
                )

        best = self.checkpoint_mgr.best_path
        if best:
            upload_checkpoint_to_wandb(best)
        finish_wandb()

    def _checkpoint_due(self) -> bool:
        return time.monotonic() - self._last_ckpt_time >= self.train_cfg.checkpoint_interval_sec

    def _checkpoint_data(self, epoch: int, val_energy: float, cell_acc: float) -> _CheckpointData:
        return _CheckpointData(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            step=self.global_step,
            val_energy=val_energy,
            cell_accuracy=cell_acc,
        )

    def _persist_full_checkpoint_if_due(self, epoch: int, val_energy: float, cell_acc: float) -> None:
        """Top-K + ``checkpoint.last.pt`` after epoch validation when the interval has elapsed."""
        if not self._checkpoint_due():
            return
        ckpt_data = self._checkpoint_data(epoch, val_energy, cell_acc)
        self.checkpoint_mgr.save(ckpt_data)
        self.checkpoint_mgr.save_last(ckpt_data, self.scheduler)
        self._last_ckpt_time = time.monotonic()
        logger.info(
            'Checkpoint at epoch %d step %d (cell_acc=%.4f)',
            epoch,
            self.global_step,
            cell_acc,
        )

    def _save_last_if_due(self, epoch: int) -> None:
        """Resume checkpoint only (no validation); uses metrics from the last epoch val."""
        if not self._checkpoint_due():
            return
        ckpt_data = self._checkpoint_data(epoch, self._last_val_energy, self._last_cell_acc)
        self.checkpoint_mgr.save_last(ckpt_data, self.scheduler)
        self._last_ckpt_time = time.monotonic()
        logger.info('Resume checkpoint at epoch %d step %d', epoch, self.global_step)

    def _train_epoch(self, epoch: int) -> float:
        """
        Run one training epoch.

        Returns:
            Mean training loss for the epoch.

        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        batch_bar = tqdm(self.train_loader, desc=f'epoch {epoch} train', leave=False)
        for batch in batch_bar:
            puzzle = batch['puzzle'].to(self.device)
            solution = batch['solution'].to(self.device)
            mask = batch['mask'].to(self.device)

            out = self.model(puzzle, solution, mask)
            loss_out = compute_loss(out, solution, mask, self.train_cfg)

            self.optimizer.zero_grad()
            loss_out.total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.grad_clip_norm)
            self.optimizer.step()
            self.scheduler.step()

            momentum = get_ema_momentum(self.global_step, self.total_steps, self.train_cfg)
            self.model.update_target_encoder(momentum)

            lr = float(self.scheduler.get_last_lr()[0])
            log_train_step(loss_out, lr, momentum, self.global_step)

            loss_val = loss_out.total.item()
            total_loss += loss_val
            n_batches += 1
            self.global_step += 1
            batch_bar.set_postfix(loss=f'{loss_val:.4f}', refresh=False)

            self._save_last_if_due(epoch)

        return total_loss / max(n_batches, 1)

    def _validate(self, epoch: int) -> tuple[float, float, float, float]:
        """Run validation for one epoch."""
        return self._evaluate_loader(self.val_loader, desc=f'epoch {epoch} val')

    @torch.no_grad()
    def _evaluate_loader(self, loader: DataLoader, desc: str) -> tuple[float, float, float, float]:
        """
        Run forward-pass metrics on a dataloader (validation or held-out test).

        Returns:
            Tuple of (mean_energy, cell_accuracy, puzzle_accuracy, z_variance).

        """
        self.model.eval()
        total_energy = 0.0
        all_cell_acc = 0.0
        all_puzzle_acc = 0.0
        all_z_var = 0.0
        n_batches = 0

        for batch in tqdm(loader, desc=desc, leave=False):
            puzzle = batch['puzzle'].to(self.device)
            solution = batch['solution'].to(self.device)
            mask = batch['mask'].to(self.device)

            out = self.model(puzzle, solution, mask)
            total_energy += out.energy.mean().item()

            pred_digits = out.decode_logits.argmax(dim=-1) + 1
            all_cell_acc += compute_cell_accuracy(pred_digits, solution, mask)
            all_puzzle_acc += compute_puzzle_accuracy(pred_digits, solution)
            all_z_var += compute_z_variance(out.z_pred)
            n_batches += 1

        divisor = max(n_batches, 1)
        return total_energy / divisor, all_cell_acc / divisor, all_puzzle_acc / divisor, all_z_var / divisor
