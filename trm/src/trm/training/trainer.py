"""Training loop with deep supervision, halt Q-head, and EMA."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from trm.config import AppConfig, ModelConfig, TrainConfig
from trm.eval.metrics import compute_metrics
from trm.model.ema import EMAModel
from trm.model.trm_mlp import TRMMLP
from trm.training.loss import masked_cross_entropy_per_sample, sequence_is_correct
from trm.training.metrics import (
    TrainStepMetrics,
    finish_wandb,
    init_wandb,
    log_num_params,
    log_train_step,
    log_validation,
    upload_checkpoint_to_wandb,
)

logger = logging.getLogger(__name__)


class Trainer:
    """Single-GPU trainer for TRM deep supervision + halt Q-head."""

    def __init__(  # noqa: PLR0913
        self,
        model: TRMMLP,
        train_loader: DataLoader,
        model_cfg: ModelConfig,
        train_cfg: TrainConfig,
        device: torch.device,
        *,
        val_loader: DataLoader | None = None,
        validate: bool = True,
    ) -> None:
        """Wire model, loaders, optimizer, and optional validation."""
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.validate = validate and val_loader is not None
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

    def _train_batch(self, puzzle: torch.Tensor, solution: torch.Tensor, mask: torch.Tensor) -> TrainStepMetrics:
        """One batch: deep-supervision steps with ACT-style halt metrics."""
        self.optimizer.zero_grad(set_to_none=True)
        batch_size = puzzle.shape[0]
        n_sup = self.model_cfg.max_supervision_steps
        x = self.model.embed_puzzle(puzzle)
        y, z = self.model.initial_state(batch_size, puzzle.device)

        active = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        total_lm = puzzle.new_zeros(())
        total_q = puzzle.new_zeros(())
        n_loss_terms = 0
        halted_count = 0
        exact_sum = 0.0
        q_acc_sum = 0.0
        steps_sum = 0.0

        for step_i in range(1, n_sup + 1):
            y, z, logits, q_halt = self.model.supervision_step(x, y, z)
            seq_correct = sequence_is_correct(logits, solution, mask)
            lm_per = masked_cross_entropy_per_sample(logits, solution, mask)
            q_per = F.binary_cross_entropy_with_logits(
                q_halt,
                seq_correct.to(dtype=q_halt.dtype),
                reduction='none',
            )
            active_f = active.to(dtype=lm_per.dtype)
            n_active = int(active_f.sum().item())
            if n_active > 0:
                total_lm = total_lm + (lm_per * active_f).sum() / n_active
                total_q = total_q + (q_per * active_f).sum() / n_active
                n_loss_terms += 1

            # Paper no_ACT_continue: halt when q_halt > 0; always halt on last step.
            should_halt = (q_halt.detach() > 0) | (step_i == n_sup)
            newly = active & should_halt
            if newly.any():
                n_new = int(newly.sum().item())
                halted_count += n_new
                exact_sum += float((newly & seq_correct).sum().item())
                q_pred_halt = q_halt.detach() >= 0
                q_acc_sum += float((newly & (q_pred_halt == seq_correct)).sum().item())
                steps_sum += float(step_i * n_new)
                active = active & ~newly
            if not active.any():
                break

        loss = total_lm + 0.5 * total_q
        if n_loss_terms > 1:
            loss = loss / n_loss_terms
        loss.backward()
        self.optimizer.step()
        self.ema.update(self.model)

        denom = max(halted_count, 1)
        return TrainStepMetrics(
            lm_loss=float(total_lm.detach().item() / max(n_loss_terms, 1)),
            q_halt_loss=float(total_q.detach().item() / max(n_loss_terms, 1)),
            exact_accuracy=exact_sum / denom,
            q_halt_accuracy=q_acc_sum / denom,
            steps=steps_sum / denom,
            lr=float(self.train_cfg.lr),
        )

    def train_epoch(self) -> float:
        """Run one training epoch; return mean ``lm_loss``."""
        self.model.train()
        total_lm = 0.0
        n_batches = 0
        for batch in tqdm(self.train_loader, desc='train', leave=False):
            puzzle = batch['puzzle'].to(self.device)
            solution = batch['solution'].to(self.device)
            mask = batch['mask'].to(self.device)
            metrics = self._train_batch(puzzle, solution, mask)
            # Paper logs train/* at each optimizer step (global_step).
            self.global_step += 1
            log_train_step(metrics, self.global_step)
            total_lm += metrics.lm_loss
            n_batches += 1
        return total_lm / max(n_batches, 1)

    @torch.no_grad()
    def _evaluate_loader(self, loader: DataLoader, desc: str) -> tuple[float, float]:
        """Run EMA shadow inference and return (cell_accuracy, puzzle_accuracy)."""
        self.ema.shadow.eval()
        all_pred: list[torch.Tensor] = []
        all_solution: list[torch.Tensor] = []
        all_mask: list[torch.Tensor] = []

        for batch in tqdm(loader, desc=desc, leave=False):
            puzzle = batch['puzzle'].to(self.device)
            logits = self.ema.shadow.forward_inference(puzzle)
            pred = logits.argmax(dim=-1) + 1
            all_pred.append(pred.cpu())
            all_solution.append(batch['solution'])
            all_mask.append(batch['mask'])

        pred_digits = torch.cat(all_pred, dim=0)
        solution = torch.cat(all_solution, dim=0)
        mask = torch.cat(all_mask, dim=0)
        metrics = compute_metrics(pred_digits, solution, mask)
        return metrics.cell_accuracy, metrics.puzzle_accuracy

    def train(self) -> None:
        """Train for configured epochs with optional validation and W&B logging."""
        init_wandb(
            self.app_cfg,
            run_name='trm-sudoku',
            extra_config={**self.model_cfg.model_dump(), **self.train_cfg.model_dump()},
        )
        n_params = sum(p.numel() for p in self.model.parameters())
        log_num_params(n_params)
        checkpoint_path = self.train_cfg.checkpoint_dir / 'checkpoint.last.pt'
        try:
            for epoch in range(self.train_cfg.epochs):
                avg_lm = self.train_epoch()
                if self.validate and self.val_loader is not None:
                    val_cell, val_puzzle = self._evaluate_loader(self.val_loader, desc='val')
                    log_validation(val_cell, val_puzzle, self.global_step)
                    logger.info(
                        'Epoch %d/%d — lm_loss %.4f | val cell %.4f puzzle %.4f',
                        epoch + 1,
                        self.train_cfg.epochs,
                        avg_lm,
                        val_cell,
                        val_puzzle,
                    )
                else:
                    logger.info('Epoch %d/%d — lm_loss %.4f', epoch + 1, self.train_cfg.epochs, avg_lm)
                self.save_checkpoint(checkpoint_path)
            upload_checkpoint_to_wandb(checkpoint_path)
        finally:
            finish_wandb()

    def save_checkpoint(self, path: Path) -> None:
        """Persist model, EMA, and model config to ``path``."""
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
