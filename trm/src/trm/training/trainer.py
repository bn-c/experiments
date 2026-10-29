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
        if train_cfg.compile:
            # Compile backbone only so EMA deepcopy stays a plain Module.
            self.model.backbone = torch.compile(self.model.backbone, mode='default')  # type: ignore[method-assign]
            logger.info('torch.compile: model.backbone (mode=default)')
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
            betas=(0.9, 0.95),
        )
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        if train_cfg.lr_schedule == 'cosine':
            t_max = train_cfg.max_steps or max(train_cfg.epochs * max(len(train_loader), 1), 1)
            floor = 0.1 * train_cfg.lr
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=t_max,
                eta_min=floor,
            )
            logger.info('LR schedule: cosine T_max=%d eta_min=%.2e', t_max, floor)

    def _forward_losses(
        self,
        puzzle: torch.Tensor,
        solution: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, TrainStepMetrics]:
        """Deep-supervision forward; return (loss, metrics) without optimizer step."""
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

        denom = max(halted_count, 1)
        lr = float(self.optimizer.param_groups[0]['lr'])
        metrics = TrainStepMetrics(
            lm_loss=float(total_lm.detach().item() / max(n_loss_terms, 1)),
            q_halt_loss=float(total_q.detach().item() / max(n_loss_terms, 1)),
            exact_accuracy=exact_sum / denom,
            q_halt_accuracy=q_acc_sum / denom,
            steps=steps_sum / denom,
            lr=lr,
        )
        return loss, metrics

    def _train_batch(self, puzzle: torch.Tensor, solution: torch.Tensor, mask: torch.Tensor) -> TrainStepMetrics:
        """One microbatch forward+backward+optimizer step (tests / single-batch path)."""
        self.optimizer.zero_grad(set_to_none=True)
        loss, metrics = self._forward_losses(puzzle, solution, mask)
        loss.backward()
        return self._optimizer_step_from_metrics(metrics)

    def _optimizer_step_from_metrics(self, metrics: TrainStepMetrics) -> TrainStepMetrics:
        """Finish an optimizer step and refresh LR on metrics."""
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.ema.update(self.model)
        return TrainStepMetrics(
            lm_loss=metrics.lm_loss,
            q_halt_loss=metrics.q_halt_loss,
            exact_accuracy=metrics.exact_accuracy,
            q_halt_accuracy=metrics.q_halt_accuracy,
            steps=metrics.steps,
            lr=float(self.optimizer.param_groups[0]['lr']),
        )

    def train_epoch(self) -> float:
        """Run one training epoch; return mean ``lm_loss``."""
        self.model.train()
        total_lm = 0.0
        n_batches = 0
        accum = max(self.train_cfg.grad_accum_steps, 1)
        pending_lm = 0.0
        pending_metrics: TrainStepMetrics | None = None
        micros = 0

        for batch in tqdm(self.train_loader, desc='train', leave=False):
            puzzle = batch['puzzle'].to(self.device)
            solution = batch['solution'].to(self.device)
            mask = batch['mask'].to(self.device)
            if micros == 0:
                self.optimizer.zero_grad(set_to_none=True)
            loss, metrics = self._forward_losses(puzzle, solution, mask)
            (loss / accum).backward()
            pending_lm += metrics.lm_loss
            pending_metrics = metrics
            micros += 1
            if micros < accum:
                continue
            assert pending_metrics is not None
            metrics = self._optimizer_step_from_metrics(pending_metrics)
            self.global_step += 1
            log_train_step(metrics, self.global_step)
            total_lm += pending_lm / accum
            n_batches += 1
            pending_lm = 0.0
            pending_metrics = None
            micros = 0
        return total_lm / max(n_batches, 1)

    def train_steps(self, max_steps: int) -> None:
        """Train for ``max_steps`` optimizer steps with optional periodic validation."""
        self.model.train()
        accum = max(self.train_cfg.grad_accum_steps, 1)
        eval_every = max(self.train_cfg.eval_every, 1)
        log_every = max(self.train_cfg.log_every, 1)
        data_iter = iter(self.train_loader)
        pbar = tqdm(total=max_steps, desc='steps', initial=self.global_step)

        while self.global_step < max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            pending: TrainStepMetrics | None = None
            lm_sum = 0.0
            for _ in range(accum):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)
                puzzle = batch['puzzle'].to(self.device)
                solution = batch['solution'].to(self.device)
                mask = batch['mask'].to(self.device)
                loss, metrics = self._forward_losses(puzzle, solution, mask)
                (loss / accum).backward()
                lm_sum += metrics.lm_loss
                pending = metrics
            assert pending is not None
            metrics = self._optimizer_step_from_metrics(
                TrainStepMetrics(
                    lm_loss=lm_sum / accum,
                    q_halt_loss=pending.q_halt_loss,
                    exact_accuracy=pending.exact_accuracy,
                    q_halt_accuracy=pending.q_halt_accuracy,
                    steps=pending.steps,
                    lr=pending.lr,
                ),
            )
            self.global_step += 1
            pbar.update(1)
            if self.global_step % log_every == 0 or self.global_step == 1:
                log_train_step(metrics, self.global_step)
                pbar.set_postfix(lm=f'{metrics.lm_loss:.3f}', lr=f'{metrics.lr:.2e}')
            if self.validate and self.val_loader is not None and self.global_step % eval_every == 0:
                val_cell, val_puzzle = self._evaluate_loader(self.val_loader, desc='val')
                log_validation(val_cell, val_puzzle, self.global_step)
                logger.info(
                    'step %d/%d — lm_loss %.4f | val blank(cell) %.4f puzzle %.4f',
                    self.global_step,
                    max_steps,
                    metrics.lm_loss,
                    val_cell,
                    val_puzzle,
                )
                self.model.train()
                self.save_checkpoint(self.train_cfg.checkpoint_dir / 'checkpoint.last.pt')
        pbar.close()

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
        """Train for configured epochs or ``max_steps`` with optional validation and W&B logging."""
        run_name = self.train_cfg.wandb_name or 'trm-sudoku'
        init_wandb(
            self.app_cfg,
            run_name=run_name,
            extra_config={**self.model_cfg.model_dump(), **self.train_cfg.model_dump()},
        )
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        log_num_params(n_params)
        logger.info('Trainable parameters: %s', f'{n_params:,}')
        print(f'N_PARAMS={n_params}', flush=True)
        checkpoint_path = self.train_cfg.checkpoint_dir / 'checkpoint.last.pt'
        try:
            if self.train_cfg.max_steps is not None:
                self.train_steps(self.train_cfg.max_steps)
                if self.validate and self.val_loader is not None:
                    val_cell, val_puzzle = self._evaluate_loader(self.val_loader, desc='val-final')
                    log_validation(val_cell, val_puzzle, self.global_step)
                    logger.info(
                        'Final val blank(cell) %.4f puzzle %.4f @ step %d',
                        val_cell,
                        val_puzzle,
                        self.global_step,
                    )
                    print(f'VAL_BLANK={val_cell:.6f}', flush=True)
                    print(f'VAL_PUZZLE={val_puzzle:.6f}', flush=True)
                    print(f'STOP_STEP={self.global_step}', flush=True)
                self.save_checkpoint(checkpoint_path)
            else:
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
        # Unwrap compiled backbone for portable state_dict keys.
        backbone = self.model.backbone
        uncompiled = getattr(backbone, '_orig_mod', None)
        if uncompiled is not None:
            self.model.backbone = uncompiled  # type: ignore[assignment]
        try:
            torch.save(
                {
                    'model_cfg': self.model_cfg.model_dump(),
                    'model': self.model.state_dict(),
                    'ema': self.ema.state_dict(),
                    'global_step': self.global_step,
                },
                path,
            )
        finally:
            if uncompiled is not None:
                self.model.backbone = backbone  # type: ignore[assignment]
        logger.info('Saved checkpoint to %s', path)
