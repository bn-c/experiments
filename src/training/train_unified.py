"""
Unified 50-Epoch Training Pipeline.

Executes a continuous 50-epoch 1-Cycle learning rate schedule with progressive layer unfreezing,
LR dampening, mixed-precision FP16 with GradScaler, and saccadic fixation jittering.
"""

from typing import Optional, Dict, Any
import time
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from ..models.pipeline import BioInspiredVisionPipeline
from ..search.saccadic_jitter import SaccadicJitterSampler
from .param_groups import DynamicUnfreezeManager


class UnifiedTrainer:
    """
    Unified 50-Epoch Pipeline Trainer.
    
    Args:
        model: BioInspiredVisionPipeline instance.
        train_loader: DataLoader for ImageNet-100 training data.
        val_loader: DataLoader for ImageNet-100 validation data.
        epochs: Total training epochs (default: 50).
        base_max_lr: Base peak learning rate for 1-Cycle schedule (default: 0.01).
        device: Execution device ('cuda' or 'cpu').
        checkpoint_dir: Directory to store model checkpoints.
    """
    def __init__(
        self,
        model: BioInspiredVisionPipeline,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        base_max_lr: float = 0.01,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "checkpoints"
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.base_max_lr = base_max_lr
        self.checkpoint_dir = checkpoint_dir

        self.unfreeze_mgr = DynamicUnfreezeManager(dampening_epochs=3)
        self.jitter_sampler = SaccadicJitterSampler(sigma=0.08)

        # 1. Parameter groups
        param_groups = self.unfreeze_mgr.build_param_groups(self.model)
        self.optimizer = torch.optim.SGD(
            param_groups,
            lr=base_max_lr * 0.01,
            momentum=0.9,
            weight_decay=5e-4
        )

        # 2. Unified 1-Cycle step scheduler across all 50 epochs
        total_steps = len(train_loader) * epochs
        max_lrs = [group["lr_mult"] * base_max_lr for group in param_groups]
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=max_lrs,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=100.0
        )

        self.scaler = GradScaler(enabled=(self.device.type == "cuda"))
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Trains a single epoch with dynamic unfreezing update."""
        self.unfreeze_mgr.update_epoch(epoch, self.optimizer)
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (x_batch, y_batch) in enumerate(self.train_loader):
            x_batch = x_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            # Saliency-guided fixation jitter
            gaze = self.jitter_sampler.sample_gaze_centers(x_batch)

            self.optimizer.zero_grad()
            with autocast(enabled=(self.device.type == "cuda"), dtype=torch.float16):
                logits = self.model(x_batch, gaze_center=gaze)
                loss = self.criterion(logits, y_batch)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item() * y_batch.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        epoch_loss = total_loss / max(1, total)
        epoch_acc = (correct / max(1, total)) * 100.0
        return {"loss": epoch_loss, "top1": epoch_acc}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Evaluates model on validation dataset."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for x_val, y_val in self.val_loader:
            x_val = x_val.to(self.device, non_blocking=True)
            y_val = y_val.to(self.device, non_blocking=True)

            with autocast(enabled=(self.device.type == "cuda"), dtype=torch.float16):
                logits = self.model(x_val)
                loss = self.criterion(logits, y_val)

            total_loss += loss.item() * y_val.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y_val).sum().item()
            total += y_val.size(0)

        val_loss = total_loss / max(1, total)
        val_acc = (correct / max(1, total)) * 100.0
        return {"val_loss": val_loss, "val_top1": val_acc}
