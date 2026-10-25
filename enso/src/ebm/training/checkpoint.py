"""Checkpoint save/load with best-K tracking."""

import heapq
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

logger = logging.getLogger(__name__)

LAST_CHECKPOINT_NAME = 'checkpoint.last.pt'
_TOP_K_PATTERN = re.compile(r'^checkpoint_epoch(\d+)_acc([\d.]+)\.pt$')


@dataclass(order=True)
class _CheckpointEntry:
    """Tracks a checkpoint by its metric value and path."""

    metric: float
    path: Path = field(compare=False)


@dataclass
class _CheckpointData:
    """Data to persist in a checkpoint file."""

    model: nn.Module
    optimizer: Optimizer
    epoch: int
    step: int
    val_energy: float
    cell_accuracy: float


class CheckpointManager:
    """
    Manages saving/loading checkpoints, keeping only the best K by cell accuracy.

    Higher cell accuracy is better. When a new checkpoint is saved and the limit
    is exceeded, the worst (lowest accuracy) checkpoint is deleted.
    """

    def __init__(self, checkpoint_dir: Path, keep_top_k: int = 3) -> None:
        """
        Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints in.
            keep_top_k: Number of best checkpoints to retain.

        """
        self.checkpoint_dir = checkpoint_dir
        self.keep_top_k = keep_top_k
        self._heap: list[_CheckpointEntry] = []

    def save(self, data: _CheckpointData) -> Path | None:
        """
        Save a checkpoint if it's among the best K.

        Args:
            data: Checkpoint data to save.

        Returns:
            Path to saved checkpoint, or None if not in top K.

        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f'checkpoint_epoch{data.epoch:03d}_acc{data.cell_accuracy:.4f}.pt'
        entry = _CheckpointEntry(metric=data.cell_accuracy, path=path)

        if len(self._heap) < self.keep_top_k:
            self._write(data, path)
            heapq.heappush(self._heap, entry)
            return path

        worst = self._heap[0]
        if data.cell_accuracy > worst.metric:
            self._write(data, path)
            evicted = heapq.heapreplace(self._heap, entry)
            if evicted.path.exists():
                evicted.path.unlink()
                logger.debug('Removed checkpoint %s', evicted.path.name)
            return path

        return None

    def save_last(self, data: _CheckpointData, scheduler: LRScheduler) -> Path:
        """Always write the latest training state for ``--continue`` resume."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / LAST_CHECKPOINT_NAME
        torch.save(
            {
                'model_state_dict': data.model.state_dict(),
                'optimizer_state_dict': data.optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': data.epoch,
                'step': data.step,
                'val_energy': data.val_energy,
                'cell_accuracy': data.cell_accuracy,
            },
            path,
        )
        return path

    def reindex_from_disk(self) -> None:
        """Rebuild the top-K heap from existing ``checkpoint_epoch*_acc*.pt`` files."""
        self._heap.clear()
        for path in sorted(self.checkpoint_dir.glob('checkpoint_epoch*.pt')):
            match = _TOP_K_PATTERN.match(path.name)
            if not match:
                continue
            entry = _CheckpointEntry(metric=float(match.group(2)), path=path)
            if len(self._heap) < self.keep_top_k:
                heapq.heappush(self._heap, entry)
            elif entry.metric > self._heap[0].metric:
                heapq.heapreplace(self._heap, entry)

    @property
    def best_path(self) -> Path | None:
        """Return the path of the highest-accuracy checkpoint, or None if empty."""
        if not self._heap:
            return None
        return max(self._heap).path

    @staticmethod
    def _write(data: _CheckpointData, path: Path) -> None:
        """Write checkpoint dict to disk."""
        torch.save(
            {
                'model_state_dict': data.model.state_dict(),
                'optimizer_state_dict': data.optimizer.state_dict(),
                'epoch': data.epoch,
                'step': data.step,
                'val_energy': data.val_energy,
                'cell_accuracy': data.cell_accuracy,
            },
            path,
        )
        logger.debug('Saved checkpoint %s (cell_acc=%.4f)', path.name, data.cell_accuracy)

    @staticmethod
    def load(
        path: Path,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
    ) -> dict:
        """
        Load a checkpoint from disk.

        Args:
            path: Path to checkpoint file.
            model: Model to load state into.
            optimizer: Optional optimizer to load state into.
            scheduler: Optional LR scheduler to load state into.

        Returns:
            The full checkpoint dict (with epoch, step, val_energy).

        """
        checkpoint = torch.load(path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint
