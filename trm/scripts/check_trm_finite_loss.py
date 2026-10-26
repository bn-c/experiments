#!/usr/bin/env python3
"""Assert full deep-supervision loss is finite on a real HF batch (desk smoke)."""

from __future__ import annotations

import sys

import torch
from torch.utils.data import DataLoader

from trm.config import ModelConfig, TrainConfig
from trm.dataset.loader import SudokuDataset
from trm.dataset.torch_dataset import SudokuTorchDataset
from trm.model.trm_mlp import TRMMLP
from trm.training.loss import masked_cross_entropy


def main() -> int:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_cfg = ModelConfig()
    train_cfg = TrainConfig(batch_size=4, train_size=32)
    df = SudokuDataset().load_split('train', n_samples=train_cfg.train_size)
    batch = next(iter(DataLoader(SudokuTorchDataset(df), batch_size=train_cfg.batch_size)))
    model = TRMMLP(model_cfg).to(device)
    puzzle = batch['puzzle'].to(device)
    solution = batch['solution'].to(device)
    mask = batch['mask'].to(device)
    _, step_logits = model.forward_supervised(puzzle)
    step_losses = [masked_cross_entropy(l, solution, mask) for l in step_logits]
    bad = [i for i, l in enumerate(step_losses) if not torch.isfinite(l).item()]
    total = torch.stack(step_losses).sum()
    print(f'device={device} steps={len(step_losses)} bad_steps={bad} total={float(total.item()):.4f}')
    if bad or not torch.isfinite(total).item():
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
