"""CLI entry point for TRM Sudoku training and evaluation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from trm.config import ModelConfig, TrainConfig, tiny_model_config, tiny_train_config
from trm.dataset.loader import SudokuDataset
from trm.dataset.torch_dataset import SudokuTorchDataset
from trm.eval.metrics import compute_metrics
from trm.model.trm_mlp import TRMMLP
from trm.training.trainer import Trainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def _resolve_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _load_model_from_checkpoint(path: Path, tiny: bool) -> tuple[TRMMLP, ModelConfig]:
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if 'model_cfg' in ckpt:
        model_cfg = ModelConfig(**ckpt['model_cfg'])
    elif tiny:
        model_cfg = tiny_model_config()
    else:
        model_cfg = ModelConfig()
    model = TRMMLP(model_cfg)
    if 'ema' in ckpt:
        model.load_state_dict(ckpt['ema']['shadow'])
    else:
        model.load_state_dict(ckpt['model'])
    return model, model_cfg


def train_cmd(args: argparse.Namespace) -> None:
    model_cfg = tiny_model_config() if args.tiny else ModelConfig()
    train_cfg = tiny_train_config() if args.tiny else TrainConfig()

    if args.epochs is not None:
        train_cfg = train_cfg.model_copy(update={'epochs': args.epochs})
    if args.batch_size is not None:
        train_cfg = train_cfg.model_copy(update={'batch_size': args.batch_size})
    if args.train_size is not None:
        train_cfg = train_cfg.model_copy(update={'train_size': args.train_size})
    if args.n is not None:
        model_cfg = model_cfg.model_copy(update={'n_latent': args.n})
    if args.T is not None:
        model_cfg = model_cfg.model_copy(update={'T': args.T})
    if args.max_steps is not None:
        model_cfg = model_cfg.model_copy(update={'max_supervision_steps': args.max_steps})

    device = _resolve_device()
    logger.info('Device: %s', device)

    ds = SudokuDataset()
    train_df = ds.load_split('train', n_samples=train_cfg.train_size)
    logger.info('Train puzzles: %d', len(train_df))

    train_loader = DataLoader(
        SudokuTorchDataset(train_df),
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory and device.type == 'cuda',
    )

    model = TRMMLP(model_cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Trainable parameters: %s', f'{n_params:,}')

    trainer = Trainer(model, train_loader, model_cfg, train_cfg, device)
    trainer.train()


def eval_cmd(args: argparse.Namespace) -> None:
    device = _resolve_device()
    checkpoint = Path(args.checkpoint)
    model, _ = _load_model_from_checkpoint(checkpoint, tiny=args.tiny)
    model.to(device)
    model.eval()

    ds = SudokuDataset()
    test_df = ds.load_split('test', n_samples=args.n_samples)
    logger.info('Test puzzles: %d', len(test_df))

    loader = DataLoader(
        SudokuTorchDataset(test_df),
        batch_size=args.batch_size or 64,
        shuffle=False,
    )

    all_pred: list[torch.Tensor] = []
    all_solution: list[torch.Tensor] = []
    all_mask: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            puzzle = batch['puzzle'].to(device)
            logits = model.forward_inference(puzzle)
            pred = logits.argmax(dim=-1) + 1
            all_pred.append(pred.cpu())
            all_solution.append(batch['solution'])
            all_mask.append(batch['mask'])

    pred_digits = torch.cat(all_pred, dim=0)
    solution = torch.cat(all_solution, dim=0)
    mask = torch.cat(all_mask, dim=0)
    metrics = compute_metrics(pred_digits, solution, mask)
    logger.info('Results on %d puzzles:', metrics.n_puzzles)
    logger.info('  Cell accuracy (empty cells): %.4f', metrics.cell_accuracy)
    logger.info('  Puzzle accuracy:             %.4f', metrics.puzzle_accuracy)


def main() -> None:
    parser = argparse.ArgumentParser(description='TRM-MLP Sudoku (sapientinc/sudoku-extreme)')
    sub = parser.add_subparsers(dest='command', required=True)

    train_parser = sub.add_parser('train', help='Train TRM-MLP with deep supervision')
    train_parser.add_argument('--tiny', action='store_true', help='Smoke preset (small model, few samples)')
    train_parser.add_argument('--epochs', type=int)
    train_parser.add_argument('--batch-size', type=int)
    train_parser.add_argument('--train-size', type=int, help='Number of HF train rows (paper: 1K base puzzles)')
    train_parser.add_argument('--n', type=int, help='Latent recursion depth n (paper default 6)')
    train_parser.add_argument('--T', type=int, help='Deep recursion outer steps T (paper default 3)')
    train_parser.add_argument('--max-steps', type=int, help='Deep supervision steps N_sup (paper default 16)')

    eval_parser = sub.add_parser('eval', help='Evaluate on HF test split')
    eval_parser.add_argument('--checkpoint', type=str, required=True)
    eval_parser.add_argument('--n-samples', type=int, help='Limit test rows')
    eval_parser.add_argument('--batch-size', type=int)
    eval_parser.add_argument('--tiny', action='store_true', help='Checkpoint was trained with --tiny')

    args = parser.parse_args()
    if args.command == 'train':
        train_cmd(args)
    elif args.command == 'eval':
        eval_cmd(args)


if __name__ == '__main__':
    main()
