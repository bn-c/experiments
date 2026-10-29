"""CLI entry point for TRM Sudoku training and evaluation."""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from trm.config import ModelConfig, TrainConfig, tiny_model_config, tiny_train_config
from trm.dataset.loader import SudokuDataset
from trm.dataset.splits import resolve_val_size, split_train_val
from trm.dataset.torch_dataset import SudokuTorchDataset
from trm.eval.metrics import compute_metrics
from trm.model.trm_mlp import TRMMLP
from trm.training.trainer import Trainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def _resolve_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _apply_model_overrides(model_cfg: ModelConfig, args: argparse.Namespace) -> ModelConfig:
    updates: dict = {}
    if args.d_model is not None:
        updates['d_model'] = args.d_model
    if args.n_layers is not None:
        updates['n_layers'] = args.n_layers
    if args.mlp_expansion is not None:
        updates['mlp_expansion'] = args.mlp_expansion
    if args.n is not None:
        updates['n_latent'] = args.n
    if args.T is not None:
        updates['T'] = args.T
    if args.max_supervision_steps is not None:
        updates['max_supervision_steps'] = args.max_supervision_steps
    return model_cfg.model_copy(update=updates) if updates else model_cfg


def _apply_train_overrides(train_cfg: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    updates: dict = {}
    if args.epochs is not None:
        updates['epochs'] = args.epochs
    if args.batch_size is not None:
        updates['batch_size'] = args.batch_size
    if args.train_size is not None:
        updates['train_size'] = args.train_size
    if args.val_size is not None:
        updates['val_size'] = args.val_size
    if args.split_seed is not None:
        updates['split_seed'] = args.split_seed
    if args.seed is not None:
        updates['seed'] = args.seed
    if args.steps is not None:
        updates['max_steps'] = args.steps
    if args.lr is not None:
        updates['lr'] = args.lr
    if args.lr_schedule is not None:
        updates['lr_schedule'] = args.lr_schedule
    if args.eval_every is not None:
        updates['eval_every'] = args.eval_every
    if args.grad_accum_steps is not None:
        updates['grad_accum_steps'] = args.grad_accum_steps
    if args.compile:
        updates['compile'] = True
    if args.checkpoint_dir is not None:
        updates['checkpoint_dir'] = Path(args.checkpoint_dir)
    if args.wandb_name is not None:
        updates['wandb_name'] = args.wandb_name
    if args.log_every is not None:
        updates['log_every'] = args.log_every
    if args.num_workers is not None:
        updates['num_workers'] = args.num_workers
    if args.pin_memory:
        updates['pin_memory'] = True
    return train_cfg.model_copy(update=updates) if updates else train_cfg


def train_cmd(args: argparse.Namespace) -> None:
    """Train TRM-MLP with optional per-epoch or step-based validation."""
    model_cfg = tiny_model_config() if args.tiny else ModelConfig()
    train_cfg = tiny_train_config() if args.tiny else TrainConfig()
    model_cfg = _apply_model_overrides(model_cfg, args)
    train_cfg = _apply_train_overrides(train_cfg, args)

    _set_seed(train_cfg.seed)
    validate = not args.no_validate
    device = _resolve_device()
    logger.info('Device: %s', device)

    ds = SudokuDataset()
    val_loader = None
    if validate:
        n_load = train_cfg.train_size + train_cfg.val_size
        df = ds.load_split('train', n_samples=n_load)
        val_n = resolve_val_size(len(df), train_cfg.val_size)
        train_df, val_df = split_train_val(df, val_size=val_n, seed=train_cfg.split_seed)
        logger.info('Train puzzles: %d | Val holdout: %d', len(train_df), len(val_df))
        val_loader = DataLoader(
            SudokuTorchDataset(val_df),
            batch_size=train_cfg.batch_size,
            shuffle=False,
            num_workers=train_cfg.num_workers,
            pin_memory=train_cfg.pin_memory and device.type == 'cuda',
        )
    else:
        train_df = ds.load_split('train', n_samples=train_cfg.train_size)
        logger.info('Train puzzles: %d (validation disabled)', len(train_df))

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
    print(f'N_PARAMS={n_params}', flush=True)

    trainer = Trainer(model, train_loader, model_cfg, train_cfg, device, val_loader=val_loader, validate=validate)
    trainer.train()


def eval_cmd(args: argparse.Namespace) -> None:
    """Evaluate a checkpoint on the HF test split."""
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
    """CLI entry for ``train`` and ``eval`` subcommands."""
    parser = argparse.ArgumentParser(description='TRM-MLP Sudoku (sapientinc/sudoku-extreme)')
    sub = parser.add_subparsers(dest='command', required=True)

    train_parser = sub.add_parser('train', help='Train TRM-MLP with deep supervision')
    train_parser.add_argument('--tiny', action='store_true', help='Smoke preset (small model, few samples)')
    train_parser.add_argument('--epochs', type=int)
    train_parser.add_argument('--steps', type=int, help='Optimizer steps (overrides epochs when set)')
    train_parser.add_argument('--batch-size', type=int)
    train_parser.add_argument('--grad-accum-steps', type=int, help='Microbatches per optimizer step')
    train_parser.add_argument('--train-size', type=int, help='Number of HF train rows (paper: 1K base puzzles)')
    train_parser.add_argument('--d-model', type=int, help='Model width')
    train_parser.add_argument('--n-layers', type=int, help='MLP-Mixer depth')
    train_parser.add_argument('--mlp-expansion', type=int, help='MLP expansion factor')
    train_parser.add_argument('--n', type=int, help='Latent recursion depth n (paper default 6)')
    train_parser.add_argument('--T', type=int, help='Deep recursion outer steps T (paper default 3)')
    train_parser.add_argument(
        '--max-steps',
        type=int,
        dest='max_supervision_steps',
        help='Deep supervision steps N_sup (paper default 16)',
    )
    train_parser.add_argument('--val-size', type=int, help='Validation holdout from HF train head (default 500)')
    train_parser.add_argument('--split-seed', type=int, help='Random seed for train/val split')
    train_parser.add_argument('--seed', type=int, help='RNG seed for torch / numpy')
    train_parser.add_argument('--lr', type=float)
    train_parser.add_argument('--lr-schedule', choices=['constant', 'cosine'])
    train_parser.add_argument('--eval-every', type=int)
    train_parser.add_argument('--log-every', type=int)
    train_parser.add_argument('--compile', action='store_true', help='torch.compile backbone')
    train_parser.add_argument('--checkpoint-dir', type=str)
    train_parser.add_argument('--wandb-name', type=str)
    train_parser.add_argument('--num-workers', type=int)
    train_parser.add_argument('--pin-memory', action='store_true')
    train_parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip validation loop and val/* W&B metrics each epoch',
    )

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
