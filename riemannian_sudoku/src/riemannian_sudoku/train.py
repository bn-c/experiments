"""CLI for Riemannian Sudoku training / eval with W&B."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from riemannian_sudoku.data import SudokuDataset, SudokuTorchDataset, split_train_val
from riemannian_sudoku.model import RiemannianSudokuSolver

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

WANDB_PROJECT = 'riemannian-sudoku'


def _resolve_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _blank_and_exact(
    pred: torch.Tensor,
    solution: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[float, float]:
    correct = pred == solution
    denom = mask.sum().clamp_min(1.0)
    blank_acc = float((correct & (mask > 0)).sum().item() / denom.item())
    exact = float(correct.all(dim=1).float().mean().item())
    return blank_acc, exact


@torch.no_grad()
def evaluate(model: RiemannianSudokuSolver, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Blank-cell accuracy and exact-board match on a loader."""
    model.eval()
    blank_sum = 0.0
    exact_sum = 0.0
    n = 0
    for batch in loader:
        puzzle = batch['puzzle'].to(device)
        solution = batch['solution'].to(device)
        mask = batch['mask'].to(device)
        out = model(puzzle)
        b_acc, e_acc = _blank_and_exact(out['pred'], solution, mask)
        bs = puzzle.shape[0]
        blank_sum += b_acc * bs
        exact_sum += e_acc * bs
        n += bs
    return {
        'blank_acc': blank_sum / max(n, 1),
        'exact_board': exact_sum / max(n, 1),
        'n': float(n),
    }


def train_cmd(args: argparse.Namespace) -> None:
    """Short desk training run with optional W&B."""
    device = _resolve_device()
    logger.info('Device: %s', device)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ds = SudokuDataset()
    n_load = args.train_size + args.val_size
    df = ds.load_split('train', n_samples=n_load)
    train_df, val_df = split_train_val(df, val_size=args.val_size, seed=args.seed)
    logger.info('Train=%d Val=%d', len(train_df), len(val_df))

    train_loader = DataLoader(
        SudokuTorchDataset(train_df, augment=not args.no_augment, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        SudokuTorchDataset(val_df, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = RiemannianSudokuSolver(
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_steps=args.num_steps,
        dt=args.dt,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Trainable parameters: %s', f'{n_params:,}')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    wandb_run = None
    if not args.no_wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=os.environ.get('WANDB_ENTITY') or None,
            config={
                'd_model': args.d_model,
                'n_heads': args.n_heads,
                'num_steps': args.num_steps,
                'dt': args.dt,
                'train_size': args.train_size,
                'val_size': args.val_size,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'max_steps': args.steps,
                'n_params': n_params,
                'seed': args.seed,
            },
        )
        logger.info('W&B run: %s', wandb_run.id)

    model.train()
    step = 0
    train_iter = iter(train_loader)
    pbar = tqdm(total=args.steps, desc='train')
    while step < args.steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        puzzle = batch['puzzle'].to(device)
        solution = batch['solution'].to(device)
        mask = batch['mask'].to(device)
        out = model(puzzle, solution, mask)
        opt.zero_grad(set_to_none=True)
        out['loss'].backward()
        opt.step()
        step += 1

        blank_acc, exact = _blank_and_exact(out['pred'].detach(), solution, mask)
        metrics = {
            'train/loss': float(out['loss'].detach()),
            'train/task': float(out['task_loss']),
            'train/kinetic': float(out['kinetic_loss']),
            'train/constraint': float(out['constraint_loss']),
            'train/blank_acc': blank_acc,
            'train/exact_board': exact,
            'train/step': step,
        }
        pbar.set_postfix(loss=f'{metrics["train/loss"]:.3f}', blank=f'{blank_acc:.2f}')
        pbar.update(1)
        if wandb_run is not None and step % args.log_every == 0:
            wandb_run.log(metrics, step=step)

        if step % args.eval_every == 0 or step == args.steps:
            val = evaluate(model, val_loader, device)
            logger.info(
                'step %d val blank=%.4f exact=%.4f (n=%d)',
                step,
                val['blank_acc'],
                val['exact_board'],
                int(val['n']),
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        'val/blank_acc': val['blank_acc'],
                        'val/exact_board': val['exact_board'],
                    },
                    step=step,
                )
            model.train()

    pbar.close()

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / 'last.pt'
    torch.save({'model': model.state_dict(), 'step': step, 'n_params': n_params}, ckpt_path)
    logger.info('Saved %s', ckpt_path)

    final_val = evaluate(model, val_loader, device)
    logger.info(
        'Final val blank=%.4f exact_board=%.4f',
        final_val['blank_acc'],
        final_val['exact_board'],
    )
    run_id = None
    if wandb_run is not None:
        wandb_run.log(
            {
                'val/blank_acc': final_val['blank_acc'],
                'val/exact_board': final_val['exact_board'],
            },
            step=step,
        )
        run_id = wandb_run.id
        wandb_run.finish()
        logger.info('W&B run id: %s', run_id)

    # Print machine-readable summary for the report.
    print(f'RUN_ID={run_id}')
    print(f'N_PARAMS={n_params}')
    print(f'VAL_BLANK={final_val["blank_acc"]:.6f}')
    print(f'VAL_EXACT={final_val["exact_board"]:.6f}')


def eval_cmd(args: argparse.Namespace) -> None:
    """Evaluate a checkpoint on HF test (or train-head holdout)."""
    device = _resolve_device()
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = RiemannianSudokuSolver(
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_steps=args.num_steps,
        dt=args.dt,
    )
    model.load_state_dict(ckpt['model'])
    model.to(device)

    ds = SudokuDataset()
    split = args.split
    df = ds.load_split(split, n_samples=args.n_samples)
    loader = DataLoader(SudokuTorchDataset(df, augment=False), batch_size=args.batch_size, shuffle=False)
    metrics = evaluate(model, loader, device)
    logger.info('blank_acc=%.4f exact_board=%.4f n=%d', metrics['blank_acc'], metrics['exact_board'], int(metrics['n']))


def main() -> None:
    """Entry: ``train`` / ``eval``."""
    parser = argparse.ArgumentParser(description='Riemannian Sudoku (S⁸₊ axial)')
    sub = parser.add_subparsers(dest='command', required=True)

    t = sub.add_parser('train', help='Short W&B training run')
    t.add_argument('--d-model', type=int, default=128)
    t.add_argument('--n-heads', type=int, default=4)
    t.add_argument('--num-steps', type=int, default=8)
    t.add_argument('--dt', type=float, default=0.5)
    t.add_argument('--train-size', type=int, default=1024)
    t.add_argument('--val-size', type=int, default=128)
    t.add_argument('--batch-size', type=int, default=32)
    t.add_argument('--lr', type=float, default=3e-4)
    t.add_argument('--steps', type=int, default=200)
    t.add_argument('--seed', type=int, default=42)
    t.add_argument('--log-every', type=int, default=10)
    t.add_argument('--eval-every', type=int, default=50)
    t.add_argument('--ckpt-dir', type=str, default='checkpoints')
    t.add_argument('--wandb-project', type=str, default=WANDB_PROJECT)
    t.add_argument('--no-wandb', action='store_true')
    t.add_argument('--no-augment', action='store_true')

    e = sub.add_parser('eval', help='Evaluate checkpoint')
    e.add_argument('--checkpoint', type=str, required=True)
    e.add_argument('--split', type=str, default='test', choices=['train', 'test'])
    e.add_argument('--n-samples', type=int, default=256)
    e.add_argument('--batch-size', type=int, default=32)
    e.add_argument('--d-model', type=int, default=128)
    e.add_argument('--n-heads', type=int, default=4)
    e.add_argument('--num-steps', type=int, default=8)
    e.add_argument('--dt', type=float, default=0.5)

    args = parser.parse_args()
    if args.command == 'train':
        train_cmd(args)
    else:
        eval_cmd(args)


if __name__ == '__main__':
    main()
