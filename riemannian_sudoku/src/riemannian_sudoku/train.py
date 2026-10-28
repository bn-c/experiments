"""CLI for Riemannian Sudoku training / eval with W&B."""

from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from riemannian_sudoku.data import SudokuDataset, SudokuTorchDataset, split_train_val
from riemannian_sudoku.geometry import sudoku_group_indices
from riemannian_sudoku.model import RiemannianSudokuSolver

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

WANDB_PROJECT = 'riemannian-sudoku'

# Fixed (27, 9) layout from sudoku_group_indices: 9 rows, 9 cols, 9 boxes.
_GROUP_INDICES = sudoku_group_indices()


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


def unit_group_accuracies(
    pred: torch.Tensor,
    solution: torch.Tensor,
    group_indices: torch.Tensor | None = None,
) -> dict[str, float]:
    """
    Fraction of full units (row/col/box) matching the target board.

    A unit counts if all 9 cells equal the solution (clues included).
    Keys: ``row_acc``, ``col_acc``, ``box_acc``.
    """
    groups = group_indices if group_indices is not None else _GROUP_INDICES
    groups = groups.to(device=pred.device)
    # (B, 27, 9)
    match = (pred[:, groups] == solution[:, groups]).all(dim=-1)
    row = match[:, 0:9].float().mean().item()
    col = match[:, 9:18].float().mean().item()
    box = match[:, 18:27].float().mean().item()
    return {'row_acc': float(row), 'col_acc': float(col), 'box_acc': float(box)}


@torch.no_grad()
def evaluate(model: RiemannianSudokuSolver, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Blank / exact / row / col / box accuracy on a loader."""
    model.eval()
    blank_sum = 0.0
    exact_sum = 0.0
    row_sum = 0.0
    col_sum = 0.0
    box_sum = 0.0
    n = 0
    for batch in loader:
        puzzle = batch['puzzle'].to(device)
        solution = batch['solution'].to(device)
        mask = batch['mask'].to(device)
        out = model(puzzle)
        b_acc, e_acc = _blank_and_exact(out['pred'], solution, mask)
        units = unit_group_accuracies(out['pred'], solution)
        bs = puzzle.shape[0]
        blank_sum += b_acc * bs
        exact_sum += e_acc * bs
        row_sum += units['row_acc'] * bs
        col_sum += units['col_acc'] * bs
        box_sum += units['box_acc'] * bs
        n += bs
    denom = max(n, 1)
    return {
        'blank_acc': blank_sum / denom,
        'exact_board': exact_sum / denom,
        'row_acc': row_sum / denom,
        'col_acc': col_sum / denom,
        'box_acc': box_sum / denom,
        'n': float(n),
    }


def _save_ckpt(
    path: Path,
    model: RiemannianSudokuSolver,
    opt: torch.optim.Optimizer,
    step: int,
    n_params: int,
    best_blank: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'model': model.state_dict(),
            'optimizer': opt.state_dict(),
            'step': step,
            'n_params': n_params,
            'best_blank': best_blank,
            'args': {
                'd_model': args.d_model,
                'n_heads': args.n_heads,
                'num_steps': args.num_steps,
                'dt': args.dt,
                'lr': args.lr,
                'batch_size': args.batch_size,
                'train_size': args.train_size,
                'val_size': args.val_size,
                'seed': args.seed,
            },
        },
        path,
    )


def train_cmd(args: argparse.Namespace) -> None:
    """Desk / full training run with optional W&B, interval + best ckpts, resume."""
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

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path = ckpt_dir / 'last.pt'
    best_path = ckpt_dir / 'best.pt'

    step = 0
    best_blank = -1.0
    best_exact = -1.0

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f'resume checkpoint not found: {resume_path}')
        ckpt = torch.load(resume_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            opt.load_state_dict(ckpt['optimizer'])
            # Move optimizer state tensors to device.
            for state in opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        step = int(ckpt.get('step', 0))
        best_blank = float(ckpt.get('best_blank', -1.0))
        logger.info('Resumed from %s at step %d (best_blank=%.4f)', resume_path, step, best_blank)

    sched = None
    if args.lr_schedule == 'cosine':
        # Cosine over remaining steps from current `step`.
        remaining = max(args.steps - step, 1)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=remaining, eta_min=args.lr * 0.01)

    wandb_run = None
    if not args.no_wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=os.environ.get('WANDB_ENTITY') or None,
            name=args.wandb_name or None,
            config={
                'd_model': args.d_model,
                'n_heads': args.n_heads,
                'num_steps': args.num_steps,
                'dt': args.dt,
                'train_size': args.train_size,
                'val_size': args.val_size,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'lr_schedule': args.lr_schedule,
                'max_steps': args.steps,
                'n_params': n_params,
                'seed': args.seed,
                'ckpt_every': args.ckpt_every,
                'resume': args.resume or '',
            },
            resume='allow' if args.resume else None,
        )
        logger.info('W&B run: %s', wandb_run.id)

    model.train()
    train_iter = iter(train_loader)
    pbar = tqdm(total=args.steps, initial=step, desc='train')
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
        if sched is not None:
            sched.step()
        step += 1

        blank_acc, exact = _blank_and_exact(out['pred'].detach(), solution, mask)
        units = unit_group_accuracies(out['pred'].detach(), solution)
        metrics = {
            'train/loss': float(out['loss'].detach()),
            'train/task': float(out['task_loss']),
            'train/kinetic': float(out['kinetic_loss']),
            'train/constraint': float(out['constraint_loss']),
            'train/blank_acc': blank_acc,
            'train/exact_board': exact,
            'train/row_acc': units['row_acc'],
            'train/col_acc': units['col_acc'],
            'train/box_acc': units['box_acc'],
            'train/lr': float(opt.param_groups[0]['lr']),
            'train/step': step,
        }
        pbar.set_postfix(
            loss=f'{metrics["train/loss"]:.3f}',
            blank=f'{blank_acc:.2f}',
            row=f'{units["row_acc"]:.2f}',
        )
        pbar.update(1)
        if wandb_run is not None and step % args.log_every == 0:
            wandb_run.log(metrics, step=step)

        if step % args.eval_every == 0 or step == args.steps:
            val = evaluate(model, val_loader, device)
            logger.info(
                'step %d val blank=%.4f row=%.4f col=%.4f box=%.4f exact=%.4f (n=%d)',
                step,
                val['blank_acc'],
                val['row_acc'],
                val['col_acc'],
                val['box_acc'],
                val['exact_board'],
                int(val['n']),
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        'val/blank_acc': val['blank_acc'],
                        'val/exact_board': val['exact_board'],
                        'val/row_acc': val['row_acc'],
                        'val/col_acc': val['col_acc'],
                        'val/box_acc': val['box_acc'],
                    },
                    step=step,
                )
            # Best by val blank; tie-break exact_board.
            is_best = val['blank_acc'] > best_blank or (
                math.isclose(val['blank_acc'], best_blank) and val['exact_board'] > best_exact
            )
            if is_best:
                best_blank = val['blank_acc']
                best_exact = val['exact_board']
                _save_ckpt(best_path, model, opt, step, n_params, best_blank, args)
                logger.info('Saved best %s (blank=%.4f)', best_path, best_blank)
            model.train()

        if args.ckpt_every > 0 and step % args.ckpt_every == 0:
            _save_ckpt(last_path, model, opt, step, n_params, best_blank, args)
            logger.info('Saved last %s at step %d', last_path, step)

    pbar.close()

    _save_ckpt(last_path, model, opt, step, n_params, best_blank, args)
    logger.info('Saved %s', last_path)

    final_val = evaluate(model, val_loader, device)
    logger.info(
        'Final val blank=%.4f row=%.4f col=%.4f box=%.4f exact=%.4f',
        final_val['blank_acc'],
        final_val['row_acc'],
        final_val['col_acc'],
        final_val['box_acc'],
        final_val['exact_board'],
    )
    run_id = None
    if wandb_run is not None:
        wandb_run.log(
            {
                'val/blank_acc': final_val['blank_acc'],
                'val/exact_board': final_val['exact_board'],
                'val/row_acc': final_val['row_acc'],
                'val/col_acc': final_val['col_acc'],
                'val/box_acc': final_val['box_acc'],
                'val/best_blank': best_blank,
            },
            step=step,
        )
        run_id = wandb_run.id
        wandb_run.finish()
        logger.info('W&B run id: %s', run_id)

    # Machine-readable summary for reports / tune scripts.
    print(f'RUN_ID={run_id}')
    print(f'N_PARAMS={n_params}')
    print(f'VAL_BLANK={final_val["blank_acc"]:.6f}')
    print(f'VAL_ROW={final_val["row_acc"]:.6f}')
    print(f'VAL_COL={final_val["col_acc"]:.6f}')
    print(f'VAL_BOX={final_val["box_acc"]:.6f}')
    print(f'VAL_EXACT={final_val["exact_board"]:.6f}')
    print(f'BEST_BLANK={best_blank:.6f}')


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
    logger.info(
        'blank_acc=%.4f row=%.4f col=%.4f box=%.4f exact_board=%.4f n=%d',
        metrics['blank_acc'],
        metrics['row_acc'],
        metrics['col_acc'],
        metrics['box_acc'],
        metrics['exact_board'],
        int(metrics['n']),
    )
    print(f'BLANK={metrics["blank_acc"]:.6f}')
    print(f'ROW={metrics["row_acc"]:.6f}')
    print(f'COL={metrics["col_acc"]:.6f}')
    print(f'BOX={metrics["box_acc"]:.6f}')
    print(f'EXACT={metrics["exact_board"]:.6f}')


def main() -> None:
    """Entry: ``train`` / ``eval``."""
    parser = argparse.ArgumentParser(description='Riemannian Sudoku (S⁸₊ axial)')
    sub = parser.add_subparsers(dest='command', required=True)

    t = sub.add_parser('train', help='W&B training run with last/best checkpoints')
    t.add_argument('--d-model', type=int, default=128)
    t.add_argument('--n-heads', type=int, default=4)
    t.add_argument('--num-steps', type=int, default=8)
    t.add_argument('--dt', type=float, default=0.5)
    t.add_argument('--train-size', type=int, default=1024)
    t.add_argument('--val-size', type=int, default=128)
    t.add_argument('--batch-size', type=int, default=32)
    t.add_argument('--lr', type=float, default=3e-4)
    t.add_argument('--lr-schedule', type=str, default='constant', choices=['constant', 'cosine'])
    t.add_argument('--steps', type=int, default=200)
    t.add_argument('--seed', type=int, default=42)
    t.add_argument('--log-every', type=int, default=10)
    t.add_argument('--eval-every', type=int, default=50)
    t.add_argument('--ckpt-every', type=int, default=1000, help='Save last.pt every N steps (0=end only)')
    t.add_argument('--ckpt-dir', type=str, default='checkpoints')
    t.add_argument('--resume', type=str, default='', help='Path to last.pt / best.pt to resume from')
    t.add_argument('--wandb-project', type=str, default=WANDB_PROJECT)
    t.add_argument('--wandb-name', type=str, default='', help='Optional W&B run name')
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
