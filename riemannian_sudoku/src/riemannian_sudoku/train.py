"""CLI for Riemannian Sudoku training / eval with W&B."""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from riemannian_sudoku.data import SudokuDataset, SudokuTorchDataset, split_train_val
from riemannian_sudoku.geometry import sudoku_group_indices
from riemannian_sudoku.model import RiemannianSudokuSolver, build_solver

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

WANDB_PROJECT = 'riemannian-sudoku'

# Fixed (27, 9) layout from sudoku_group_indices: 9 rows, 9 cols, 9 boxes.
_GROUP_INDICES = sudoku_group_indices()


def _resolve_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _resolve_mixer(cli_mixer: str | None, ckpt_args: dict | None = None) -> str:
    """CLI override wins; else ckpt ``args.mixer``; else ``euclid_axial``."""
    if cli_mixer:
        return cli_mixer
    if ckpt_args:
        return str(ckpt_args.get('mixer', 'euclid_axial'))
    return 'euclid_axial'


def _resolve_use_gauge(cli_no_gauge: bool, ckpt_args: dict | None = None) -> bool:
    """``--no-gauge`` wins; else ckpt ``args.use_gauge``; else True (live T² on)."""
    if cli_no_gauge:
        return False
    if ckpt_args is not None and 'use_gauge' in ckpt_args:
        return bool(ckpt_args['use_gauge'])
    return True


def _resolve_mlp_activation(cli_act: str | None, ckpt_args: dict | None = None) -> str:
    """CLI ``--mlp-activation`` wins; else ckpt; else ``gelu`` (cross MLP default)."""
    if cli_act:
        return cli_act
    if ckpt_args is not None and 'mlp_activation' in ckpt_args:
        return str(ckpt_args['mlp_activation'])
    return 'gelu'


def _resolve_mlp_layers(cli_layers: int | None, ckpt_args: dict | None = None) -> int:
    """CLI ``--mlp-layers`` wins when set (>0); else ckpt; else 1 (single mid block)."""
    if cli_layers is not None and int(cli_layers) > 0:
        return int(cli_layers)
    if ckpt_args is not None and 'mlp_layers' in ckpt_args:
        return max(1, int(ckpt_args['mlp_layers']))
    return 1


def _resolve_mlp_width(cli_width: str | None, ckpt_args: dict | None = None) -> str:
    """CLI ``--mlp-width`` wins; else ckpt; else ``default`` (hidden=512)."""
    if cli_width:
        return cli_width
    if ckpt_args is not None and 'mlp_width' in ckpt_args:
        return str(ckpt_args['mlp_width'])
    return 'default'


def _cosine_t_max(*, steps: int, step: int, lr_cosine_steps: int) -> int | None:
    """CosineAnnealingLR ``T_max``, or ``None`` to hold the floor (past horizon).

    ``lr_cosine_steps=0`` → cool over remaining train steps (legacy).
    ``lr_cosine_steps>0`` → cool over that many global steps, then hold ``η_min``.
    """
    if lr_cosine_steps <= 0:
        return max(steps - step, 1)
    if step >= lr_cosine_steps:
        return None
    return max(lr_cosine_steps - step, 1)


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
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Blank / exact / row / col / box accuracy on a loader."""
    model.eval()
    blank_sum = 0.0
    exact_sum = 0.0
    row_sum = 0.0
    col_sum = 0.0
    box_sum = 0.0
    n = 0
    for batch in loader:
        puzzle = batch['puzzle'].to(device, non_blocking=True)
        solution = batch['solution'].to(device, non_blocking=True)
        mask = batch['mask'].to(device, non_blocking=True)
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
    model: torch.nn.Module,
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
                'kinetic_weight': args.kinetic_weight,
                'gauge_kinetic_weight': args.gauge_kinetic_weight,
                'constraint_weight': args.constraint_weight,
                'supervise_all_steps': args.supervise_all_steps,
                'no_augment': args.no_augment,
                'mixer': args.mixer,
                'use_gauge': args.use_gauge,
                'mlp_activation': args.mlp_activation,
                'mlp_layers': args.mlp_layers,
                'mlp_width': args.mlp_width,
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

    use_cuda = device.type == 'cuda'
    pin = bool(args.pin_memory and use_cuda)
    nw = int(args.num_workers)
    loader_kw: dict = {
        'num_workers': nw,
        'pin_memory': pin,
        'persistent_workers': nw > 0,
    }
    if nw > 0:
        loader_kw['prefetch_factor'] = max(2, int(args.prefetch_factor))
    train_loader = DataLoader(
        SudokuTorchDataset(train_df, augment=not args.no_augment, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kw,
    )
    val_loader = DataLoader(
        SudokuTorchDataset(val_df, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kw,
    )
    logger.info('DataLoader workers=%d pin_memory=%s', nw, pin)

    # Peek resume ckpt before building the model so mixer matches the module tree
    # (cross_coupled vs euclid_axial). Same pattern as eval_cmd.
    resume_ckpt = None
    resume_path: Path | None = None
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f'resume checkpoint not found: {resume_path}')
        resume_ckpt = torch.load(resume_path, map_location='cpu', weights_only=False)

    ckpt_args = (resume_ckpt.get('args') or {}) if resume_ckpt is not None else None
    args.mixer = _resolve_mixer(args.mixer, ckpt_args)
    args.use_gauge = _resolve_use_gauge(bool(getattr(args, 'no_gauge', False)), ckpt_args)
    args.mlp_activation = _resolve_mlp_activation(getattr(args, 'mlp_activation', None), ckpt_args)
    args.mlp_layers = _resolve_mlp_layers(getattr(args, 'mlp_layers', None), ckpt_args)
    args.mlp_width = _resolve_mlp_width(getattr(args, 'mlp_width', None), ckpt_args)

    model = build_solver(
        args.mixer,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_steps=args.num_steps,
        dt=args.dt,
        kinetic_weight=args.kinetic_weight,
        gauge_kinetic_weight=args.gauge_kinetic_weight,
        constraint_weight=args.constraint_weight,
        supervise_all_steps=args.supervise_all_steps,
        use_gauge=args.use_gauge,
        mlp_activation=args.mlp_activation,
        mlp_layers=args.mlp_layers,
        mlp_width=args.mlp_width,
    ).to(device)
    args.mlp_layers = int(getattr(model, 'mlp_layers', 1))
    args.mlp_width = str(getattr(model, 'mlp_width', 'default'))
    # Model may force use_gauge=False for non-cross mixers; keep summary in sync.
    args.use_gauge = bool(getattr(model, 'use_gauge', False))
    if args.compile:
        # Desk ROCm 7.1: mode=default (reduce-overhead/CUDAGraphs hit HSA packet errors).
        # Same hot-path wrap idea as sfprm_miniarc/train.py --compile.
        model.forward = torch.compile(model.forward, mode='default')  # type: ignore[method-assign]
        logger.info('torch.compile: model.forward (mode=default)')
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

    if resume_ckpt is not None:
        assert resume_path is not None
        model.load_state_dict(resume_ckpt['model'])
        if 'optimizer' in resume_ckpt:
            opt.load_state_dict(resume_ckpt['optimizer'])
            # Move optimizer state tensors to device.
            for state in opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        step = int(resume_ckpt.get('step', 0))
        best_blank = float(resume_ckpt.get('best_blank', -1.0))
        logger.info(
            'Resumed from %s at step %d (best_blank=%.4f, mixer=%s)',
            resume_path,
            step,
            best_blank,
            args.mixer,
        )

    sched = None
    if args.lr_schedule == 'cosine':
        # Floor at 10% of peak. Optional --lr-cosine-steps decouples T_max from --steps
        # so long runs can cool like a short A/B (avoids stretched-cosine cliff).
        floor = args.lr * 0.1
        t_max = _cosine_t_max(steps=args.steps, step=step, lr_cosine_steps=args.lr_cosine_steps)
        if t_max is None:
            for pg in opt.param_groups:
                pg['lr'] = floor
            logger.info('Cosine horizon passed (step=%d); holding lr=%.6g', step, floor)
        else:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max, eta_min=floor)
            logger.info('CosineAnnealingLR T_max=%d eta_min=%.6g', t_max, floor)

    early_bad = 0
    early_stopped = False

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
                'lr_cosine_steps': args.lr_cosine_steps,
                'grad_clip': args.grad_clip,
                'early_stop_drop': args.early_stop_drop,
                'early_stop_patience': args.early_stop_patience,
                'kinetic_weight': args.kinetic_weight,
                'gauge_kinetic_weight': args.gauge_kinetic_weight,
                'constraint_weight': args.constraint_weight,
                'supervise_all_steps': args.supervise_all_steps,
                'no_augment': args.no_augment,
                'mixer': args.mixer,
                'use_gauge': args.use_gauge,
                'mlp_activation': args.mlp_activation,
                'mlp_layers': args.mlp_layers,
                'mlp_width': args.mlp_width,
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
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    t_wall0 = time.perf_counter()
    # Discard compile/cudagraph capture from steady-state it/s (first N steps).
    warmup_steps = int(args.throughput_warmup)
    t_steady0: float | None = None
    steady_start_step = 0
    while step < args.steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        nb = pin
        puzzle = batch['puzzle'].to(device, non_blocking=nb)
        solution = batch['solution'].to(device, non_blocking=nb)
        mask = batch['mask'].to(device, non_blocking=nb)
        out = model(puzzle, solution, mask)
        opt.zero_grad(set_to_none=True)
        out['loss'].backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        opt.step()
        # Stop stepping after T_max so LR holds at eta_min (no cosine rebound).
        if sched is not None and sched.last_epoch < sched.T_max:
            sched.step()
        step += 1
        if t_steady0 is None and step >= warmup_steps:
            if use_cuda:
                torch.cuda.synchronize()
            t_steady0 = time.perf_counter()
            steady_start_step = step

        blank_acc, exact = _blank_and_exact(out['pred'].detach(), solution, mask)
        units = unit_group_accuracies(out['pred'].detach(), solution)
        metrics = {
            'train/loss': float(out['loss'].detach()),
            'train/task': float(out['task_loss']),
            'train/kinetic': float(out['kinetic_loss']),
            'train/gauge_kinetic': float(out['gauge_kinetic_loss']),
            'train/constraint': float(out['constraint_loss']),
            'train/blank_acc': blank_acc,
            'train/exact_board': exact,
            'train/row_acc': units['row_acc'],
            'train/col_acc': units['col_acc'],
            'train/box_acc': units['box_acc'],
            'train/lr': float(opt.param_groups[0]['lr']),
            'train/step': step,
        }
        # Bipartite diags: ||ω_1||², ||ω_K||², scratch/claim energy, NP residual, logit_scale.
        for key, wandb_key in (
            ('omega_sq_1', 'train/omega_sq_1'),
            ('omega_sq_k', 'train/omega_sq_k'),
            ('gate_mean', 'train/gate_mean'),
            ('scratch_energy', 'train/scratch_energy'),
            ('blank_np', 'train/blank_np'),
            ('claim_energy', 'train/claim_energy'),
            ('logit_scale', 'train/logit_scale'),
        ):
            if key in out:
                metrics[wandb_key] = float(out[key])
        pbar.set_postfix(
            loss=f'{metrics["train/loss"]:.3f}',
            blank=f'{blank_acc:.2f}',
            kin=f'{metrics["train/kinetic"]:.3f}',
            row=f'{units["row_acc"]:.2f}',
        )
        pbar.update(1)
        if step % args.log_every == 0:
            logger.info(
                'step %d train loss=%.4f task=%.4f kinetic=%.6f gauge_kinetic=%.6f blank=%.4f',
                step,
                metrics['train/loss'],
                metrics['train/task'],
                metrics['train/kinetic'],
                metrics['train/gauge_kinetic'],
                blank_acc,
            )
            if wandb_run is not None:
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
                early_bad = 0
                _save_ckpt(best_path, model, opt, step, n_params, best_blank, args)
                logger.info('Saved best %s (blank=%.4f)', best_path, best_blank)
            elif (
                args.early_stop_drop > 0
                and args.early_stop_patience > 0
                and best_blank >= 0
                and (best_blank - val['blank_acc']) >= args.early_stop_drop
            ):
                early_bad += 1
                logger.info(
                    'Early-stop warn %d/%d: blank=%.4f is >=%.3f below best=%.4f',
                    early_bad,
                    args.early_stop_patience,
                    val['blank_acc'],
                    args.early_stop_drop,
                    best_blank,
                )
                if early_bad >= args.early_stop_patience:
                    early_stopped = True
                    logger.info('Early-stop at step %d; keeping best.pt (blank=%.4f)', step, best_blank)
                    model.train()
                    break
            else:
                early_bad = 0
            model.train()

        if args.ckpt_every > 0 and step % args.ckpt_every == 0:
            _save_ckpt(last_path, model, opt, step, n_params, best_blank, args)
            logger.info('Saved last %s at step %d', last_path, step)

    pbar.close()
    if use_cuda:
        torch.cuda.synchronize()
    wall_s = time.perf_counter() - t_wall0
    steps_done = max(step, 1)
    it_s = steps_done / wall_s if wall_s > 0 else 0.0
    steady_steps = max(step - steady_start_step, 0)
    if t_steady0 is not None and steady_steps > 0:
        steady_s = time.perf_counter() - t_steady0
        it_s_steady = steady_steps / steady_s if steady_s > 0 else 0.0
    else:
        steady_s = 0.0
        it_s_steady = it_s
    max_vram_gb = (
        torch.cuda.max_memory_allocated(device) / (1024**3) if use_cuda else 0.0
    )
    logger.info(
        'Throughput: %.2f it/s overall (%d steps, %.1fs); steady %.2f it/s after warmup=%d (%d steps, %.1fs); max_vram=%.2f GiB',
        it_s,
        steps_done,
        wall_s,
        it_s_steady,
        warmup_steps,
        steady_steps,
        steady_s,
        max_vram_gb,
    )

    _save_ckpt(last_path, model, opt, step, n_params, best_blank, args)
    logger.info('Saved %s', last_path)

    # Headline final metrics from best.pt when available (esp. after early-stop).
    if best_path.is_file() and best_blank >= 0:
        ckpt = torch.load(best_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        logger.info('Loaded best.pt for final eval (blank=%.4f @ step %s)', best_blank, ckpt.get('step'))
    final_val = evaluate(model, val_loader, device)
    logger.info(
        'Final val blank=%.4f row=%.4f col=%.4f box=%.4f exact=%.4f%s',
        final_val['blank_acc'],
        final_val['row_acc'],
        final_val['col_acc'],
        final_val['box_acc'],
        final_val['exact_board'],
        ' (early-stop)' if early_stopped else '',
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
                'train/early_stopped': int(early_stopped),
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
    print(f'IT_S={it_s:.4f}')
    print(f'IT_S_STEADY={it_s_steady:.4f}')
    print(f'WALL_S={wall_s:.2f}')
    print(f'MAX_VRAM_GB={max_vram_gb:.3f}')
    print(f'COMPILE={int(args.compile)}')
    print(f'MIXER={args.mixer}')
    print(f'USE_GAUGE={int(args.use_gauge)}')
    print(f'MLP_ACTIVATION={args.mlp_activation}')
    print(f'MLP_LAYERS={args.mlp_layers}')
    print(f'MLP_WIDTH={args.mlp_width}')
    print(f'EARLY_STOPPED={int(early_stopped)}')
    print(f'STOP_STEP={step}')


def eval_cmd(args: argparse.Namespace) -> None:
    """Evaluate a checkpoint on HF test (or train-head holdout)."""
    device = _resolve_device()
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    ckpt_args = ckpt.get('args') or {}
    mixer = _resolve_mixer(getattr(args, 'mixer', None), ckpt_args)
    use_gauge = _resolve_use_gauge(bool(getattr(args, 'no_gauge', False)), ckpt_args)
    mlp_activation = _resolve_mlp_activation(getattr(args, 'mlp_activation', None), ckpt_args)
    mlp_layers = _resolve_mlp_layers(getattr(args, 'mlp_layers', None), ckpt_args)
    mlp_width = _resolve_mlp_width(getattr(args, 'mlp_width', None), ckpt_args)
    model = build_solver(
        mixer,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_steps=args.num_steps,
        dt=args.dt,
        use_gauge=use_gauge,
        mlp_activation=mlp_activation,
        mlp_layers=mlp_layers,
        mlp_width=mlp_width,
        gauge_kinetic_weight=float(ckpt_args.get('gauge_kinetic_weight', 0.0)),
        supervise_all_steps=bool(ckpt_args.get('supervise_all_steps', False)),
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
    t.add_argument(
        '--lr-cosine-steps',
        type=int,
        default=0,
        help='Cosine T_max in global steps (0=remaining --steps). After horizon, hold 0.1·peak',
    )
    t.add_argument('--grad-clip', type=float, default=1.0, help='clip_grad_norm_ max_norm; 0 disables')
    t.add_argument(
        '--early-stop-drop',
        type=float,
        default=0.0,
        help='Stop if val blank is this far below best for --early-stop-patience evals (0=off)',
    )
    t.add_argument(
        '--early-stop-patience',
        type=int,
        default=0,
        help='Consecutive cliff evals before early-stop (0=off)',
    )
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
    t.add_argument('--kinetic-weight', type=float, default=0.05, help='λ_k on final-step cell velocity energy')
    t.add_argument(
        '--gauge-kinetic-weight',
        type=float,
        default=0.0,
        help='λ_g on final-step gauge velocity energy (cross_coupled only; 0=off, axial unchanged)',
    )
    t.add_argument('--constraint-weight', type=float, default=0.1, help='λ_c on unit-sum MSE')
    t.add_argument(
        '--supervise-all-steps',
        action='store_true',
        help='Average blank NLL over ODE steps (default: final state only)',
    )
    t.add_argument(
        '--mixer',
        type=str,
        default=None,
        choices=[
            'euclid_axial',
            'cross_coupled',
            'geo_karcher',
            'geo_log',
            'geo_transformer',
            'bipartite_sn_tm',
        ],
        help='velocity mixer (default euclid_axial; on --resume reads ckpt args unless set)',
    )
    t.add_argument(
        '--no-gauge',
        action='store_true',
        help=(
            'cross_coupled cells-only ablation (omit live T²): 729→MLP→729. '
            'Default OFF = Method 1+2 live T² init + MLP updates g (765→MLP→765)'
        ),
    )
    t.add_argument(
        '--mlp-activation',
        type=str,
        default=None,
        choices=['gelu', 'silu', 'swiglu'],
        help='cross_coupled ambient MLP nonlinearity (default gelu; axial unchanged; on --resume reads ckpt)',
    )
    t.add_argument(
        '--mlp-layers',
        type=int,
        default=None,
        help='cross_coupled mid nonlinear blocks (default 1; swiglu: SwiGLU×N; on --resume reads ckpt)',
    )
    t.add_argument(
        '--mlp-width',
        type=str,
        default=None,
        choices=['default', 'expand'],
        help=(
            'cross_coupled MLP hidden: default=512; expand=2·in_dim '
            '(live gauge: 1530; --no-gauge: 1458); on --resume reads ckpt'
        ),
    )
    t.add_argument(
        '--compile',
        action='store_true',
        help='wrap model.forward with torch.compile(mode=default); default off',
    )
    t.add_argument('--num-workers', type=int, default=4, help='DataLoader workers (0=main process)')
    t.add_argument(
        '--pin-memory',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='pin_memory for CUDA H2D (default on; --no-pin-memory to disable)',
    )
    t.add_argument('--prefetch-factor', type=int, default=2, help='DataLoader prefetch_factor when workers>0')
    t.add_argument(
        '--throughput-warmup',
        type=int,
        default=100,
        help='Steps excluded from IT_S_STEADY (compile capture / dataloader fill)',
    )

    e = sub.add_parser('eval', help='Evaluate checkpoint')
    e.add_argument('--checkpoint', type=str, required=True)
    e.add_argument('--split', type=str, default='test', choices=['train', 'test'])
    e.add_argument('--n-samples', type=int, default=256)
    e.add_argument('--batch-size', type=int, default=32)
    e.add_argument('--d-model', type=int, default=128)
    e.add_argument('--n-heads', type=int, default=4)
    e.add_argument('--num-steps', type=int, default=8)
    e.add_argument('--dt', type=float, default=0.5)
    e.add_argument(
        '--mixer',
        type=str,
        default=None,
        choices=[
            'euclid_axial',
            'cross_coupled',
            'geo_karcher',
            'geo_log',
            'geo_transformer',
            'bipartite_sn_tm',
        ],
        help='override mixer; default reads ckpt args',
    )
    e.add_argument(
        '--no-gauge',
        action='store_true',
        help='force cells-only (omit live T²); default reads ckpt args',
    )
    e.add_argument(
        '--mlp-activation',
        type=str,
        default=None,
        choices=['gelu', 'silu', 'swiglu'],
        help='override cross MLP activation; default reads ckpt args',
    )
    e.add_argument(
        '--mlp-layers',
        type=int,
        default=None,
        help='override cross MLP mid-block count; default reads ckpt args',
    )
    e.add_argument(
        '--mlp-width',
        type=str,
        default=None,
        choices=['default', 'expand'],
        help='override cross MLP hidden width; default reads ckpt args',
    )

    args = parser.parse_args()
    if args.command == 'train':
        train_cmd(args)
    else:
        eval_cmd(args)


if __name__ == '__main__':
    main()
