#!/usr/bin/env python3
"""Post-hoc K_eval on the same train/val split as a cooled desk run.

Optional eval-only damp for steps k > --damp-start:
  η_k = 1/(1+β(k-start)) · γ^(k-start) on ω_y, ω_z (and optionally u_c).
Does not retrain; leaves u_x / S^8 undamped.
"""

from __future__ import annotations

import argparse
import sys

import torch
from torch.utils.data import DataLoader

from riemannian_sudoku.data import SudokuDataset, SudokuTorchDataset, split_train_val
from riemannian_sudoku.model import build_solver
from riemannian_sudoku.train import evaluate


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--ks', type=int, nargs='+', default=[8, 12, 16, 24, 32, 64, 128, 256, 512])
    p.add_argument('--train-size', type=int, default=100_000)
    p.add_argument('--val-size', type=int, default=2048)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--damp-start', type=int, default=16, help='Damp for 1-based k > start')
    p.add_argument('--damp-beta', type=float, default=0.0, help='η = 1/(1+β(k-start))')
    p.add_argument('--damp-gamma', type=float, default=1.0, help='extra γ^(k-start); 1=off')
    p.add_argument(
        '--damp-uc',
        action='store_true',
        help='Also damp u_c (S^{54}); never damps u_x',
    )
    p.add_argument(
        '--compile',
        action='store_true',
        help='wrap model.forward with torch.compile(mode=default)',
    )
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    ckpt_args = ckpt.get('args') or {}
    use_global_torus = bool(ckpt_args.get('use_global_torus', True))
    ds = SudokuDataset()
    df = ds.load_split('train', n_samples=args.train_size + args.val_size)
    _, val_df = split_train_val(df, val_size=args.val_size, seed=args.seed)
    loader = DataLoader(
        SudokuTorchDataset(val_df, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
    )
    print(f'CKPT={args.checkpoint}')
    print(f'USE_GLOBAL_TORUS={int(use_global_torus)}')
    print(f'VAL_N={len(val_df)}')
    print(
        f'DAMP_START={args.damp_start} BETA={args.damp_beta} '
        f'GAMMA={args.damp_gamma} UC={int(args.damp_uc)} COMPILE={int(args.compile)}'
    )
    for k in args.ks:
        model = build_solver(
            'bipartite_sn_tm',
            num_steps=k,
            supervise_all_steps=bool(ckpt_args.get('supervise_all_steps', False)),
            use_global_torus=use_global_torus,
        )
        model.load_state_dict(ckpt['model'])
        model.eval_damp_start = int(args.damp_start)
        model.eval_damp_beta = float(args.damp_beta)
        model.eval_damp_gamma = float(args.damp_gamma)
        model.eval_damp_uc = bool(args.damp_uc)
        model.to(device)
        if args.compile:
            model.forward = torch.compile(model.forward, mode='default')  # type: ignore[method-assign]
        m = evaluate(model, loader, device)
        print(
            f"K={k} BLANK={m['blank_acc']:.6f} ROW={m['row_acc']:.6f} "
            f"COL={m['col_acc']:.6f} BOX={m['box_acc']:.6f} "
            f"EXACT={m['exact_board']:.6f} n={int(m['n'])}"
        )
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return 0


if __name__ == '__main__':
    sys.exit(main())
