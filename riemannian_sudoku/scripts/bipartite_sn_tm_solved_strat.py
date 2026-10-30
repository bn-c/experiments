#!/usr/bin/env python3
"""Per-board exact/blank export + solved-vs-unsolved stratification on tip ckpt."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from riemannian_sudoku.data import SudokuDataset, SudokuTorchDataset, split_train_val
from riemannian_sudoku.model import build_solver


@torch.no_grad()
def per_board_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> list[dict[str, float | int]]:
    """Return one row per board: exact (0/1), blank_correct, n_blank, blank_acc."""
    model.eval()
    rows: list[dict[str, float | int]] = []
    for batch in loader:
        puzzle = batch['puzzle'].to(device, non_blocking=True)
        solution = batch['solution'].to(device, non_blocking=True)
        mask = batch['mask'].to(device, non_blocking=True)
        pred = model(puzzle)['pred']
        correct = pred == solution
        exact = correct.all(dim=1)
        blank_correct = (correct & (mask > 0)).sum(dim=1)
        n_blank = mask.sum(dim=1).clamp_min(1.0)
        for i in range(puzzle.shape[0]):
            nb = int(n_blank[i].item())
            bc = int(blank_correct[i].item())
            rows.append(
                {
                    'exact': int(exact[i].item()),
                    'blank_correct': bc,
                    'n_blank': nb,
                    'blank_acc': bc / nb,
                }
            )
    return rows


def _hist(values: list[int]) -> dict[str, int]:
    c = Counter(values)
    return {str(k): c[k] for k in sorted(c)}


def _bucket_rating(r: float) -> str:
    if r < 10:
        return '0-9'
    if r < 20:
        return '10-19'
    if r < 40:
        return '20-39'
    if r < 80:
        return '40-79'
    if r < 160:
        return '80-159'
    return '160+'


def stratify(rows: list[dict]) -> dict:
    solved = [r for r in rows if r['exact'] == 1]
    unsolved = [r for r in rows if r['exact'] == 0]
    n = len(rows)
    n_s = len(solved)
    n_u = len(unsolved)

    def clue_hist(group: list[dict]) -> dict[str, int]:
        return _hist([int(r['n_clue']) for r in group])

    def rating_hist(group: list[dict]) -> dict[str, int]:
        return _hist([_bucket_rating(float(r['rating'])) for r in group])  # type: ignore[arg-type]

    def source_hist(group: list[dict]) -> dict[str, int]:
        c = Counter(str(r['source']) for r in group)
        return dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))

    def mean_clues(group: list[dict]) -> float:
        if not group:
            return float('nan')
        return sum(int(r['n_clue']) for r in group) / len(group)

    def mean_rating(group: list[dict]) -> float:
        if not group:
            return float('nan')
        return sum(float(r['rating']) for r in group) / len(group)

    def mean_blank(group: list[dict]) -> float:
        if not group:
            return float('nan')
        return sum(float(r['blank_acc']) for r in group) / len(group)

    # Exact rate by clue / rating / source
    by_clue: dict[str, dict[str, float | int]] = {}
    clue_groups: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        clue_groups[int(r['n_clue'])].append(r)
    for clue, g in sorted(clue_groups.items()):
        ns = sum(1 for x in g if x['exact'] == 1)
        by_clue[str(clue)] = {
            'n': len(g),
            'n_exact': ns,
            'exact_rate': ns / len(g),
            'mean_blank': mean_blank(g),
        }

    by_rating: dict[str, dict[str, float | int]] = {}
    rating_groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        rating_groups[_bucket_rating(float(r['rating']))].append(r)
    for bucket in ('0-9', '10-19', '20-39', '40-79', '80-159', '160+'):
        g = rating_groups.get(bucket, [])
        if not g:
            continue
        ns = sum(1 for x in g if x['exact'] == 1)
        by_rating[bucket] = {
            'n': len(g),
            'n_exact': ns,
            'exact_rate': ns / len(g),
            'mean_blank': mean_blank(g),
        }

    by_source: dict[str, dict[str, float | int]] = {}
    source_groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        source_groups[str(r['source'])].append(r)
    for src, g in sorted(source_groups.items(), key=lambda kv: -len(kv[1])):
        ns = sum(1 for x in g if x['exact'] == 1)
        by_source[src] = {
            'n': len(g),
            'n_exact': ns,
            'exact_rate': ns / len(g),
            'mean_blank': mean_blank(g),
        }

    return {
        'n': n,
        'n_exact': n_s,
        'exact_rate': n_s / n if n else float('nan'),
        'mean_blank_all': mean_blank(rows),
        'mean_blank_unsolved': mean_blank(unsolved),
        'mean_blank_solved': mean_blank(solved),
        'solved': {
            'n': n_s,
            'mean_clues': mean_clues(solved),
            'mean_rating': mean_rating(solved),
            'clue_hist': clue_hist(solved),
            'rating_hist': rating_hist(solved),
            'source_hist': source_hist(solved),
        },
        'unsolved': {
            'n': n_u,
            'mean_clues': mean_clues(unsolved),
            'mean_rating': mean_rating(unsolved),
            'mean_blank': mean_blank(unsolved),
            'clue_hist': clue_hist(unsolved),
            'rating_hist': rating_hist(unsolved),
            'source_hist': source_hist(unsolved),
        },
        'by_clue': by_clue,
        'by_rating': by_rating,
        'by_source': by_source,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--ks', type=int, nargs='+', default=[8, 12])
    p.add_argument('--train-size', type=int, default=100_000)
    p.add_argument('--val-size', type=int, default=2048)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--csv-out', type=str, default='')
    p.add_argument('--json-out', type=str, default='')
    p.add_argument('--compile', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    ckpt_args = ckpt.get('args') or {}

    ds = SudokuDataset()
    df = ds.load_split('train', n_samples=args.train_size + args.val_size)
    _, val_df = split_train_val(df, val_size=args.val_size, seed=args.seed)

    # Metadata currently dropped by SudokuTorchDataset — keep alongside tensors.
    n_clue = [sum(c != '0' for c in puzzle) for puzzle in val_df['puzzle']]
    has_rating = 'rating' in val_df.columns
    has_source = 'source' in val_df.columns
    ratings = (
        [float(x) for x in val_df['rating'].tolist()]
        if has_rating
        else [float('nan')] * len(val_df)
    )
    sources = (
        [str(x) for x in val_df['source'].tolist()]
        if has_source
        else ['unknown'] * len(val_df)
    )

    loader = DataLoader(
        SudokuTorchDataset(val_df, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
    )

    all_csv_rows: list[dict] = []
    summary: dict = {
        'checkpoint': args.checkpoint,
        'val_n': len(val_df),
        'has_rating': has_rating,
        'has_source': has_source,
        'ckpt_step': ckpt.get('step'),
        'ckpt_best_blank': ckpt.get('best_blank'),
        'by_k': {},
    }

    print(f'CKPT={args.checkpoint}')
    print(f'VAL_N={len(val_df)} rating={has_rating} source={has_source}')

    for k in args.ks:
        t0 = time.perf_counter()
        model = build_solver(
            'bipartite_sn_tm',
            num_steps=k,
            supervise_all_steps=bool(ckpt_args.get('supervise_all_steps', False)),
            use_global_torus=bool(ckpt_args.get('use_global_torus', True)),
        )
        model.load_state_dict(ckpt['model'])
        model.to(device)
        if args.compile:
            model = torch.compile(model)  # type: ignore[assignment]
        metrics = per_board_metrics(model, loader, device)
        elapsed = time.perf_counter() - t0

        rows = []
        for i, m in enumerate(metrics):
            row = {
                'k': k,
                'val_idx': i,
                'exact': m['exact'],
                'blank_acc': m['blank_acc'],
                'blank_correct': m['blank_correct'],
                'n_blank': m['n_blank'],
                'n_clue': n_clue[i],
                'rating': ratings[i],
                'source': sources[i],
            }
            rows.append(row)
            all_csv_rows.append(row)

        st = stratify(rows)
        st['wall_sec'] = elapsed
        # Aggregate blank matches micro-avg over blanks (matches train.evaluate)
        total_bc = sum(int(r['blank_correct']) for r in rows)
        total_nb = sum(int(r['n_blank']) for r in rows)
        st['blank_acc_micro'] = total_bc / max(total_nb, 1)
        summary['by_k'][str(k)] = st

        print(
            f"K={k} BLANK_MICRO={st['blank_acc_micro']:.6f} "
            f"EXACT={st['exact_rate']:.6f} n_exact={st['n_exact']} "
            f"blank_unsolved_mean={st['mean_blank_unsolved']:.4f} "
            f"wall={elapsed:.1f}s"
        )
        print(
            f"  solved mean_clues={st['solved']['mean_clues']:.2f} "
            f"mean_rating={st['solved']['mean_rating']:.2f}"
        )
        print(
            f"  unsolved mean_clues={st['unsolved']['mean_clues']:.2f} "
            f"mean_rating={st['unsolved']['mean_rating']:.2f}"
        )

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if args.csv_out:
        out = Path(args.csv_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('w', newline='') as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    'k',
                    'val_idx',
                    'exact',
                    'blank_acc',
                    'blank_correct',
                    'n_blank',
                    'n_clue',
                    'rating',
                    'source',
                ],
            )
            w.writeheader()
            w.writerows(all_csv_rows)
        print(f'CSV={out}')

    if args.json_out:
        outj = Path(args.json_out)
        outj.parent.mkdir(parents=True, exist_ok=True)
        outj.write_text(json.dumps(summary, indent=2) + '\n')
        print(f'JSON={outj}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
