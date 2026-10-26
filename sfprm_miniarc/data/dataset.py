"""Mini-ARC ingestion, D₄ + color-perm augmentation, in-device tensor cache.

Sequence packing (gap fill; documented here and in README):
  PAD_ID = 10 (vocab_size=11). Colors are tokens 0–9.
  For each sample we concatenate up to ``max_demos`` demonstration pairs as
  ``flatten(input) || flatten(output)``, then the query input grid.
  ``query_mask`` selects the 25 query-input positions; the supervision target
  at those positions is the flattened query output (predict transform in place).
  With max_seq_len=128 this fits 2 demos + query (125 tokens), then PAD to 128.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

GRID = 5
CELLS = GRID * GRID
PAD_ID = 10
NUM_COLORS = 10  # 0–9; 0 is background (not permuted)


@dataclass
class MiniARCCache:
    """In-device cached tensors for training or evaluation."""

    tokens: torch.Tensor  # [N, L] long
    targets: torch.Tensor  # [N, L] long (valid under query_mask)
    query_mask: torch.Tensor  # [N, L] bool
    task_ids: list[str]

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

    def batch(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.tokens[indices], self.targets[indices], self.query_mask[indices]


def _as_grid(arr) -> np.ndarray:
    g = np.asarray(arr, dtype=np.int64)
    if g.shape != (GRID, GRID):
        raise ValueError(f"expected 5x5 grid, got {g.shape}")
    return g


def load_task_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def list_task_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(
            f"No Mini-ARC JSON in {data_dir}. Run: python data/download_miniarc.py"
        )
    return files


def dihedral(grid: np.ndarray, k: int) -> np.ndarray:
    """Apply one of 8 D₄ transforms (k in 0..7): rotations then optional reflect."""
    out = np.rot90(grid, k % 4)
    if k >= 4:
        out = np.fliplr(out)
    return np.ascontiguousarray(out)


def apply_color_perm(grid: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """perm maps old color -> new; index 0 fixed as background."""
    return perm[grid]


def random_color_perm(rng: random.Random) -> np.ndarray:
    colors = list(range(1, NUM_COLORS))
    rng.shuffle(colors)
    perm = np.arange(NUM_COLORS, dtype=np.int64)
    perm[1:] = np.asarray(colors, dtype=np.int64)
    return perm


def pack_sequence(
    demos: list[tuple[np.ndarray, np.ndarray]],
    query_in: np.ndarray,
    query_out: np.ndarray,
    max_seq_len: int = 128,
    max_demos: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack demos + query into fixed-length token / target / mask arrays."""
    # Reserve CELLS for query_in; each demo costs 2 * CELLS.
    room = max_seq_len - CELLS
    fit = max(0, room // (2 * CELLS))
    if max_demos is not None:
        fit = min(fit, max_demos)
    demos = demos[:fit]

    tokens: list[int] = []
    for inp, out in demos:
        tokens.extend(inp.reshape(-1).tolist())
        tokens.extend(out.reshape(-1).tolist())
    q_start = len(tokens)
    tokens.extend(query_in.reshape(-1).tolist())
    q_end = len(tokens)

    if len(tokens) > max_seq_len:
        raise ValueError(f"packed length {len(tokens)} > max_seq_len={max_seq_len}")

    pad_n = max_seq_len - len(tokens)
    tokens.extend([PAD_ID] * pad_n)

    targets = np.full(max_seq_len, PAD_ID, dtype=np.int64)
    targets[q_start:q_end] = query_out.reshape(-1)

    mask = np.zeros(max_seq_len, dtype=bool)
    mask[q_start:q_end] = True

    return (
        np.asarray(tokens, dtype=np.int64),
        targets,
        mask,
    )


def _transform_task(
    task: dict,
    d4_k: int,
    color_perm: np.ndarray,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    demos = []
    for pair in task["train"]:
        inp = apply_color_perm(dihedral(_as_grid(pair["input"]), d4_k), color_perm)
        out = apply_color_perm(dihedral(_as_grid(pair["output"]), d4_k), color_perm)
        demos.append((inp, out))
    test = task["test"][0]
    q_in = apply_color_perm(dihedral(_as_grid(test["input"]), d4_k), color_perm)
    q_out = apply_color_perm(dihedral(_as_grid(test["output"]), d4_k), color_perm)
    return demos, q_in, q_out


def build_cache(
    data_dir: Path,
    *,
    device: torch.device,
    max_seq_len: int = 128,
    augment: bool = True,
    n_color_perms: int = 8,
    seed: int = 0,
    max_tasks: int | None = None,
) -> MiniARCCache:
    """
    Load Mini-ARC tasks and cache packed tensors on ``device``.

    When ``augment`` is True: 8 D₄ × ``n_color_perms`` random color maps per task.
    When False: one unaugmented sample per task (eval).
    """
    files = list_task_files(data_dir)
    if max_tasks is not None:
        files = files[:max_tasks]

    rng = random.Random(seed)
    identity = np.arange(NUM_COLORS, dtype=np.int64)

    all_tokens: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    task_ids: list[str] = []

    for path in files:
        task = load_task_json(path)
        tid = path.stem
        if augment:
            perms = [identity] + [random_color_perm(rng) for _ in range(n_color_perms - 1)]
            d4_ks = range(8)
        else:
            perms = [identity]
            d4_ks = [0]

        for d4_k in d4_ks:
            for perm in perms:
                demos, q_in, q_out = _transform_task(task, d4_k, perm)
                tok, tgt, mask = pack_sequence(demos, q_in, q_out, max_seq_len=max_seq_len)
                all_tokens.append(tok)
                all_targets.append(tgt)
                all_masks.append(mask)
                task_ids.append(tid)

    tokens = torch.tensor(np.stack(all_tokens), dtype=torch.long, device=device)
    targets = torch.tensor(np.stack(all_targets), dtype=torch.long, device=device)
    query_mask = torch.tensor(np.stack(all_masks), dtype=torch.bool, device=device)
    return MiniARCCache(tokens=tokens, targets=targets, query_mask=query_mask, task_ids=task_ids)


def make_train_indices(n: int, batch_size: int, step: int, generator: torch.Generator) -> torch.Tensor:
    """Sample a batch of indices with replacement (step-based training)."""
    return torch.randint(0, n, (batch_size,), generator=generator)
