"""Mini-ARC ingestion, D₄ + color-perm augmentation, in-device tensor cache.

Paper Mini-ARC (App. B / Table 3): one 5×5 input grid → predict the 5×5 output
(``seq_len=25`` cell tokens + puzzle emb). Train on the task's demonstration
pairs; eval on the test pair. Matches "528 demo pairs × augs" / "a puzzle is
25 cell tokens".

PAD_ID = 10; SEP_ID = 11 reserved (vocab_size=12). Colors are tokens 0–9.

Note: pre-seq25 desk runs (EX-33) used packed demos+query ≤128. That path is
gone; those checkpoints are unsupported — retrain with this formulation.
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
PAPER_SEQ_LEN = CELLS  # paper Mini-ARC sequence length
PAD_ID = 10
SEP_ID = 11  # paper Mini-ARC vocab=12 (colors 0–9 + PAD + SEP)
NUM_COLORS = 10  # 0–9; 0 is background (not permuted)
VOCAB_SIZE = 12


@dataclass
class MiniARCCache:
    """In-device cached tensors for training or evaluation."""

    tokens: torch.Tensor  # [N, L] long
    targets: torch.Tensor  # [N, L] long (valid under query_mask)
    query_mask: torch.Tensor  # [N, L] bool
    puzzle_ids: torch.Tensor  # [N] long — integer task index for puzzle-emb
    task_ids: list[str]

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

    def batch(
        self, indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.tokens[indices],
            self.targets[indices],
            self.query_mask[indices],
            self.puzzle_ids[indices],
        )


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


def pack_demo_pair(
    grid_in: np.ndarray,
    grid_out: np.ndarray,
    max_seq_len: int = PAPER_SEQ_LEN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Paper seq-25: tokens = flatten(input), targets = flatten(output)."""
    if max_seq_len < CELLS:
        raise ValueError(f"max_seq_len={max_seq_len} < CELLS={CELLS}")
    tokens = np.full(max_seq_len, PAD_ID, dtype=np.int64)
    targets = np.full(max_seq_len, PAD_ID, dtype=np.int64)
    mask = np.zeros(max_seq_len, dtype=bool)
    tokens[:CELLS] = grid_in.reshape(-1)
    targets[:CELLS] = grid_out.reshape(-1)
    mask[:CELLS] = True
    return tokens, targets, mask


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
    max_seq_len: int = PAPER_SEQ_LEN,
    augment: bool = True,
    n_color_perms: int = 8,
    seed: int = 0,
    max_tasks: int | None = None,
) -> MiniARCCache:
    """
    Load Mini-ARC tasks and cache tensors on ``device``.

    When ``augment`` is True: 8 D₄ × ``n_color_perms`` random color maps per task;
    emit one seq-25 sample per demonstration pair × aug.
    When False: one unaugmented test I→O sample per task (eval).
    """
    files = list_task_files(data_dir)
    if max_tasks is not None:
        files = files[:max_tasks]

    rng = random.Random(seed)
    identity = np.arange(NUM_COLORS, dtype=np.int64)

    all_tokens: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    all_puzzle: list[int] = []
    task_ids: list[str] = []
    name_to_id: dict[str, int] = {}

    for path in files:
        task = load_task_json(path)
        tid = path.stem
        if tid not in name_to_id:
            name_to_id[tid] = len(name_to_id)
        pid = name_to_id[tid]
        if augment:
            perms = [identity] + [random_color_perm(rng) for _ in range(n_color_perms - 1)]
            d4_ks = range(8)
        else:
            perms = [identity]
            d4_ks = [0]

        for d4_k in d4_ks:
            for perm in perms:
                demos, q_in, q_out = _transform_task(task, d4_k, perm)
                # Paper: train on demonstration pairs; eval on the test pair.
                pairs = demos if augment else [(q_in, q_out)]
                for inp, out in pairs:
                    tok, tgt, mask = pack_demo_pair(inp, out, max_seq_len=max_seq_len)
                    all_tokens.append(tok)
                    all_targets.append(tgt)
                    all_masks.append(mask)
                    all_puzzle.append(pid)
                    task_ids.append(tid)

    tokens = torch.tensor(np.stack(all_tokens), dtype=torch.long, device=device)
    targets = torch.tensor(np.stack(all_targets), dtype=torch.long, device=device)
    query_mask = torch.tensor(np.stack(all_masks), dtype=torch.bool, device=device)
    puzzle_ids = torch.tensor(all_puzzle, dtype=torch.long, device=device)
    return MiniARCCache(
        tokens=tokens,
        targets=targets,
        query_mask=query_mask,
        puzzle_ids=puzzle_ids,
        task_ids=task_ids,
    )


def make_train_indices(n: int, batch_size: int, step: int, generator: torch.Generator) -> torch.Tensor:
    """Sample a batch of indices with replacement (step-based training)."""
    return torch.randint(0, n, (batch_size,), generator=generator)
