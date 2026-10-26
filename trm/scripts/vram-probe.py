#!/usr/bin/env python3
"""Find the largest TRM training batch size that fits on the current GPU (synthetic batch)."""

from __future__ import annotations

import argparse
import sys

import torch

from trm.config import ModelConfig
from trm.model.trm_mlp import TRMMLP
from trm.training.loss import masked_cross_entropy


def _synthetic_batch(batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    puzzle = torch.randint(0, 10, (batch_size, 81), device=device)
    solution = torch.randint(1, 10, (batch_size, 81), device=device)
    mask = (puzzle == 0).float()
    return {'puzzle': puzzle, 'solution': solution, 'mask': mask}


def _fits(model: TRMMLP, batch_size: int, device: torch.device) -> bool:
    torch.cuda.empty_cache()
    try:
        batch = _synthetic_batch(batch_size, device)
        model.train()
        _, step_logits = model.forward_supervised(batch['puzzle'])
        loss = sum(masked_cross_entropy(logits, batch['solution'], batch['mask']) for logits in step_logits)
        loss.backward()
        return True
    except torch.cuda.OutOfMemoryError:
        return False
    finally:
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description='Probe max TRM train batch size on this GPU')
    parser.add_argument('--min-batch', type=int, default=1)
    parser.add_argument('--max-batch', type=int, default=2048)
    parser.add_argument('--tiny', action='store_true', help='Use tiny model config (smoke / CPU)')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print('No CUDA device; cannot probe VRAM.', file=sys.stderr)
        sys.exit(1)

    device = torch.device('cuda')
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024**3)
    print(f'Device: {props.name} ({vram_gb:.1f} GB)')

    if args.tiny:
        from trm.config import tiny_model_config

        model_cfg = tiny_model_config()
    else:
        model_cfg = ModelConfig()

    model = TRMMLP(model_cfg).to(device)
    lo, hi = args.min_batch, args.max_batch
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if _fits(model, mid, device):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    print(f'Max batch size (synthetic train step): {best}')


if __name__ == '__main__':
    main()
