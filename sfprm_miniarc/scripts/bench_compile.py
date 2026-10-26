#!/usr/bin/env python3
"""Timed eager vs torch.compile for SFPRM hot path (``model.step``).

Microbench only: ``torch.no_grad``, static shapes, no BPTT / solver schedule /
vote-augs. Not a train-throughput claim. Desk dims default to paper tree
``d_model=384`` (16GB); override with ``--d-model 512`` for full width.

Paper API (``main``): ``SFPRM.step(z, z_input, eta, ...) -> (z_next, z_tilde, residual)``.
``train.py --compile`` remains a dead flag on this tree (not wired to
``solver_step`` / ``model.step``).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.sfprm_core import DEFAULT_VOCAB, SFPRM  # noqa: E402
from models.spherical_ops import project_to_sphere  # noqa: E402


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _timed_iters(
    fn,
    args: tuple,
    *,
    warmup: int,
    iters: int,
    device: torch.device,
) -> list[float]:
    for _ in range(warmup):
        fn(*args)
    _sync(device)
    times: list[float] = []
    for _ in range(iters):
        _sync(device)
        t0 = time.perf_counter()
        fn(*args)
        _sync(device)
        times.append(time.perf_counter() - t0)
    return times


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d-model", type=int, default=384, help="desk 16GB default; paper Mini-ARC uses 512")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=128, help="latent sequence length (incl. puzzle emb if any)")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA/HIP not available")
        return 1

    device = torch.device(args.device)
    print(
        f"torch={torch.__version__} hip={getattr(torch.version, 'hip', None)} "
        f"device={torch.cuda.get_device_name(0) if device.type == 'cuda' else device}"
    )
    print(
        f"dims: d_model={args.d_model} batch={args.batch} seq={args.seq_len} "
        f"warmup={args.warmup} iters={args.iters} (microbench; not train)"
    )

    model = SFPRM(vocab_size=DEFAULT_VOCAB, d_model=args.d_model, max_seq_len=args.seq_len).to(device)
    model.eval()
    # Match desk amp path used in train (bf16 on ROCm).
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = model.to(dtype=dtype)

    z = project_to_sphere(torch.randn(args.batch, args.seq_len, args.d_model, device=device, dtype=dtype))
    z_input = project_to_sphere(torch.randn(args.batch, args.seq_len, args.d_model, device=device, dtype=dtype))
    # Paper step takes per-sample η in (0, 1]; schedule lives in solver_step — fixed here.
    eta = torch.ones(args.batch, device=device, dtype=torch.float32)

    def eager_step(za: torch.Tensor, zi: torch.Tensor, e: torch.Tensor) -> tuple[torch.Tensor, ...]:
        with torch.no_grad():
            return model.step(za, zi, e)

    compiled_step = torch.compile(model.step, mode="reduce-overhead")

    def compile_step(za: torch.Tensor, zi: torch.Tensor, e: torch.Tensor) -> tuple[torch.Tensor, ...]:
        with torch.no_grad():
            return compiled_step(za, zi, e)

    call_args = (z, z_input, eta)

    # First compile call can be long; time it separately then steady-state.
    _sync(device)
    t_compile0 = time.perf_counter()
    _ = compile_step(*call_args)
    _sync(device)
    first_compile_s = time.perf_counter() - t_compile0
    print(f"first torch.compile call: {first_compile_s * 1000:.1f} ms")

    eager_t = _timed_iters(eager_step, call_args, warmup=args.warmup, iters=args.iters, device=device)
    compile_t = _timed_iters(compile_step, call_args, warmup=args.warmup, iters=args.iters, device=device)

    def summarize(name: str, times: list[float]) -> tuple[float, float]:
        ms = [t * 1000.0 for t in times]
        mean_ms = statistics.fmean(ms)
        med_ms = statistics.median(ms)
        it_s = 1000.0 / mean_ms
        print(
            f"{name}: mean={mean_ms:.3f} ms/iter  median={med_ms:.3f} ms/iter  "
            f"({it_s:.2f} it/s)  min={min(ms):.3f} max={max(ms):.3f}"
        )
        return mean_ms, it_s

    e_ms, e_its = summarize("eager  ", eager_t)
    c_ms, c_its = summarize("compile", compile_t)
    speedup = e_ms / c_ms if c_ms > 0 else float("nan")
    print(f"speedup (eager_ms / compile_ms): {speedup:.3f}x")
    print(f"throughput ratio (compile_it/s / eager_it/s): {c_its / e_its:.3f}x")

    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(
            f"VRAM after bench: used={(total - free) / 1e9:.2f} GB / "
            f"total={total / 1e9:.2f} GB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
