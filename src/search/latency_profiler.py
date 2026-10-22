"""
Latency and VRAM Profiler module.

Measures direct wall-clock inference latency (ms/frame) on 1080p tensors (1920x1080).
Uses PyTorch CUDA events for microsecond-accurate GPU execution timing,
with seamless fallback to synchronized high-resolution CPU performance counters.
"""

from typing import Dict, Any, Tuple
import time
import torch
import torch.nn as nn


class LatencyProfiler:
    """
    Profiles end-to-end forward inference latency and peak memory usage.
    
    Args:
        input_resolution: (H, W) of benchmarking tensor (default: (1080, 1920)).
        warmup_iterations: Number of warmup passes to prime CUDA kernels and cache.
        benchmark_iterations: Number of timed evaluation passes.
    """
    def __init__(
        self,
        input_resolution: Tuple[int, int] = (1080, 1920),
        warmup_iterations: int = 5,
        benchmark_iterations: int = 20
    ):
        self.input_resolution = input_resolution
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations

    @torch.no_grad()
    def profile(
        self,
        model: nn.Module,
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ) -> Dict[str, float]:
        """
        Executes profiling on a single-image tensor of resolution (1, 3, H, W).
        
        Returns:
            Dictionary with 'latency_ms', 'fps', and 'peak_vram_mb'.
        """
        model.eval()
        model.to(device=device, dtype=dtype)

        H, W = self.input_resolution
        dummy_input = torch.randn(1, 3, H, W, device=device, dtype=dtype)

        # 1. Warmup passes
        for _ in range(self.warmup_iterations):
            _ = model(dummy_input)

        use_cuda = device.type == "cuda" and torch.cuda.is_available()

        if use_cuda:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            for _ in range(self.benchmark_iterations):
                _ = model(dummy_input)
            end_event.record()

            torch.cuda.synchronize(device)
            total_time_ms = start_event.elapsed_time(end_event)
            latency_ms = total_time_ms / self.benchmark_iterations
            peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        else:
            # CPU fallback timing
            t0 = time.perf_counter()
            for _ in range(self.benchmark_iterations):
                _ = model(dummy_input)
            t1 = time.perf_counter()
            latency_ms = ((t1 - t0) / self.benchmark_iterations) * 1000.0
            peak_vram_mb = 0.0

        fps = 1000.0 / (latency_ms + 1e-8)

        return {
            "latency_ms": float(latency_ms),
            "fps": float(fps),
            "peak_vram_mb": float(peak_vram_mb)
        }
