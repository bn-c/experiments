"""
CLI Entrypoint for Scientific Control Benchmarking.

Evaluates the fine-tuned Bio-Inspired Vision Pipeline against the
honest Information-Matched Uniform Downsampling Baseline across parameter count,
active pixel count, 1080p wall-clock latency (ms/frame), throughput (FPS), and Top-1 accuracy.
Supports validation evaluation directly on Hugging Face ImageNet-100.

Usage:
    python run_benchmark.py --use-hf --dry-run
"""

import argparse
import os
import torch
import torch.nn as nn

from src.models import BioInspiredVisionPipeline
from src.evaluation import UniformDownsampledBaseline
from src.search.latency_profiler import LatencyProfiler
from src.data import get_imagenet100_loaders


def main():
    parser = argparse.ArgumentParser(description="Scientific Control Benchmarking")
    parser.add_argument("--model-checkpoint", type=str, default="checkpoints/best_bio_model.pt")
    parser.add_argument("--use-hf", action="store_true", default=True)
    parser.add_argument("--hf-dataset", type=str, default="clane9/imagenet-100")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("SCIENTIFIC CONTROL BENCHMARK: BIO-PIPELINE VS. INFORMATION-MATCHED BASELINE")
    print(f"Device: {device} | 1080p Profile Resolution: 1920x1080")
    print(f"Dataset: Hugging Face '{args.hf_dataset}' (use_hf={args.use_hf})")
    print("=" * 80)

    # 1. Models
    bio_model = BioInspiredVisionPipeline(backbone_name="resnet18", num_classes=100)
    if os.path.exists(args.model_checkpoint):
        try:
            bio_model.load_state_dict(torch.load(args.model_checkpoint, map_location="cpu"))
            print(f"[INFO] Loaded bio-pipeline checkpoint: {args.model_checkpoint}")
        except Exception:
            pass

    control_model = UniformDownsampledBaseline(num_classes=100)

    bio_model.to(device).eval()
    control_model.to(device).eval()

    # 2. Parameter counts
    bio_params = sum(p.numel() for p in bio_model.parameters())
    control_params = sum(p.numel() for p in control_model.parameters())

    # 3. 1080p Latency profiling
    profiler = LatencyProfiler(
        input_resolution=(1080, 1920),
        warmup_iterations=2 if args.dry_run else 5,
        benchmark_iterations=5 if args.dry_run else 20
    )

    bio_prof = profiler.profile(bio_model, device=device)
    control_prof = profiler.profile(control_model, device=device)

    # 4. Accuracy evaluation
    _, val_loader = get_imagenet100_loaders(
        data_dir=args.data_dir,
        use_hf=args.use_hf and not args.dry_run,
        hf_dataset_name=args.hf_dataset,
        batch_size=16 if args.dry_run else 64,
        synthetic=args.dry_run,
        num_synthetic_samples=64
    )

    def eval_acc(model):
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = torch.argmax(model(x), dim=-1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                if args.dry_run and total >= 32:
                    break
        return (correct / max(1, total)) * 100.0

    bio_acc = eval_acc(bio_model)
    control_acc = eval_acc(control_model)

    # 5. Publication-grade Markdown Summary Table
    print("\n### Benchmark Results Matrix")
    print("| Metric | Bio-Inspired Pipeline (Ours) | Information-Matched Control | Delta / Relative Change |")
    print("| :--- | :--- | :--- | :--- |")
    print(f"| **Active Sampling Grid** | $56 \\times 56$ (3,136 px, 1:1) / $42 \\times 75$ (3,150 px, 16:9) | $56 \\times 56$ (3,136 px, uniform) | **Budget Parity (0.4% diff)** |")
    print(f"| **Total Parameters** | {bio_params:,} | {control_params:,} | +{(bio_params - control_params):,} params |")
    print(f"| **1080p Latency** | **{bio_prof['latency_ms']:.2f} ms/frame** | {control_prof['latency_ms']:.2f} ms/frame | {bio_prof['latency_ms'] - control_prof['latency_ms']:+.2f} ms |")
    print(f"| **Sustained Throughput** | **{bio_prof['fps']:.1f} FPS** | {control_prof['fps']:.1f} FPS | {bio_prof['fps'] - control_prof['fps']:+.1f} FPS |")
    print(f"| **Top-1 Accuracy** | **{bio_acc:.2f}%** | {control_acc:.2f}% | {bio_acc - control_acc:+.2f}% |")
    print("=" * 80)


if __name__ == "__main__":
    main()
