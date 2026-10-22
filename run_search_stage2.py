"""
CLI Entrypoint for Stage 2: Temporal Parameter Search.

Usage:
    python run_search_stage2.py --config configs/search_stage2_temporal.yaml --dry-run
"""

import argparse
import json
import os
import yaml
import torch

from src.models import BioInspiredVisionPipeline
from src.video import TemporalFeatureGater
from src.data import get_video_clip_loaders
from src.search import NSGAIIEngine


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Temporal Parameter Search on Video Sequences")
    parser.add_argument("--config", type=str, default="configs/search_stage2_temporal.yaml")
    parser.add_argument("--spatial-checkpoint", type=str, default="checkpoints/stage1_best_spatial.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load optimal spatial genes from Stage 1 if present
    spatial_genes = {"r_rel": 0.015, "r_fovea": 0.15, "r_norm": 0.035, "alpha_fovea": 0.5, "gamma_periph": 1.5}
    if os.path.exists(args.spatial_checkpoint):
        try:
            with open(args.spatial_checkpoint, "r") as f:
                spatial_genes = json.load(f)
            print(f"[INFO] Loaded optimal spatial parameters from: {args.spatial_checkpoint}")
        except Exception:
            pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = BioInspiredVisionPipeline(
        backbone_name="resnet18",
        num_classes=11,  # CamVid 11 semantic classes
        r_rel=spatial_genes.get("r_rel", 0.015),
        r_fovea=spatial_genes.get("r_fovea", 0.15),
        r_norm=spatial_genes.get("r_norm", 0.035)
    ).to(device)
    pipeline.eval()

    # 2. Video DataLoader
    video_loader = get_video_clip_loaders(
        batch_size=8 if args.dry_run else cfg["video_dataset"]["batch_size"],
        clip_len=cfg["video_dataset"]["clip_length"],
        synthetic=True
    )
    video_clips = [clip for i, (clip, _) in enumerate(video_loader) if i < (2 if args.dry_run else 8)]

    # 3. Evaluator for temporal parameters (delta_threshold, state_reuse_threshold)
    def temporal_evaluator(genes):
        tau_reuse = genes.get("state_reuse_threshold", 0.10)
        gater = TemporalFeatureGater(state_reuse_threshold=tau_reuse)

        total_bypasses = 0
        total_frames = 0
        with torch.no_grad():
            for clip in video_clips:
                gater.reset()
                # clip shape: (B, T, 3, H, W)
                B, T, C, H, W = clip.shape
                for t in range(T):
                    frame = clip[0:1, t].to(device)
                    _, bypassed, _ = gater.process_frame(pipeline, frame)
                    if bypassed:
                        total_bypasses += 1
                    total_frames += 1

        bypass_pct = (total_bypasses / max(1, total_frames)) * 100.0
        # Multi-objective: [Bypass Rate, -Reuse Threshold]
        return [float(bypass_pct), float(-tau_reuse)]

    bounds = {
        "delta_threshold": tuple(cfg["bounds"]["delta_threshold"]),
        "state_reuse_threshold": tuple(cfg["bounds"]["state_reuse_threshold"]),
    }

    pop_size = 4 if args.dry_run else cfg["search"]["population_size"]
    generations = 2 if args.dry_run else cfg["search"]["generations"]

    print("=" * 70)
    print("STAGE 2: TEMPORAL PARAMETER SEARCH")
    print(f"Population Size: {pop_size} | Generations: {generations}")
    print("=" * 70)

    engine = NSGAIIEngine(
        bounds=bounds,
        evaluator_fn=temporal_evaluator,
        population_size=pop_size,
        generations=generations
    )
    engine.initialize_population()

    for gen in range(1, generations + 1):
        pareto = engine.step_generation()
        best_bypass = max(ind.objectives[0] for ind in pareto)
        print(f"Generation {gen:02d}/{generations:02d} | Pareto Front Size: {len(pareto)} | Best Bypass Rate: {best_bypass:.1f}%")

    final_pareto = engine.get_pareto_front()
    best_temporal = max(final_pareto, key=lambda ind: ind.objectives[0])

    best_file = os.path.join(args.output_dir, "stage2_best_temporal.json")
    with open(best_file, "w") as f:
        json.dump(best_temporal.genes, f, indent=2)

    print("\n" + "=" * 70)
    print(f"[SUCCESS] Stage 2 Temporal Search Completed! Saved to: {best_file}")
    print(f"Optimal Temporal Parameters: {json.dumps(best_temporal.genes, indent=2)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
