"""
CLI Entrypoint for Stage 1: Spatial Parameter Search (NSGA-II).

Supports loading ImageNet-100 directly from Hugging Face (`clane9/imagenet-100`)
with optional streaming mode for zero-wait iteration.

Usage:
    python run_search_stage1.py --use-hf --streaming --dry-run
    python run_search_stage1.py --use-hf --hf-dataset clane9/imagenet-100
"""

import argparse
import json
import os
import yaml
import torch

from src.data import get_imagenet100_loaders
from src.search import NSGAIIEngine, ProxyEvaluator


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Spatial Parameter Evolutionary Search")
    parser.add_argument("--config", type=str, default="configs/search_stage1_spatial.yaml", help="Path to config file")
    parser.add_argument("--use-hf", action="store_true", default=True, help="Load dataset directly from Hugging Face")
    parser.add_argument("--no-hf", dest="use_hf", action="store_false", help="Disable Hugging Face dataset")
    parser.add_argument("--hf-dataset", type=str, default="clane9/imagenet-100", help="Hugging Face dataset identifier")
    parser.add_argument("--streaming", action="store_true", help="Stream dataset directly from Hugging Face hub")
    parser.add_argument("--data-dir", type=str, default=None, help="Local ImageNet-100 path fallback")
    parser.add_argument("--dry-run", action="store_true", help="Quick verification run with small pop/gens")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic mock data")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    pop_size = 4 if args.dry_run else cfg["search"]["population_size"]
    generations = 2 if args.dry_run else cfg["search"]["generations"]
    proxy_steps = 2 if args.dry_run else cfg["proxy_adaptation"]["num_steps"]

    print("=" * 70)
    print("STAGE 1: SPATIAL PARAMETER SEARCH (NSGA-II)")
    print(f"Population Size: {pop_size} | Generations: {generations} | Proxy Steps: {proxy_steps}")
    print(f"Hugging Face: {args.use_hf} ('{args.hf_dataset}', streaming={args.streaming})")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'} | Dry-Run: {args.dry_run}")
    print("=" * 70)

    # 1. Dataset Loaders from Hugging Face or fallback
    batch_size = 16 if args.dry_run else cfg["proxy_adaptation"]["batch_size"]
    train_loader, val_loader = get_imagenet100_loaders(
        data_dir=args.data_dir,
        use_hf=args.use_hf and not args.synthetic,
        hf_dataset_name=args.hf_dataset,
        streaming=args.streaming,
        batch_size=batch_size,
        synthetic=args.synthetic or (args.dry_run and not args.use_hf)
    )

    # Pre-fetch mini-batches for fast proxy evaluation
    train_batches = []
    for i, (x, y) in enumerate(train_loader):
        train_batches.append((x, y))
        if len(train_batches) >= proxy_steps:
            break

    val_batches = []
    for i, (x, y) in enumerate(val_loader):
        val_batches.append((x, y))
        if len(val_batches) >= 2:
            break

    # 2. Proxy Evaluator
    evaluator = ProxyEvaluator(
        num_classes=100,
        num_steps=proxy_steps,
        lr=cfg["proxy_adaptation"]["lr"],
        latency_benchmark=not args.dry_run
    )

    bounds = {
        "r_rel": tuple(cfg["bounds"]["r_rel"]),
        "r_fovea": tuple(cfg["bounds"]["r_fovea"]),
        "r_norm": tuple(cfg["bounds"]["r_norm"]),
        "alpha_fovea": tuple(cfg["bounds"]["alpha_fovea"]),
        "gamma_periph": tuple(cfg["bounds"]["gamma_periph"]),
    }

    def candidate_evaluator(genes):
        return evaluator.evaluate_candidate(genes, train_batches, val_batches)

    # 3. Evolutionary Search Engine
    engine = NSGAIIEngine(
        bounds=bounds,
        evaluator_fn=candidate_evaluator,
        population_size=pop_size,
        generations=generations,
        crossover_prob=cfg["search"]["crossover_prob"],
        mutation_prob=cfg["search"]["mutation_prob"]
    )

    print("\n[INFO] Initializing population across spatial bounds...")
    engine.initialize_population()

    for gen in range(1, generations + 1):
        pareto = engine.step_generation()
        best_acc = max(ind.objectives[0] for ind in pareto)
        best_lat = min(-ind.objectives[1] for ind in pareto)
        print(f"Generation {gen:02d}/{generations:02d} | Pareto Front Size: {len(pareto)} | Best Proxy Acc: {best_acc:.2f}% | Best Latency: {best_lat:.2f} ms")

    # 4. Save Pareto Results
    final_pareto = engine.get_pareto_front()
    results = []
    for ind in final_pareto:
        results.append({
            "genes": ind.genes,
            "objectives": {
                "top1_accuracy": ind.objectives[0],
                "latency_1080p_ms": -ind.objectives[1],
                "active_pixel_ratio": -ind.objectives[2]
            }
        })

    pareto_file = os.path.join(args.output_dir, "stage1_pareto_front.json")
    best_file = os.path.join(args.output_dir, "stage1_best_spatial.json")

    with open(pareto_file, "w") as f:
        json.dump(results, f, indent=2)

    best_ind = max(final_pareto, key=lambda ind: ind.objectives[0])
    with open(best_file, "w") as f:
        json.dump(best_ind.genes, f, indent=2)

    print("\n" + "=" * 70)
    print(f"[SUCCESS] Stage 1 Search Completed! Optimal Spatial Parameters Saved to: {best_file}")
    print(f"Optimal Spatial Parameters: {json.dumps(best_ind.genes, indent=2)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
