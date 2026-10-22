"""
CLI Entrypoint for Unified 50-Epoch Fine-Tuning Pipeline.

Supports training directly on Hugging Face ImageNet-100 (`clane9/imagenet-100`)
with optional streaming or local caching.

Usage:
    python run_train.py --use-hf --dry-run
    python run_train.py --use-hf --hf-dataset clane9/imagenet-100 --config configs/train_50ep_finetune.yaml
"""

import argparse
import json
import os
import yaml
import torch

from src.models import BioInspiredVisionPipeline
from src.data import get_imagenet100_loaders
from src.training import UnifiedTrainer


def main():
    parser = argparse.ArgumentParser(description="Unified 50-Epoch Fine-Tuning Execution")
    parser.add_argument("--config", type=str, default="configs/train_50ep_finetune.yaml")
    parser.add_argument("--use-hf", action="store_true", default=True, help="Load dataset directly from Hugging Face")
    parser.add_argument("--no-hf", dest="use_hf", action="store_false", help="Disable Hugging Face dataset")
    parser.add_argument("--hf-dataset", type=str, default="clane9/imagenet-100")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--spatial-checkpoint", type=str, default="checkpoints/stage1_best_spatial.json")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Quick 2-epoch test run")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load optimal spatial hyperparameters
    spatial_genes = {"r_rel": 0.015, "r_fovea": 0.15, "r_norm": 0.035, "alpha_fovea": 0.5, "gamma_periph": 1.5}
    if os.path.exists(args.spatial_checkpoint):
        try:
            with open(args.spatial_checkpoint, "r") as f:
                spatial_genes = json.load(f)
            print(f"[INFO] Loaded optimal spatial parameters from: {args.spatial_checkpoint}")
        except Exception:
            pass

    epochs = 2 if args.dry_run else cfg["training"]["epochs"]
    batch_size = 16 if args.dry_run else cfg["training"]["batch_size"]

    print("=" * 70)
    print("UNIFIED FINE-TUNING EXECUTION PIPELINE")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Hugging Face: {args.use_hf} ('{args.hf_dataset}', streaming={args.streaming})")
    print("=" * 70)

    # 2. Data Loaders
    train_loader, val_loader = get_imagenet100_loaders(
        data_dir=args.data_dir,
        use_hf=args.use_hf and not args.synthetic,
        hf_dataset_name=args.hf_dataset,
        streaming=args.streaming,
        batch_size=batch_size,
        synthetic=args.synthetic or (args.dry_run and not args.use_hf)
    )

    # 3. Model Pipeline
    model = BioInspiredVisionPipeline(
        backbone_name="resnet18",
        num_classes=cfg["data"]["num_classes"],
        pretrained_backbone=False,
        r_rel=spatial_genes.get("r_rel", 0.015),
        r_fovea=spatial_genes.get("r_fovea", 0.15),
        r_norm=spatial_genes.get("r_norm", 0.035)
    )

    # 4. Unified Trainer
    trainer = UnifiedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        base_max_lr=cfg["scheduler"]["base_max_lr"],
        checkpoint_dir=args.output_dir
    )

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_metrics = trainer.train_epoch(epoch)
        val_metrics = trainer.validate()

        is_best = val_metrics["val_top1"] > best_val_acc
        if is_best:
            best_val_acc = val_metrics["val_top1"]
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_bio_model.pt"))

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] | "
            f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['top1']:.2f}% | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | Val Acc: {val_metrics['val_top1']:.2f}% "
            f"{'(*)' if is_best else ''}"
        )

    print("\n" + "=" * 70)
    print(f"[SUCCESS] Training complete! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {os.path.join(args.output_dir, 'best_bio_model.pt')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
