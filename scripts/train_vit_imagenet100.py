#!/usr/bin/env python
"""
Train a small ViT on the DRF biomimetic front-end's 12x64x64 feature output,
using the ImageNet-100 subset from Hugging Face (`clane9/imagenet-100`).

Example:
    python scripts/train_vit_imagenet100.py --epochs 30 --batch-size 128
"""

import argparse
import os
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from drf import DRFViT


def _to_tensor(img: Image.Image) -> torch.Tensor:
    arr = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    return arr.view(img.size[1], img.size[0], 3).permute(2, 0, 1).float() / 255.0


def _random_resized_crop(img: Image.Image, size: int, scale=(0.65, 1.0)) -> Image.Image:
    w, h = img.size
    area = w * h
    for _ in range(10):
        target_area = random.uniform(*scale) * area
        aspect = random.uniform(3 / 4, 4 / 3)
        cw = int(round((target_area * aspect) ** 0.5))
        ch = int(round((target_area / aspect) ** 0.5))
        if cw <= w and ch <= h:
            x0 = random.randint(0, w - cw)
            y0 = random.randint(0, h - ch)
            return img.crop((x0, y0, x0 + cw, y0 + ch)).resize((size, size), Image.BILINEAR)
    # fallback: center crop to the shorter side, then resize
    s = min(w, h)
    x0, y0 = (w - s) // 2, (h - s) // 2
    return img.crop((x0, y0, x0 + s, y0 + s)).resize((size, size), Image.BILINEAR)


def _center_crop_resize(img: Image.Image, size: int, resize_to: int) -> Image.Image:
    w, h = img.size
    scale = resize_to / min(w, h)
    img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)
    w, h = img.size
    x0, y0 = (w - size) // 2, (h - size) // 2
    return img.crop((x0, y0, x0 + size, y0 + size))


def build_transforms(image_size: int) -> tuple:
    def train_tf(img: Image.Image) -> torch.Tensor:
        img = _random_resized_crop(img, image_size)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return _to_tensor(img)

    def eval_tf(img: Image.Image) -> torch.Tensor:
        img = _center_crop_resize(img, image_size, int(image_size * 1.15))
        return _to_tensor(img)

    return train_tf, eval_tf


def make_collate(transform):
    def collate(batch):
        images = torch.stack([transform(ex["image"].convert("RGB")) for ex in batch])
        labels = torch.tensor([ex["label"] for ex in batch], dtype=torch.long)
        return images, labels
    return collate


def run_epoch(model, loader, device, epoch=None, epochs=None, optimizer=None, scaler=None, grad_accum_steps=1):
    is_train = optimizer is not None
    model.train(is_train)
    criterion = nn.CrossEntropyLoss()

    desc = f"epoch {epoch:03d}/{epochs} [{'train' if is_train else 'val'}]" if epoch else ("train" if is_train else "val")
    pbar = tqdm(loader, desc=desc, leave=False)

    total_loss, total_correct, total_n = 0.0, 0, 0
    if is_train:
        optimizer.zero_grad(set_to_none=True)

    for step, (images, labels) in enumerate(pbar):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train:
                (loss / grad_accum_steps).backward()
                if (step + 1) % grad_accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_n += images.size(0)
        pbar.set_postfix(loss=total_loss / total_n, acc=total_correct / total_n)

    # Flush any remaining accumulated gradients from a batch count not divisible by grad_accum_steps.
    if is_train and (step + 1) % grad_accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / total_n, total_correct / total_n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="clane9/imagenet-100")
    parser.add_argument("--image-size", type=int, default=160, help="RGB crop size fed into the DRF front-end")
    parser.add_argument("--drf-resolution", type=int, default=64, help="DRF output grid (H=W)")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="optimizer.step() every N batches")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--checkpoint", default="outputs/drf_vit_imagenet100.pt")
    parser.add_argument(
        "--continue", dest="resume", action="store_true",
        help="resume from the latest checkpoint (<checkpoint>.last.pt) if it exists",
    )
    args = parser.parse_args()
    last_checkpoint = os.path.splitext(args.checkpoint)[0] + ".last.pt"

    # ROCm exposes itself through the same "cuda" device namespace as CUDA builds.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")

    ds = load_dataset(args.dataset)
    train_tf, eval_tf = build_transforms(args.image_size)

    train_loader = DataLoader(
        ds["train"], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=make_collate(train_tf), pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        ds["validation"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=make_collate(eval_tf), pin_memory=True,
    )

    num_classes = ds["train"].features["label"].num_classes
    model = DRFViT(
        num_classes=num_classes,
        drf_resolution=(args.drf_resolution, args.drf_resolution),
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch, best_acc = 1, 0.0
    if args.resume and os.path.exists(last_checkpoint):
        ckpt = torch.load(last_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt["best_acc"]
        print(f"Resumed from {last_checkpoint} at epoch {start_epoch} (best_acc={best_acc:.4f})")

    epoch_bar = tqdm(range(start_epoch, args.epochs + 1), desc="epochs", initial=start_epoch - 1, total=args.epochs)
    for epoch in epoch_bar:
        t0 = time.time()
        train_loss, train_acc = run_epoch(
            model, train_loader, device, epoch=epoch, epochs=args.epochs,
            optimizer=optimizer, grad_accum_steps=args.grad_accum_steps,
        )
        val_loss, val_acc = run_epoch(model, val_loader, device, epoch=epoch, epochs=args.epochs)
        scheduler.step()

        tqdm.write(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"({time.time() - t0:.1f}s)"
        )
        epoch_bar.set_postfix(val_acc=val_acc, best_acc=best_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "args": vars(args), "val_acc": val_acc}, args.checkpoint)

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "args": vars(args),
        }, last_checkpoint)

    print(f"Best val_acc={best_acc:.4f}, checkpoint saved to {args.checkpoint}")


if __name__ == "__main__":
    main()
