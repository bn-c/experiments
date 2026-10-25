#!/usr/bin/env python
"""
Train FracSRF-ResNet-32 (Saldanha et al., BMVC 2021, arXiv:2111.06660) on
CIFAR-10, loaded from the Hugging Face `uoft-cs/cifar10` dataset.

Example:
    python scripts/train_fracsrf_resnet32_cifar10.py --epochs 200 --batch-size 256
"""

import argparse
import os
import random
import time

import torch
import torch.nn as nn
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from fracsrf import FracSRFResNet32

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def _to_tensor(img: Image.Image) -> torch.Tensor:
    arr = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    t = arr.view(img.size[1], img.size[0], 3).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(CIFAR_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR_STD).view(3, 1, 1)
    return (t - mean) / std


def _pad_reflect(img: Image.Image, pad: int) -> Image.Image:
    w, h = img.size
    canvas = Image.new("RGB", (w + 2 * pad, h + 2 * pad))
    canvas.paste(img, (pad, pad))
    left = img.crop((0, 0, pad, h)).transpose(Image.FLIP_LEFT_RIGHT)
    right = img.crop((w - pad, 0, w, h)).transpose(Image.FLIP_LEFT_RIGHT)
    canvas.paste(left, (0, pad))
    canvas.paste(right, (w + pad, pad))
    top = canvas.crop((0, pad, w + 2 * pad, 2 * pad)).transpose(Image.FLIP_TOP_BOTTOM)
    bottom = canvas.crop((0, h, w + 2 * pad, h + pad)).transpose(Image.FLIP_TOP_BOTTOM)
    canvas.paste(top, (0, 0))
    canvas.paste(bottom, (0, h + pad))
    return canvas


def build_transforms(image_size: int = 32, pad: int = 4):
    def train_tf(img: Image.Image) -> torch.Tensor:
        img = _pad_reflect(img, pad)
        w, h = img.size
        x0 = random.randint(0, w - image_size)
        y0 = random.randint(0, h - image_size)
        img = img.crop((x0, y0, x0 + image_size, y0 + image_size))
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return _to_tensor(img)

    def eval_tf(img: Image.Image) -> torch.Tensor:
        return _to_tensor(img)

    return train_tf, eval_tf


def make_collate(transform, image_key: str):
    def collate(batch):
        images = torch.stack([transform(ex[image_key].convert("RGB")) for ex in batch])
        labels = torch.tensor([ex["label"] for ex in batch], dtype=torch.long)
        return images, labels
    return collate


def run_epoch(model, loader, device, epoch=None, epochs=None, optimizer=None, clip_orders=False):
    is_train = optimizer is not None
    model.train(is_train)
    criterion = nn.CrossEntropyLoss()

    desc = f"epoch {epoch:03d}/{epochs} [{'train' if is_train else 'val'}]" if epoch else ("train" if is_train else "val")
    pbar = tqdm(loader, desc=desc, leave=False)

    total_loss, total_correct, total_n = 0.0, 0, 0
    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if clip_orders:
                    model.clip_orders_()

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_n += images.size(0)
        pbar.set_postfix(loss=total_loss / total_n, acc=total_correct / total_n)

    if total_n == 0:
        raise ValueError("run_epoch() received an empty loader; nothing to iterate over.")

    return total_loss / total_n, total_correct / total_n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="uoft-cs/cifar10")
    parser.add_argument("--image-key", default="img")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.05, help="LR for alpha/nu/BN/FC params (paper setup)")
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--learn-sigma", action="store_true", default=True)
    parser.add_argument("--no-learn-sigma", dest="learn_sigma", action="store_false")
    parser.add_argument("--sigma-lr", type=float, default=0.001)
    parser.add_argument("--sigma-weight-decay", type=float, default=0.01)
    parser.add_argument("--order-init-low", type=float, default=1.0)
    parser.add_argument("--order-init-high", type=float, default=3.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0, help="seed for python/torch RNGs")
    parser.add_argument("--checkpoint", default="outputs/fracsrf_resnet32_cifar10.pt")
    parser.add_argument(
        "--continue", dest="resume", action="store_true",
        help="resume from the latest checkpoint (<checkpoint>.last.pt) if it exists",
    )
    args = parser.parse_args()
    last_checkpoint = os.path.splitext(args.checkpoint)[0] + ".last.pt"
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")

    ds = load_dataset(args.dataset)
    train_tf, eval_tf = build_transforms(image_size=32)

    train_loader = DataLoader(
        ds["train"], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=make_collate(train_tf, args.image_key), pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        ds["test"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=make_collate(eval_tf, args.image_key), pin_memory=True,
    )

    model = FracSRFResNet32(
        num_classes=10,
        order_init_range=(args.order_init_low, args.order_init_high),
        learn_sigma=args.learn_sigma,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"FracSRF-ResNet-32: {n_params:,} trainable parameters")

    param_groups = [{"params": list(model.non_sigma_parameters()), "lr": args.lr, "weight_decay": args.weight_decay}]
    sigma_params = list(model.sigma_parameters())
    if sigma_params:
        param_groups.append({"params": sigma_params, "lr": args.sigma_lr, "weight_decay": args.sigma_weight_decay})

    optimizer = torch.optim.SGD(param_groups, momentum=0.9)
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
            optimizer=optimizer, clip_orders=True,
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
