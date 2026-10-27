"""HuggingFace vision / tabular-friendly image classification loaders.

Presets (namespace/name Hub ids — current ``datasets`` rejects bare names):

* ``cifar10`` / ``cifar100`` — ``uoft-cs/cifar*``
* ``mnist`` — ``ylecun/mnist``
* ``svhn`` — ``ufldl-stanford/svhn`` (cropped_digits)
* ``stl10`` — ``tanganke/stl10`` (ImageNette-scale stand-in; resized)
* ``fashion_mnist`` — ``zalando-datasets/fashion_mnist``

Generic entry ``hf`` takes the same knobs without a preset. Images are resized
(optional), converted to float tensors in ``[0, 1]``, flattened for linear/MLP,
and train-normalized. Use ``max_train`` / ``max_test`` for CPU-cheap smokes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from datasets import load_dataset
except ImportError as exc:  # pragma: no cover
    raise ImportError("linear_bench HF loaders require the `datasets` package") from exc

from PIL import Image

PRESETS: dict[str, dict[str, Any]] = {
    "cifar10": {
        "hf_path": "uoft-cs/cifar10",
        "image_key": "img",
        "label_key": "label",
        "n_classes": 10,
    },
    "cifar100": {
        "hf_path": "uoft-cs/cifar100",
        "image_key": "img",
        "label_key": "fine_label",
        "n_classes": 100,
    },
    "mnist": {
        "hf_path": "ylecun/mnist",
        "image_key": "image",
        "label_key": "label",
        "n_classes": 10,
    },
    "svhn": {
        "hf_path": "ufldl-stanford/svhn",
        "hf_name": "cropped_digits",
        "image_key": "image",
        "label_key": "label",
        "n_classes": 10,
    },
    "stl10": {
        "hf_path": "tanganke/stl10",
        "image_key": "image",
        "label_key": "label",
        "n_classes": 10,
        "resize": 32,  # 96→32 keeps linear/MLP smoke cheap
    },
    "fashion_mnist": {
        "hf_path": "zalando-datasets/fashion_mnist",
        "image_key": "image",
        "label_key": "label",
        "n_classes": 10,
    },
}


class _FlatImageDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def _split_slice(split: str, limit: int | None) -> str:
    if limit is None:
        return split
    return f"{split}[:{int(limit)}]"


def _to_rgb_array(img: Any, resize: int | None) -> np.ndarray:
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.asarray(img))
    if resize is not None:
        img = img.resize((int(resize), int(resize)), Image.BILINEAR)
    if img.mode != "RGB":
        # Keep single-channel as RGB-stacked so in_dim is consistent when mixing
        # grayscale presets; callers that want L can set keep_grayscale.
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def _materialize(
    rows: Any,
    image_key: str,
    label_key: str,
    resize: int | None,
    keep_grayscale: bool,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for row in rows:
        img = row[image_key]
        if keep_grayscale:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.asarray(img))
            if resize is not None:
                img = img.resize((int(resize), int(resize)), Image.BILINEAR)
            if img.mode != "L":
                img = img.convert("L")
            arr = np.asarray(img, dtype=np.float32) / 255.0
        else:
            arr = _to_rgb_array(img, resize)
        xs.append(arr.reshape(-1))
        ys.append(int(row[label_key]))
    return np.stack(xs, axis=0), np.asarray(ys, dtype=np.int64)


def load_hf_vision(kwargs: dict[str, Any]) -> tuple[Dataset, Dataset, dict[str, Any]]:
    """Load a preset or arbitrary HF image-classification dataset."""
    preset_name = kwargs.get("preset")
    cfg: dict[str, Any] = {}
    if preset_name:
        if preset_name not in PRESETS:
            raise KeyError(f"unknown HF preset {preset_name!r}; known: {sorted(PRESETS)}")
        cfg.update(PRESETS[preset_name])
    cfg.update({k: v for k, v in kwargs.items() if k != "preset" and v is not None})

    hf_path = cfg.get("hf_path")
    if not hf_path:
        raise ValueError("hf loader requires hf_path or a known preset")

    hf_name = cfg.get("hf_name")
    image_key = str(cfg.get("image_key", "image"))
    label_key = str(cfg.get("label_key", "label"))
    train_split = str(cfg.get("train_split", "train"))
    test_split = str(cfg.get("test_split", "test"))
    max_train = cfg.get("max_train")
    max_test = cfg.get("max_test")
    resize = cfg.get("resize")
    keep_grayscale = bool(cfg.get("keep_grayscale", False))
    # Grayscale presets default to L→flat unless explicitly RGB
    if preset_name in {"mnist", "fashion_mnist"} and "keep_grayscale" not in kwargs:
        keep_grayscale = True

    load_kw: dict[str, Any] = {}
    if hf_name is not None:
        load_kw["name"] = hf_name

    train_rows = load_dataset(hf_path, split=_split_slice(train_split, max_train), **load_kw)
    test_rows = load_dataset(hf_path, split=_split_slice(test_split, max_test), **load_kw)

    x_tr, y_tr = _materialize(train_rows, image_key, label_key, resize, keep_grayscale)
    x_te, y_te = _materialize(test_rows, image_key, label_key, resize, keep_grayscale)

    # SVHN cropped_digits historically uses label 10 for digit 0 in some dumps;
    # Hub parquet uses 0–9, but clamp defensively if 10 appears.
    if int(y_tr.max()) == 10 or int(y_te.max()) == 10:
        y_tr = np.where(y_tr == 10, 0, y_tr)
        y_te = np.where(y_te == 10, 0, y_te)

    mean = x_tr.mean(axis=0, keepdims=True)
    std = x_tr.std(axis=0, keepdims=True) + 1e-6
    x_tr = (x_tr - mean) / std
    x_te = (x_te - mean) / std

    n_classes = int(cfg.get("n_classes") or (max(int(y_tr.max()), int(y_te.max())) + 1))
    in_dim = int(x_tr.shape[1])

    meta: dict[str, Any] = {
        "task": "classification",
        "in_dim": in_dim,
        "out_dim": n_classes,
        "n_classes": n_classes,
        "hf_path": hf_path,
        "hf_name": hf_name,
        "preset": preset_name,
        "image_key": image_key,
        "label_key": label_key,
        "resize": resize,
        "keep_grayscale": keep_grayscale,
        "n_train": int(x_tr.shape[0]),
        "n_test": int(x_te.shape[0]),
        "source": "huggingface",
    }
    return _FlatImageDataset(x_tr, y_tr), _FlatImageDataset(x_te, y_te), meta


def _preset_loader(name: str):
    def _load(kwargs: dict[str, Any]) -> tuple[Dataset, Dataset, dict[str, Any]]:
        merged = dict(kwargs)
        merged.setdefault("preset", name)
        return load_hf_vision(merged)

    _load.__name__ = f"load_{name}"
    _load.__doc__ = f"HF preset loader for {name}."
    return _load


load_cifar10 = _preset_loader("cifar10")
load_cifar100 = _preset_loader("cifar100")
load_mnist = _preset_loader("mnist")
load_svhn = _preset_loader("svhn")
load_stl10 = _preset_loader("stl10")
load_fashion_mnist = _preset_loader("fashion_mnist")
