"""
Regression tests for the training-loop helpers in `scripts/`.

The scripts are not an importable package, so each one is loaded from its file path.
Covered here:
 1. `run_epoch` rejects an empty loader instead of raising UnboundLocalError /
    ZeroDivisionError half-way through the summary maths.
 2. The trailing partial gradient-accumulation group is rescaled, so a loader whose
    batch count is not a multiple of `grad_accum_steps` does not under-weight its last
    update.
"""

import unittest

import torch
import torch.nn as nn

from tests._script_loader import load_script

vit_training = load_script("train_vit_imagenet100", "train_vit_imagenet100.py")
fracsrf_training = load_script("train_fracsrf_resnet32_cifar10", "train_fracsrf_resnet32_cifar10.py")


def _quiet_tqdm(iterable, *args, **kwargs):
    """Stand-in for tqdm so the tests do not print progress bars."""

    class _Bar:
        def __init__(self, it):
            self._it = it

        def __iter__(self):
            return iter(self._it)

        def set_postfix(self, *a, **k):
            pass

    return _Bar(iterable)


for _module in (vit_training, fracsrf_training):
    _module.tqdm = _quiet_tqdm


class _TinyClassifier(nn.Module):
    """2-class linear model over 3 inputs, shaped like the scripts' (images, labels) batches."""

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(3, 2)

    def forward(self, x):
        return self.layer(x)


class TestEmptyLoader(unittest.TestCase):
    def test_vit_run_epoch_rejects_empty_loader(self):
        model = _TinyClassifier()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        with self.assertRaises(ValueError):
            vit_training.run_epoch(model, [], torch.device("cpu"), epoch=1, epochs=2,
                                   optimizer=optimizer, grad_accum_steps=2)
        with self.assertRaises(ValueError):
            vit_training.run_epoch(model, [], torch.device("cpu"), epoch=1, epochs=2)

    def test_fracsrf_run_epoch_rejects_empty_loader(self):
        model = _TinyClassifier()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        with self.assertRaises(ValueError):
            fracsrf_training.run_epoch(model, [], torch.device("cpu"), epoch=1, epochs=2,
                                       optimizer=optimizer)


class TestGradientAccumulation(unittest.TestCase):
    """A single-batch step must be independent of `grad_accum_steps`."""

    def _weights_after_one_batch(self, accum_steps: int) -> torch.Tensor:
        torch.manual_seed(0)
        model = _TinyClassifier()
        torch.manual_seed(1)                     # same data regardless of accum
        images = torch.randn(3, 3)
        labels = torch.tensor([0, 1, 0])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        vit_training.run_epoch(model, [(images, labels)], torch.device("cpu"), epoch=1, epochs=1,
                               optimizer=optimizer, grad_accum_steps=accum_steps)
        return torch.cat([p.detach().flatten() for p in model.parameters()])

    def test_tail_group_is_rescaled(self):
        reference = self._weights_after_one_batch(1)
        for accum_steps in (2, 3, 5):
            weights = self._weights_after_one_batch(accum_steps)
            self.assertTrue(
                torch.allclose(weights, reference, atol=1e-6),
                f"accum={accum_steps} under-weighted its only (partial) group: "
                f"max diff {(weights - reference).abs().max().item():.3e}",
            )


if __name__ == "__main__":
    unittest.main()
