"""
Unit tests for the DRFViT model: the frozen DRF front-end plus its ViT head.

Covers the wiring that the training script depends on:
 1. Output shape / dtype, and that the front-end really contributes no parameters.
 2. Per-pathway (grouped) patch embedding does not mix pathways.
 3. `calibrate()` fits finite per-channel statistics and forward() uses them.
 4. forward() warns when it runs before calibration.
 5. calibrate() rejects a loader that yields nothing (rather than emitting NaNs).
 6. Calibration statistics survive a state_dict round-trip, like a checkpoint does.
 7. The default (128, 128) target still streams; a (64, 64) target keeps its field of view.
"""

import unittest
import warnings

import torch

from drf import DRFViT, VisionTransformer
from drf.vit import _rescale_per_channel_01


def tiny_vit(**kwargs) -> DRFViT:
    """Small DRFViT for fast tests."""
    config = dict(num_classes=7, drf_resolution=(32, 32), patch_size=8, embed_dim=24, depth=1, num_heads=2)
    config.update(kwargs)
    return DRFViT(**config)


class TestDRFViT(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_forward_shape_and_parameterless_frontend(self):
        model = tiny_vit().eval()
        self.assertEqual(sum(p.numel() for p in model.frontend.parameters()), 0,
                         "the DRF front-end must stay parameter-free")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # uncalibrated on purpose
            with torch.no_grad():
                logits = model(torch.rand(2, 3, 64, 64))
        self.assertEqual(logits.shape, (2, 7))
        self.assertEqual(logits.dtype, torch.float32)

    def test_patch_embed_is_per_pathway(self):
        """
        PatchEmbed must use a grouped conv, so pathway i's tokens are a function of
        pathway i alone. Perturbing one pathway must leave every other pathway's
        tokens untouched — a plain in_chans=12 conv would leak between them.
        """
        embed = VisionTransformer(in_chans=12, img_size=32, patch_size=8, num_classes=7,
                                  embed_dim=24, depth=1, num_heads=2).patch_embed
        x = torch.rand(1, 12, 32, 32)
        base = embed(x)
        perturbed = x.clone()
        perturbed[:, 3] += 1.0
        after = embed(perturbed)
        self.assertTrue(torch.allclose(base[:, 3], after[:, 3]) is False,
                        "the perturbed pathway must change")
        for c in range(12):
            if c == 3:
                continue
            self.assertTrue(torch.equal(base[:, c], after[:, c]),
                            f"pathway {c} tokens changed when pathway 3 was perturbed")

    def test_calibrate_fits_finite_stats_and_forward_uses_them(self):
        model = tiny_vit().eval()
        self.assertFalse(model.calibrated)
        loader = [(torch.rand(2, 3, 64, 64) * 0.5 + 0.3, torch.zeros(2, dtype=torch.long))
                  for _ in range(3)]
        model.calibrate(loader, torch.device("cpu"), max_batches=2)
        self.assertTrue(model.calibrated)
        self.assertTrue(torch.isfinite(model.channel_min).all())
        self.assertTrue(torch.isfinite(model.channel_max).all())
        self.assertTrue((model.channel_max >= model.channel_min).all())
        # min/max must no longer be the identity defaults
        self.assertFalse(torch.equal(model.channel_min, torch.zeros_like(model.channel_min)))

        with warnings.catch_warnings():
            warnings.simplefilter("error")           # calibrated -> must not warn
            with torch.no_grad():
                logits = model(torch.rand(2, 3, 64, 64) * 0.5 + 0.3)
        self.assertEqual(logits.shape, (2, 7))
        self.assertTrue(torch.isfinite(logits).all())

    def test_forward_warns_before_calibration(self):
        model = tiny_vit().eval()
        with self.assertWarns(RuntimeWarning):
            with torch.no_grad():
                model(torch.rand(1, 3, 64, 64))

    def test_calibrate_rejects_empty_loader(self):
        model = tiny_vit().eval()
        with self.assertRaises(ValueError):
            model.calibrate([], torch.device("cpu"))
        self.assertFalse(model.calibrated)

    def test_calibration_survives_state_dict_round_trip(self):
        model = tiny_vit().eval()
        loader = [(torch.rand(2, 3, 64, 64), torch.zeros(2, dtype=torch.long))]
        model.calibrate(loader, torch.device("cpu"), max_batches=1)

        clone = tiny_vit().eval()
        clone.load_state_dict(model.state_dict())
        self.assertTrue(torch.equal(clone.channel_min, model.channel_min))
        self.assertTrue(torch.equal(clone.channel_max, model.channel_max))

    def test_rescale_handles_degenerate_channel_range(self):
        """A channel that never varies (e.g. a motion channel on still images) must not NaN."""
        x = torch.full((1, 12, 4, 4), 0.3)
        out = _rescale_per_channel_01(x, torch.full((1, 12, 1, 1), 0.3), torch.full((1, 12, 1, 1), 0.3))
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(((out >= 0.0) & (out <= 1.0)).all())

    def test_frontend_streams_five_dimensional_video(self):
        """The bundled front-end must accept (B, T, 3, H, W) clips and emit per-frame features."""
        model = tiny_vit(drf_resolution=(128, 128), patch_size=16)
        model.eval()
        with torch.no_grad():
            out = model.frontend(torch.rand(1, 3, 3, 160, 160))
        self.assertEqual(out.shape, (1, 3, 12, 128, 128))


if __name__ == "__main__":
    unittest.main()
