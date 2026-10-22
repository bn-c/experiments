"""
Comprehensive unit test suite for BioInspiredVisionPipeline tensor shapes,
parameter counts, gradient flow, and resolution transformations across 224p and 1080p.
"""

import sys
from pathlib import Path
import unittest
import torch
import torch.nn as nn

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import (
    BioFrontEnd,
    SpaceVariantFoveatedPool,
    GroupedAdapter,
    BioInspiredVisionPipeline
)


class TestBioVisionPipeline(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_grouped_adapter_parameter_count(self):
        """Verifies adapter uses groups=256 and has ~43K params instead of ~1.18M."""
        adapter = GroupedAdapter(in_channels=512, out_channels=64)
        
        # Verify groups configuration
        self.assertEqual(adapter.grouped_conv.groups, 256)
        
        # Total parameters
        total_params = sum(p.numel() for p in adapter.parameters())
        print(f"[TEST] GroupedAdapter total parameters: {total_params:,}")
        
        # Should be strictly under 50,000 parameters
        self.assertLess(total_params, 50000)

    def test_shape_progression_224p(self):
        """Tests complete forward pass shape progression on 224x224 input."""
        model = BioInspiredVisionPipeline(backbone_name="resnet18", num_classes=100)
        model.eval()

        B, C, H, W = 2, 3, 224, 224
        x = torch.randn(B, C, H, W)

        with torch.no_grad():
            feats = model.forward_features(x)
            logits = model(x)

        # Check shape assertions
        self.assertEqual(feats["v1"].shape, (B, 512, 112, 112))
        self.assertEqual(feats["fov"].shape, (B, 512, 56, 56))
        self.assertEqual(feats["adapter"].shape, (B, 64, 56, 56))
        self.assertEqual(feats["layer1"].shape, (B, 64, 56, 56))
        self.assertEqual(feats["layer2"].shape, (B, 128, 28, 28))
        self.assertEqual(feats["layer3"].shape, (B, 256, 14, 14))
        self.assertEqual(feats["layer4"].shape, (B, 512, 7, 7))
        self.assertEqual(logits.shape, (B, 100))
        print("[TEST] 224x224 shape verification passed successfully!")

    def test_shape_progression_1080p(self):
        """Tests complete forward pass shape progression on 1080p (1920x1080) input."""
        model = BioInspiredVisionPipeline(backbone_name="resnet18", num_classes=100)
        model.eval()

        B, C, H, W = 1, 3, 1080, 1920
        x = torch.randn(B, C, H, W)

        with torch.no_grad():
            feats = model.forward_features(x)
            logits = model(x)

        # Check shape assertions for widescreen
        self.assertEqual(feats["v1"].shape, (B, 512, 540, 960))
        # 16:9 aspect-ratio-preserving grid: 42 x 75 = 3,150 points
        self.assertEqual(feats["fov"].shape, (B, 512, 42, 75))
        self.assertEqual(feats["adapter"].shape, (B, 64, 42, 75))
        self.assertEqual(feats["layer1"].shape, (B, 64, 42, 75))
        self.assertEqual(feats["layer2"].shape, (B, 128, 21, 38))
        self.assertEqual(feats["layer3"].shape, (B, 256, 11, 19))
        self.assertEqual(feats["layer4"].shape, (B, 512, 6, 10))
        self.assertEqual(logits.shape, (B, 100))
        print("[TEST] 1920x1080 widescreen shape verification passed successfully!")

    def test_gaze_fixation_jitter(self):
        """Tests forward pass with dynamic off-center fixation coordinates."""
        model = BioInspiredVisionPipeline(backbone_name="resnet18", num_classes=100)
        model.eval()

        B = 2
        x = torch.randn(B, 3, 224, 224)
        gaze = torch.tensor([[0.15, -0.20], [-0.10, 0.25]])

        with torch.no_grad():
            logits = model(x, gaze_center=gaze)

        self.assertEqual(logits.shape, (B, 100))
        self.assertFalse(torch.isnan(logits).any())
        print("[TEST] Gaze fixation jittering test passed successfully!")

    def test_gradient_flow(self):
        """Verifies backward pass gradients flow through adapter and backbone."""
        model = BioInspiredVisionPipeline(backbone_name="resnet18", num_classes=100)
        model.train()

        x = torch.randn(2, 3, 224, 224)
        targets = torch.tensor([12, 85])

        logits = model(x)
        loss = nn.functional.cross_entropy(logits, targets)
        loss.backward()

        # Check adapter gradients
        self.assertIsNotNone(model.adapter.grouped_conv.weight.grad)
        self.assertIsNotNone(model.adapter.pointwise_conv.weight.grad)
        # Check backbone gradients
        self.assertIsNotNone(model.fc.weight.grad)
        print("[TEST] Backward gradient flow verified successfully!")

    def test_temporal_cached_bypass(self):
        """Verifies temporal bypass when cached Layer 4 features are provided."""
        model = BioInspiredVisionPipeline(backbone_name="resnet18", num_classes=100)
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        cached_l4 = torch.randn(1, 512, 7, 7)

        with torch.no_grad():
            logits = model(dummy_input, cached_l4=cached_l4)

        self.assertEqual(logits.shape, (1, 100))
        print("[TEST] Temporal bypass verification passed successfully!")


if __name__ == "__main__":
    unittest.main()
