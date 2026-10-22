"""
Unit tests for Video Temporal Gating, Dynamic Unfreezing Manager, and Uniform Control Baseline.
"""

import sys
from pathlib import Path
import unittest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import BioInspiredVisionPipeline
from src.video import TemporalFeatureGater
from src.training import DynamicUnfreezeManager
from src.evaluation import UniformDownsampledBaseline


class TestTrainingAndVideo(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_temporal_feature_gater(self):
        """Tests ego-motion robust temporal gating on synthetic sequential frames."""
        pipeline = BioInspiredVisionPipeline(backbone_name="resnet18", num_classes=10)
        pipeline.eval()
        gater = TemporalFeatureGater(state_reuse_threshold=0.15)

        frame1 = torch.randn(1, 3, 112, 112)
        logits1, bypassed1, delta1 = gater.process_frame(pipeline, frame1)

        # First frame should never bypass
        self.assertFalse(bypassed1)
        self.assertEqual(logits1.shape, (1, 10))

        # Frame 2 is identical to Frame 1 (delta ~ 0.0) -> should bypass Stages 3 & 4
        frame2 = frame1.clone()
        logits2, bypassed2, delta2 = gater.process_frame(pipeline, frame2)
        self.assertTrue(bypassed2)
        self.assertLess(delta2, 0.01)

        # Frame 3 is completely different -> should NOT bypass
        frame3 = torch.randn(1, 3, 112, 112)
        logits3, bypassed3, delta3 = gater.process_frame(pipeline, frame3)
        self.assertFalse(bypassed3)

        self.assertGreater(gater.bypass_rate, 0.0)
        print(f"[TEST] Temporal feature gating verified (bypass rate: {gater.bypass_rate:.1f}%)!")

    def test_dynamic_unfreeze_manager(self):
        """Verifies layer unfreezing sequence and 3-epoch learning rate dampening."""
        pipeline = BioInspiredVisionPipeline(backbone_name="resnet18", num_classes=10)
        mgr = DynamicUnfreezeManager(dampening_epochs=3)
        param_groups = mgr.build_param_groups(pipeline)
        optimizer = torch.optim.SGD(param_groups, lr=0.01)

        # Epoch 1: adapter & fc should be active; layer4, layer3, layer2, layer1 should be frozen
        mgr.update_epoch(1, optimizer)
        for group in optimizer.param_groups:
            name = group["name"]
            if name in ["adapter", "fc"]:
                self.assertTrue(group["params"][0].requires_grad)
            else:
                self.assertFalse(group["params"][0].requires_grad)

        # Epoch 6: Layer 4 should unfreeze with dampening = 1/3 ~ 0.33
        mgr.update_epoch(6, optimizer)
        for group in optimizer.param_groups:
            if group["name"] == "layer4":
                self.assertTrue(group["params"][0].requires_grad)
                self.assertAlmostEqual(group["lr_mult"], 1.0 * (1.0 / 3.0), places=3)

        # Epoch 8: Layer 4 dampening should be 3/3 = 1.0; Layer 3 should unfreeze with dampening = 1/3
        mgr.update_epoch(8, optimizer)
        for group in optimizer.param_groups:
            if group["name"] == "layer4":
                self.assertAlmostEqual(group["lr_mult"], 1.0, places=3)
            elif group["name"] == "layer3":
                self.assertTrue(group["params"][0].requires_grad)
                self.assertAlmostEqual(group["lr_mult"], 0.1 * (1.0 / 3.0), places=3)

        print("[TEST] Dynamic unfreezing schedule and LR dampening ramp verified!")

    def test_uniform_downsampled_control_baseline(self):
        """Verifies information-matched uniform control baseline avoids 2x2 Layer 4 collapse."""
        control = UniformDownsampledBaseline(num_classes=100)
        control.eval()

        # Input is 56x56
        x_56 = torch.randn(2, 3, 56, 56)
        with torch.no_grad():
            entry_out = control.entry(x_56)
            l1_out = control.layer1(entry_out)
            l4_out = control.layer4(control.layer3(control.layer2(l1_out)))
            logits = control(x_56)

        # Layer 1 must receive (2, 64, 56, 56)
        self.assertEqual(l1_out.shape, (2, 64, 56, 56))
        # Layer 4 must retain (2, 512, 7, 7) instead of collapsing to (2, 512, 2, 2)
        self.assertEqual(l4_out.shape, (2, 512, 7, 7))
        self.assertEqual(logits.shape, (2, 100))

        # Test full 224x224 auto-interpolation
        x_224 = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits_224 = control(x_224)
        self.assertEqual(logits_224.shape, (2, 100))

        print("[TEST] Information-matched uniform control baseline verified (Layer 4: 7x7)!")


if __name__ == "__main__":
    unittest.main()
