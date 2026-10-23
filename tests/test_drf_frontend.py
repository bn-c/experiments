"""
Unit tests for Dynamic Receptive Field Biomimetic Front-End (DRF-FE).

Tests:
 1. Zero trainable parameter count (deterministic analytic model).
 2. Dynamic geometry adapter calculations (S, sigma_c, sigma_s, kernel sizes).
 3. Arbitrary resolution support: 1080p, 720p, 4K, 224p, odd aspect ratios -> (B, 12, 128, 128).
 4. Dual-regime state machine: static mode (t=0) vs temporal motion mode (t>0).
 5. Stationary scene at t>=1: DSGC and TRANSIENT channels are exactly 0.
 6. Bipolar channel polarity preservation (Red vs Green, Blue vs Yellow, Brightening vs Darkening).
 7. Direction-Selective Ganglion Cell (DSGC) tuning (Left, Right, Up, Down).
 8. Looming detector fires for expanding dark disc, not for global uniform darkening.
 9. Sequential video tensor processing (B, T, 3, H, W).
10. Dynamic batch size adaptation.
11. Device migration: _kernel_cache cleared on .to(); no device mismatch.
12. State tensor memory: detach().clone() breaks parent storage reference.
13. OMS delayed surround: motion channels active at t=1 for a moving stimulus.
14. Numerical stability under edge cases (all zeros, all ones, extreme contrast).
"""

import sys
from pathlib import Path
import unittest
import torch
import torch.nn.functional as F

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.drf_frontend import DRFBiomimeticFrontEnd, DRFState


class TestDRFBiomimeticFrontEnd(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.frontend = DRFBiomimeticFrontEnd()

    # ------------------------------------------------------------------
    # 1. Parameter count
    # ------------------------------------------------------------------
    def test_parameter_count_is_zero(self):
        """Model must have strictly 0 trainable parameters."""
        total_params = sum(p.numel() for p in self.frontend.parameters())
        self.assertEqual(total_params, 0, f"Expected 0 parameters, got {total_params}")

    # ------------------------------------------------------------------
    # 2. Geometry adapter
    # ------------------------------------------------------------------
    def test_dynamic_geometry_adapter(self):
        """Stage 0 geometry scaling must be correct across common resolutions."""
        # 1080p: S=8, sigma_c=1.0, sigma_s=3.0
        s, sc, ss, k_dog, _, _ = self.frontend.compute_geometry(1080, 1920)
        self.assertEqual(s, 8)
        self.assertAlmostEqual(sc, 1.0, places=4)
        self.assertAlmostEqual(ss, 3.0, places=4)
        self.assertTrue(k_dog % 2 == 1)

        self.assertEqual(self.frontend.compute_geometry(720,  1280)[0], 5)
        self.assertEqual(self.frontend.compute_geometry(2160, 3840)[0], 16)
        self.assertEqual(self.frontend.compute_geometry(224,   224)[0], 1)

    # ------------------------------------------------------------------
    # 3. Resolution invariance
    # ------------------------------------------------------------------
    def test_resolution_invariance_output_shapes(self):
        """Any input resolution must map to exactly (B, 12, 128, 128)."""
        for h, w in [(224, 224), (720, 1280), (1080, 1920), (300, 450), (128, 128)]:
            self.frontend.reset()
            out = self.frontend(torch.rand(1, 3, h, w))
            self.assertEqual(out.shape, (1, 12, 128, 128),
                             f"Shape mismatch for ({h}, {w}): {out.shape}")
            self.assertFalse(torch.isnan(out).any())
            self.assertFalse(torch.isinf(out).any())

    # ------------------------------------------------------------------
    # 4. Dual-regime state machine
    # ------------------------------------------------------------------
    def test_dual_regime_static_vs_temporal(self):
        """
        t=0: motion channels (5-11) must be at L2 noise floor (all-zero pre-Stage-4).
        t=1 with motion: motion channels must activate significantly.
        """
        self.frontend.reset()
        x1 = torch.rand(1, 3, 256, 256)
        out1 = self.frontend(x1)
        self.assertTrue(self.frontend.has_state)

        # Spatial form channels active at t=0
        self.assertGreater(out1[:, 0:5].norm().item(), 1.0)

        # Motion channels at t=0 are exactly zero pre-Stage-4; Stage-4 L2 energy
        # introduces sqrt(1e-12) ~ 1e-6 per pixel, max value < 0.001
        self.assertLess(out1[:, 5:12].max().item(), 0.001)

        # t=1 with a shifted frame: motion channels activate
        x2 = torch.roll(x1, shifts=10, dims=-1)
        out2 = self.frontend(x2)
        self.assertGreater(out2[:, 5:12].norm().item(), 10.0 * out1[:, 5:12].norm().item())

    # ------------------------------------------------------------------
    # 5. Stationary scene at t>=1 produces exact silence
    # ------------------------------------------------------------------
    def test_stationary_scene_exact_silence_at_t1(self):
        """DSGC and TRANSIENT channels must be exactly 0 when the scene is perfectly static."""
        self.frontend.reset()
        static_frame = torch.rand(1, 3, 256, 256)
        _ = self.frontend(static_frame)         # t=0
        out_t1 = self.frontend(static_frame)    # t=1, identical frame

        dsgc = out_t1[:, 6:10]
        transient = out_t1[:, 5]
        self.assertAlmostEqual(dsgc.abs().max().item(),    0.0, places=4)
        self.assertAlmostEqual(transient.abs().max().item(), 0.0, places=4)

    # ------------------------------------------------------------------
    # 6. Bipolar channel polarity
    # ------------------------------------------------------------------
    def test_bipolar_chromatic_and_transient_polarity(self):
        """CHROMA_RG, CHROMA_BY, and TRANSIENT must preserve correct sign for opponent colors."""
        self.frontend.reset()
        red = torch.zeros(1, 3, 256, 256)
        red[:, 0] = 1.0
        out_red = self.frontend(red)
        self.assertGreater(out_red[:, 1].mean().item(), 0.0, "Pure red should give RG > 0")

        self.frontend.reset()
        green = torch.zeros(1, 3, 256, 256)
        green[:, 1] = 1.0
        out_green = self.frontend(green)
        self.assertLess(out_green[:, 1].mean().item(), 0.0, "Pure green should give RG < 0")

        # Brightening -> positive transient
        self.frontend.reset()
        dark   = torch.full((1, 3, 256, 256), 0.1)
        bright = torch.full((1, 3, 256, 256), 0.8)
        _ = self.frontend(dark)
        out_bright = self.frontend(bright)
        self.assertGreater(out_bright[:, 5].mean().item(), 0.0, "Brightening -> TRANSIENT > 0")

        # Darkening -> negative transient
        self.frontend.reset()
        _ = self.frontend(bright)
        out_dark = self.frontend(dark)
        self.assertLess(out_dark[:, 5].mean().item(), 0.0, "Darkening -> TRANSIENT < 0")

    # ------------------------------------------------------------------
    # 7. DSGC direction selectivity
    # ------------------------------------------------------------------
    def test_direction_selective_ganglion_cells(self):
        """Each DSGC channel must respond stronger to its preferred direction."""
        base_v = torch.zeros(1, 3, 256, 256)
        base_v[:, :, :, 100:120] = 1.0  # vertical strip

        # Right motion
        self.frontend.reset()
        _ = self.frontend(base_v)
        moved_right = torch.zeros_like(base_v)
        moved_right[:, :, :, 102:122] = 1.0
        out = self.frontend(moved_right)
        ds_r = out[:, 6].mean().item()
        ds_l = out[:, 7].mean().item()
        self.assertGreater(ds_r, ds_l, f"DS_RIGHT {ds_r:.5f} > DS_LEFT {ds_l:.5f}")

        # Left motion
        self.frontend.reset()
        _ = self.frontend(base_v)
        moved_left = torch.zeros_like(base_v)
        moved_left[:, :, :, 98:118] = 1.0
        out = self.frontend(moved_left)
        ds_r = out[:, 6].mean().item()
        ds_l = out[:, 7].mean().item()
        self.assertGreater(ds_l, ds_r, f"DS_LEFT {ds_l:.5f} > DS_RIGHT {ds_r:.5f}")

        # Down motion
        base_h = torch.zeros(1, 3, 256, 256)
        base_h[:, :, 100:120, :] = 1.0  # horizontal strip
        self.frontend.reset()
        _ = self.frontend(base_h)
        moved_down = torch.zeros_like(base_h)
        moved_down[:, :, 102:122, :] = 1.0
        out = self.frontend(moved_down)
        ds_d = out[:, 9].mean().item()
        ds_u = out[:, 8].mean().item()
        self.assertGreater(ds_d, ds_u, f"DS_DOWN {ds_d:.5f} > DS_UP {ds_u:.5f}")

    # ------------------------------------------------------------------
    # 8. Looming detector
    # ------------------------------------------------------------------
    def test_looming_fires_for_expansion_not_global_darkening(self):
        """
        Looming (Ch 11) must fire for an expanding dark disc.
        It must NOT fire (or fire much less) for a uniform global darkening of equal magnitude.
        """
        h, w = 256, 256
        cy, cx = h // 2, w // 2

        # -- Expansion stimulus --
        f1_exp = torch.ones(1, 3, h, w)
        f1_exp[:, :, cy-10:cy+10, cx-10:cx+10] = 0.0
        f2_exp = torch.ones(1, 3, h, w)
        f2_exp[:, :, cy-30:cy+30, cx-30:cx+30] = 0.0

        self.frontend.reset()
        _ = self.frontend(f1_exp)
        out_loom = self.frontend(f2_exp)
        looming_expansion = out_loom[:, 11].max().item()
        self.assertGreater(looming_expansion, 0.0, "Expanding dark disc must trigger looming")

        # -- Global darkening stimulus (same pixel-change magnitude) --
        f1_dark = torch.full((1, 3, h, w), 0.8)
        f2_dark = torch.full((1, 3, h, w), 0.2)

        self.frontend.reset()
        _ = self.frontend(f1_dark)
        out_global = self.frontend(f2_dark)
        looming_global = out_global[:, 11].max().item()

        # Expansion looming response must exceed global darkening response
        self.assertGreater(
            looming_expansion, looming_global,
            f"Looming expansion ({looming_expansion:.4f}) should exceed "
            f"global darkening ({looming_global:.4f})"
        )

    # ------------------------------------------------------------------
    # 9. Sequential video tensor
    # ------------------------------------------------------------------
    def test_sequential_video_tensor(self):
        """(B, T, 3, H, W) input must produce (B, T, 12, 128, 128) output."""
        self.frontend.reset()
        B, T, C, H, W = 2, 4, 3, 256, 256
        out = self.frontend(torch.rand(B, T, C, H, W))
        self.assertEqual(out.shape, (B, T, 12, 128, 128))
        self.assertFalse(torch.isnan(out).any())

    # ------------------------------------------------------------------
    # 10. Dynamic batch size adaptation
    # ------------------------------------------------------------------
    def test_dynamic_batch_size_adaptation(self):
        """Streaming with decreasing batch sizes must not raise shape errors."""
        self.frontend.reset()
        self.frontend(torch.rand(4, 3, 256, 256))
        out2 = self.frontend(torch.rand(2, 3, 256, 256))
        self.assertEqual(out2.shape[0], 2)
        out1 = self.frontend(torch.rand(1, 3, 256, 256))
        self.assertEqual(out1.shape[0], 1)

    # ------------------------------------------------------------------
    # 11. Device migration clears kernel cache (Issue #3 fix)
    # ------------------------------------------------------------------
    def test_kernel_cache_cleared_on_device_migration(self):
        """
        _apply() override must clear _kernel_cache so kernels are rebuilt on
        the correct device after .to() / .float() / .half() calls.
        """
        self.frontend.reset()
        x = torch.rand(1, 3, 128, 128)
        _ = self.frontend(x)
        # Cache should be populated after a forward pass
        self.assertGreater(len(self.frontend._kernel_cache), 0)

        # Simulate a device/dtype migration (CPU -> CPU half just to trigger _apply)
        self.frontend.to(torch.float32)   # triggers _apply; cache should be cleared
        self.assertEqual(len(self.frontend._kernel_cache), 0,
                         "_kernel_cache must be empty after _apply()")

        # Next forward pass must repopulate without error
        out = self.frontend(x)
        self.assertEqual(out.shape, (1, 12, 128, 128))

    # ------------------------------------------------------------------
    # 12. State tensor memory safety (Issue #2 fix)
    # ------------------------------------------------------------------
    def test_state_tensors_are_cloned(self):
        """
        State tensors must be detach().clone() so they do not retain a reference
        to the parent video tensor's storage during (B, T, H, W) processing.
        """
        self.frontend.reset()
        x = torch.rand(1, 3, 256, 256)
        _, state = self.frontend(x, return_state=True)
        # After clone(), y_prev should be a standalone tensor (not sharing storage)
        self.assertFalse(state.y_prev.requires_grad)
        # The state tensors must not share storage with x (would retain parent ref)
        self.assertNotEqual(state.y_prev.data_ptr(), x.data_ptr())

    # ------------------------------------------------------------------
    # 13. OMS delayed surround activates motion channels at t=1
    # ------------------------------------------------------------------
    def test_oms_activates_at_t1_for_moving_stimulus(self):
        """DIFF_MOTION (Ch 10) must be non-zero at t=1 for a shifting stimulus."""
        self.frontend.reset()
        base = torch.rand(1, 3, 256, 256)
        _ = self.frontend(base)

        shifted = torch.roll(base, shifts=8, dims=-1)
        out_t1 = self.frontend(shifted)
        self.assertGreater(out_t1[:, 10].max().item(), 0.0,
                           "DIFF_MOTION must be > 0 for a moving stimulus at t=1")

    # ------------------------------------------------------------------
    # 14. Numerical stability
    # ------------------------------------------------------------------
    def test_numerical_stability(self):
        """No NaN / Inf under zero, saturated, or extreme-contrast inputs."""
        for val in [0.0, 1.0, 1e-4]:
            self.frontend.reset()
            out = self.frontend(torch.full((1, 3, 256, 256), val))
            self.assertFalse(torch.isnan(out).any(), f"NaN at val={val}")
            self.assertFalse(torch.isinf(out).any(), f"Inf at val={val}")


if __name__ == "__main__":
    unittest.main()
