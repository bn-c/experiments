"""
Unit tests for Dynamic Receptive Field Biomimetic Front-End (DRF-FE).

Tests:
 1. Zero trainable parameter count (deterministic analytic model).
 2. Dynamic geometry adapter calculations (S, sigma_c, sigma_s, kernel sizes), that the
    stride follows `target_resolution`, and that the retained field of view is
    continuous across the stride's rounding boundary.
 3. Arbitrary resolution support: 1080p, 720p, 4K, 224p, odd aspect ratios -> (B, 12, 128, 128).
 4. Dual-regime state machine: static mode (t=0) vs temporal motion mode (t>0).
 5. Stationary scene at t>=1: DSGC and TRANSIENT channels are exactly 0.
 6. Chromatic opponency: sign for opponent colours, and exact cancellation for grey.
 7. Direction-Selective Ganglion Cell (DSGC) tuning: preferred > null for each axis, and
    *orthogonal* channels stay silent on a pure translation.
 8. Looming detector: fires for an expanding dark disc, not for global darkening, and not
    for a rigidly translating dark object of constant size.
 9. Sequential video tensor processing (B, T, 3, H, W).
10. Dynamic batch adaptation: shrinking and growing batch sizes mid-stream.
11. Device migration: _kernel_cache cleared and stale _state dropped on .to().
12. State tensor memory: detach().clone() breaks parent storage reference.
13. OMS delayed surround: motion channels active at t=1 for a moving stimulus.
14. Numerical stability under edge cases (all zeros, all ones, extreme contrast).
15. Border-artifact free: spatial filters replicate-pad, so a uniform field produces
    no spurious edge response at the frame boundary.
16. Letterboxing: non-square (wide/tall) inputs are padded to a square before Stage 0,
    so far-edge content survives instead of being center-cropped away.
17. Letterbox bars are flat, not streaked copies of a textured border row, and are
    constant for a whole clip so they cannot manufacture motion in the bar band.
18. No global-luminance false positives: a uniform brightening/darkening drives no
    chromatic, DSGC or looming channel.
19. Stride-1 decimation is the identity (no needless low-pass when nothing is downsampled).
"""

import sys
from pathlib import Path
import unittest
import torch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drf import DRFBiomimeticFrontEnd


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
        s, sc, ss, k_center, _, _ = self.frontend.compute_geometry(1080, 1920)
        self.assertEqual(s, 8)
        self.assertAlmostEqual(sc, 1.0, places=4)
        self.assertAlmostEqual(ss, 3.0, places=4)
        self.assertTrue(k_center % 2 == 1)

        self.assertEqual(self.frontend.compute_geometry(720, 1280)[0], 6)
        self.assertEqual(self.frontend.compute_geometry(2160, 3840)[0], 17)
        self.assertEqual(self.frontend.compute_geometry(224, 224)[0], 2)

    def test_stride_uses_configured_target_resolution(self):
        """
        The stride must come from `target_resolution`, not from a hard-coded 128:
        otherwise a (64, 64) front-end downsamples as if it were 128 and then center
        crops, throwing away most of the frame.
        """
        for target, size in [((128, 128), 256), ((64, 64), 256), ((32, 32), 256)]:
            fe = DRFBiomimeticFrontEnd(target_resolution=target)
            self.assertEqual(fe.compute_geometry(size, size)[0], size // target[0])

    def test_stride_is_continuous_across_its_rounding_boundary(self):
        """
        A one-pixel change in input size must not swing the retained field of view.
        Flooring the ratio did exactly that: 255px -> S=1 -> half the frame cropped
        away, 256px -> S=2 -> the whole frame kept.
        """
        marker = torch.zeros(1, 3, 1, 1)
        survivals = []
        for n in (254, 255, 256, 257, 510, 511, 512, 513):
            x = torch.zeros(1, 3, n, n)
            span = max(4, n // 40)
            x[:, :, :span, :span] = 1.0          # marker in the far corner
            fe = DRFBiomimeticFrontEnd()
            fe.reset()
            survives = fe(x)[:, 0].max().item() > 0.05
            survivals.append(survives)
            self.assertTrue(survives, f"corner content lost at {n}x{n} (S={fe.compute_geometry(n, n)[0]})")
        self.assertTrue(all(survivals))

    def test_non_default_target_keeps_field_of_view(self):
        """A (64, 64) or (32, 32) front-end must still see the whole frame, not a crop."""
        for target in [(64, 64), (32, 32)]:
            fe = DRFBiomimeticFrontEnd(target_resolution=target)
            fe.reset()
            x = torch.zeros(1, 3, 160, 160)
            x[:, :, 4:24, 4:24] = 1.0            # far corner
            out = fe(x)
            self.assertEqual(out.shape[-2:], target)
            self.assertGreater(out[:, 0].max().item(), 0.05,
                               f"corner content lost with target_resolution={target}")

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
    # 16. Letterboxing preserves far-edge content on wide/tall inputs
    # ------------------------------------------------------------------
    def test_letterbox_preserves_wide_aspect_ratio_content(self):
        """
        A bright marker near the far edge of a 16:9 frame must still show up in the
        output. With letterbox=False (legacy crop_pad-only scaling), the marker is
        center-cropped away; with letterbox=True (default) it must survive.
        """
        x = torch.zeros(1, 3, 1080, 1920)
        x[:, :, 500:580, 20:100] = 1.0  # near the far-left edge of a wide frame

        letterboxed = DRFBiomimeticFrontEnd(letterbox=True)
        letterboxed.reset()
        out_letterboxed = letterboxed(x)
        self.assertGreater(out_letterboxed[:, 0].max().item(), 0.0,
                           "Letterboxed model must preserve far-edge content")

        cropped = DRFBiomimeticFrontEnd(letterbox=False)
        cropped.reset()
        out_cropped = cropped(x)
        self.assertAlmostEqual(out_cropped[:, 0].max().item(), 0.0, places=4,
                               msg="Non-letterboxed model center-crops the marker away")

    # ------------------------------------------------------------------
    # 17. Letterbox bars are flat, not streaked
    # ------------------------------------------------------------------
    def test_letterbox_bars_are_flat_not_streaked(self):
        """
        A textured border row/column must not be replicated verbatim down the letterbox
        bar (which would stripe its per-column/per-row texture into streaks); the bar
        must be a flat mean-color fill instead.
        """
        x = torch.rand(1, 3, 1080, 1920)  # high-frequency texture reaches every edge
        padded = DRFBiomimeticFrontEnd._letterbox_to_square(x)
        self.assertEqual(padded.shape[-2:], (1920, 1920))

        bar_top = padded[:, :, :420, :]  # letterbox bar added above the real content
        # A flat fill has ~zero variance across the bar; a replicated textured row would not.
        self.assertLess(bar_top.var(dim=(-2, -1)).max().item(), 1e-6,
                        "Letterbox bar must be a flat fill, not a replicated textured row")

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

        # Motion channels at t=0 are exactly zero (Stage 3 rectifies without a floor)
        self.assertEqual(out1[:, 5:12].abs().max().item(), 0.0)

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
    # 6. Chromatic opponency
    # ------------------------------------------------------------------
    def test_bipolar_chromatic_and_transient_polarity(self):
        """CHROMA_RG, CHROMA_BY, and TRANSIENT must preserve correct sign for opponent colors."""
        # CHROMA_RG/BY are raw (unfiltered) opponent channels, so a coloured square against
        # a neutral background is what exercises their sign. The square is cropped in
        # *output* (128x128) coordinates since Stage 3 decimates by stride S.
        out_crop = slice(54, 74)
        self.frontend.reset()
        red = torch.full((1, 3, 256, 256), 0.5)
        red[:, 0, 108:148, 108:148] = 1.0
        red[:, 1:, 108:148, 108:148] = 0.0
        out_red = self.frontend(red)
        self.assertGreater(out_red[:, 1, out_crop, out_crop].mean().item(), 0.0, "Red square should give RG > 0")

        self.frontend.reset()
        green = torch.full((1, 3, 256, 256), 0.5)
        green[:, 1, 108:148, 108:148] = 1.0
        green[:, 0, 108:148, 108:148] = 0.0
        green[:, 2, 108:148, 108:148] = 0.0
        out_green = self.frontend(green)
        self.assertLess(out_green[:, 1, out_crop, out_crop].mean().item(), 0.0, "Green square should give RG < 0")

        # Blue>Yellow -> positive BY; a yellow square -> negative BY.
        self.frontend.reset()
        blue = torch.full((1, 3, 256, 256), 0.5)
        blue[:, 2, 108:148, 108:148] = 1.0
        out_blue = self.frontend(blue)
        self.assertGreater(out_blue[:, 2, out_crop, out_crop].mean().item(), 0.0, "Blue square should give BY > 0")

        self.frontend.reset()
        yellow = torch.full((1, 3, 256, 256), 0.5)
        yellow[:, 0, 108:148, 108:148] = 1.0
        yellow[:, 1, 108:148, 108:148] = 1.0
        yellow[:, 2, 108:148, 108:148] = 0.0
        out_yellow = self.frontend(yellow)
        self.assertLess(out_yellow[:, 2, out_crop, out_crop].mean().item(), 0.0, "Yellow square should give BY < 0")

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

    def test_dsgc_orthogonal_channels_stay_silent(self):
        """
        A horizontally moving bar must not drive the vertical DSGC pair (and vice versa).

        This is the property the old null-side-shift formulation got wrong: at the
        leading edge of any moving edge the null sample is empty, so *all four*
        channels saturated to 1.0 regardless of direction.
        """
        n = 256
        bar_v = torch.zeros(1, 3, n, n)
        bar_v[:, :, :, 100:120] = 1.0
        moved_right = torch.zeros_like(bar_v)
        moved_right[:, :, :, 108:128] = 1.0      # 8 px right: a realistic step

        self.frontend.reset()
        _ = self.frontend(bar_v)
        out = self.frontend(moved_right)
        self.assertEqual(out[:, 8].abs().max().item(), 0.0, "DS_UP must not fire for horizontal motion")
        self.assertEqual(out[:, 9].abs().max().item(), 0.0, "DS_DOWN must not fire for horizontal motion")
        self.assertGreater(out[:, 6].max().item(), 0.05, "DS_RIGHT must fire for rightward motion")

        bar_h = torch.zeros(1, 3, n, n)
        bar_h[:, :, 100:120, :] = 1.0
        moved_down = torch.zeros_like(bar_h)
        moved_down[:, :, 108:128, :] = 1.0

        self.frontend.reset()
        _ = self.frontend(bar_h)
        out = self.frontend(moved_down)
        self.assertEqual(out[:, 6].abs().max().item(), 0.0, "DS_RIGHT must not fire for vertical motion")
        self.assertEqual(out[:, 7].abs().max().item(), 0.0, "DS_LEFT must not fire for vertical motion")
        self.assertGreater(out[:, 9].max().item(), 0.05, "DS_DOWN must fire for downward motion")

    def test_dsgc_preferred_channel_dominates_its_null(self):
        """
        Opposite channels are not just ordered but separated: at the same edge the null
        channel must stay an order of magnitude below the preferred one.
        """
        n = 256
        bar = torch.zeros(1, 3, n, n)
        bar[:, :, :, 100:120] = 1.0
        moved = torch.zeros_like(bar)
        moved[:, :, :, 104:124] = 1.0           # 4 px right

        self.frontend.reset()
        _ = self.frontend(bar)
        out = self.frontend(moved)
        preferred = torch.quantile(out[0, 6].flatten(), 0.99).item()
        null = torch.quantile(out[0, 7].flatten(), 0.99).item()
        self.assertGreater(preferred, 10.0 * null,
                           f"DS_RIGHT p99 {preferred:.4f} should dominate DS_LEFT p99 {null:.4f}")

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

    def test_looming_is_not_fooled_by_translation(self):
        """
        A dark object of *constant size* moving across the frame is not approaching the
        observer, so it must not read as looming. Only a genuine expansion should.

        The previous implementation scored a translating disc, an approaching disc and a
        truly expanding one identically (0.166 in all three cases).
        """
        h, w = 256, 256
        cy, cx = h // 2, w // 2

        def dark_disc(center_y, center_x, radius):
            f = torch.ones(1, 3, h, w)
            f[:, :, center_y - radius:center_y + radius, center_x - radius:center_x + radius] = 0.0
            return f

        def looming_of(f1, f2):
            self.frontend.reset()
            _ = self.frontend(f1)
            return self.frontend(f2)[:, 11].max().item()

        expanding = looming_of(dark_disc(cy, cx, 10), dark_disc(cy, cx, 30))
        translating = looming_of(dark_disc(cy, cx - 60, 20), dark_disc(cy, cx + 60, 20))
        approaching = looming_of(dark_disc(cy, cx - 60, 15), dark_disc(cy, cx - 10, 15))

        self.assertGreater(expanding, 0.01, "expansion must fire")
        self.assertLess(translating, 0.2 * expanding,
                        f"translation ({translating:.4f}) must not read as looming (expansion {expanding:.4f})")
        self.assertLess(approaching, 0.2 * expanding,
                        f"constant-size approach ({approaching:.4f}) must not read as looming")

    # ------------------------------------------------------------------
    # 18. No global-luminance false positives
    # ------------------------------------------------------------------
    def test_global_luminance_change_drives_no_motion_or_colour_channel(self):
        """
        A spatially uniform brightening is not motion and is not colour: every chromatic,
        DSGC and looming channel must stay at exactly zero.

        Previously a flat grey frame leaked 59% of its luminance into CHROMA_BY, and a
        uniform brightening drove all four DSGCs to 0.20.
        """
        n = 128
        dim = torch.full((1, 3, n, n), 0.2)
        bright = torch.full((1, 3, n, n), 0.7)

        self.frontend.reset()
        _ = self.frontend(dim)
        out_up = self.frontend(bright)

        self.frontend.reset()
        _ = self.frontend(bright)
        out_down = self.frontend(dim)

        for name, out in (("brightening", out_up), ("darkening", out_down)):
            # channels 1-4: chroma + subunits; 6-11: DSGC/OMS/looming.
            # A tiny tolerance covers fp32 round-off in the opponent-matrix einsum and
            # in the replicate-padded blurs; the pre-fix leakage was 0.2-0.5, i.e. 4+
            # orders of magnitude larger.
            for ch in (1, 2, 3, 4, 6, 7, 8, 9, 11):
                self.assertLess(
                    out[0, ch].abs().max().item(), 1e-5,
                    f"{DRFBiomimeticFrontEnd.CHANNEL_NAMES[ch]} fired on a uniform {name}",
                )

    def test_chroma_axes_are_zero_sum(self):
        """Both opponent rows must sum to zero, so an achromatic input cancels exactly."""
        row_sums = self.frontend.opponency_matrix.sum(dim=1)
        self.assertAlmostEqual(row_sums[0].item(), 1.0, places=5)   # luminance weights
        self.assertAlmostEqual(row_sums[1].item(), 0.0, places=5)
        self.assertAlmostEqual(row_sums[2].item(), 0.0, places=5)

        # and grey maps to zero in both opponent channels
        n = 128
        for v in (0.15, 0.5, 0.9):
            self.frontend.reset()
            out = self.frontend(torch.full((1, 3, n, n), v))
            self.assertAlmostEqual(out[0, 1].abs().max().item(), 0.0, places=6, msg=f"RG at grey {v}")
            self.assertAlmostEqual(out[0, 2].abs().max().item(), 0.0, places=6, msg=f"BY at grey {v}")

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

    # ------------------------------------------------------------------
    # 15. Border-artifact free spatial filtering
    # ------------------------------------------------------------------
    def test_no_spurious_border_response_on_uniform_field(self):
        """
        A spatially uniform frame has zero contrast everywhere, including at the
        frame border. Center-surround/OMS/BlurPool convolutions must replicate-pad
        (not zero-pad) so edge pixels aren't misread as a dark border artifact.
        """
        self.frontend.reset()
        out = self.frontend(torch.full((1, 3, 256, 256), 0.5))
        # SUBUNIT_ON/OFF (Ch 3-4) are high-pass responses: must be ~0 even
        # at the outermost ring of the output grid, where border padding would bite.
        border = torch.cat(
            [out[:, 3:5, 0, :], out[:, 3:5, -1, :], out[:, 3:5, :, 0], out[:, 3:5, :, -1]],
            dim=-1,
        )
        self.assertLess(border.abs().max().item(), 1e-3,
                        "High-pass channels must be ~0 at the border for a uniform field")

    # ------------------------------------------------------------------
    # 10b. Batch growth must not discard history for the samples that have it
    # ------------------------------------------------------------------
    def test_batch_growth_preserves_existing_history(self):
        """
        Growing the streaming batch size must keep the cached history for the leading
        samples and treat only the *new* samples as first frames.

        Previously any growth silently set the whole batch's state to None, so every
        sample reported "no motion" for that step.
        """
        n = 128
        first = torch.rand(1, 3, n, n)
        self.frontend.reset()
        _ = self.frontend(first)
        self.assertTrue(self.frontend.has_state)

        moved = torch.roll(first, shifts=8, dims=-1)
        newcomers = torch.rand(2, 3, n, n)
        batch = torch.cat([moved, newcomers], dim=0)

        out = self.frontend(batch)
        self.assertEqual(out.shape[0], 3)
        # sample 0 continued the clip -> real motion response
        self.assertGreater(out[0:1, 5].abs().max().item(), 0.01,
                           "sample with history must still report its transient")
        self.assertGreater(out[0:1, 6].max().item(), 0.01,
                           "sample with history must still report its DSGC response")
        # samples 1-2 have no history -> first-frame regime, exactly zero
        self.assertEqual(out[1:, 5:12].abs().max().item(), 0.0,
                         "new samples must behave like first frames, not fabricate motion")

    # ------------------------------------------------------------------
    # 11b. Device/dtype migration must also drop the stale temporal state
    # ------------------------------------------------------------------
    def test_device_migration_drops_stale_state(self):
        """
        `.to()` clears the kernel cache *and* the cached DRFState.

        Keeping the old-dtype state made the next forward pass raise
        "Input type (FloatTensor) and weight type (DoubleTensor) should be the same".
        """
        self.frontend.reset()
        x32 = torch.rand(1, 3, 128, 128)
        _ = self.frontend(x32)
        self.assertTrue(self.frontend.has_state)

        self.frontend.to(torch.float64)
        self.assertFalse(self.frontend.has_state, "stale state must be dropped on .to()")

        out = self.frontend(x32.double())          # must not raise
        self.assertEqual(out.dtype, torch.float64)
        self.assertEqual(out.shape, (1, 12, 128, 128))

        # and the module still streams normally afterwards
        self.assertTrue(self.frontend.has_state)

    # ------------------------------------------------------------------
    # 17b. Clip-constant letterbox fill cannot manufacture motion
    # ------------------------------------------------------------------
    def test_letterbox_fill_is_constant_for_a_clip(self):
        """
        The bar colour is measured once per clip, not per frame: a per-frame border
        mean changes brightness every frame and injects transients/DSGC/looming
        responses into a band that contains no image at all.
        """
        h, w = 180, 320
        self.frontend.reset()
        fill_from_first = None
        for t in range(4):
            frame = torch.rand(1, 3, h, w)
            self.frontend(frame)
            if t == 0:
                fill_from_first = self.frontend._pad_fill.clone()
            else:
                self.assertTrue(torch.equal(self.frontend._pad_fill, fill_from_first),
                                "letterbox fill must stay fixed for the whole clip")

        # Two different frames with the same clip fill must produce identical bars.
        f1 = torch.rand(1, 3, h, w)
        f2 = torch.rand(1, 3, h, w)
        lb1 = DRFBiomimeticFrontEnd._letterbox_to_square(f1, fill=fill_from_first)
        lb2 = DRFBiomimeticFrontEnd._letterbox_to_square(f2, fill=fill_from_first)
        self.assertEqual(lb1.shape[-2:], (w, w))
        self.assertTrue(torch.equal(lb1[..., :70, :], lb2[..., :70, :]),
                        "top bar must be identical across frames")

    def test_no_motion_in_letterbox_band(self):
        """
        A vertical pan on a 16:9 clip must not produce motion responses *inside* the
        letterbox band, up to the spatial reach of the surround kernels.
        """
        h, w = 180, 320
        scene = torch.rand(1, 3, h + 200, w)
        frames = [scene[..., t * 20:t * 20 + h, :].contiguous() for t in range(5)]

        self.frontend.reset()
        outs = [self.frontend(f) for f in frames]
        out = torch.stack(outs, dim=0)             # (T, 1, 12, 128, 128)
        bar_rows = torch.cat([out[1:, :, :, 0:10, :], out[1:, :, :, 118:, :]], dim=3)
        self.assertLess(bar_rows[:, :, 5:12].abs().max().item(), 1e-6,
                        "motion channels must be silent inside the letterbox band")

    # ------------------------------------------------------------------
    # 19. Stride-1 decimation is the identity
    # ------------------------------------------------------------------
    def test_stride_one_decimation_is_identity(self):
        """
        At S=1 nothing is downsampled, so Stage 3 must not low-pass the only resolution
        that is already on the target grid (it used to apply a 5x5 binomial blur).
        """
        fe = DRFBiomimeticFrontEnd()
        kernel, pad = fe._get_anti_aliasing_blur_kernel(1, 3, torch.device("cpu"), torch.float32)
        self.assertEqual(tuple(kernel.shape), (3, 1, 1, 1))
        self.assertEqual(pad, 0)
        self.assertTrue(torch.equal(kernel, torch.ones_like(kernel)))

        activity = torch.rand(2, 12, 128, 128)     # non-negative, 128 == target -> S=1
        self.assertTrue(torch.equal(fe._decimate_and_snap(activity, 1), activity))

    # ------------------------------------------------------------------
    # Constructor validation
    # ------------------------------------------------------------------
    def test_constructor_rejects_invalid_arguments(self):
        for kwargs in (
            dict(alpha=1.0),
            dict(alpha=-0.1),
            dict(beta=-1.0),
            dict(gamma=-1.0),
            dict(sigma_surround_ratio=0.0),
            dict(snap_mode="bogus"),
            dict(target_resolution=(0, 128)),
        ):
            with self.assertRaises(ValueError, msg=f"accepted {kwargs}"):
                DRFBiomimeticFrontEnd(**kwargs)

    # ------------------------------------------------------------------
    # Every snap mode, and the legacy letterbox=False path
    # ------------------------------------------------------------------
    def test_all_snap_modes_produce_the_target_shape(self):
        for snap_mode in ("crop_pad", "adaptive_pool", "interpolate"):
            for letterbox in (True, False):
                fe = DRFBiomimeticFrontEnd(snap_mode=snap_mode, letterbox=letterbox)
                fe.reset()
                out = fe(torch.rand(1, 3, 300, 450))
                self.assertEqual(out.shape, (1, 12, 128, 128), f"{snap_mode}/letterbox={letterbox}")
                self.assertFalse(torch.isnan(out).any(), f"{snap_mode}/letterbox={letterbox}")
                self.assertFalse(torch.isinf(out).any(), f"{snap_mode}/letterbox={letterbox}")

    def test_legacy_no_letterbox_video_path(self):
        """letterbox=False must still stream 5-D clips (no clip fill is computed)."""
        fe = DRFBiomimeticFrontEnd(letterbox=False)
        fe.reset()
        out = fe(torch.rand(2, 3, 3, 180, 320))
        self.assertEqual(out.shape, (2, 3, 12, 128, 128))
        self.assertIsNone(fe._pad_fill)
        self.assertFalse(torch.isnan(out).any())


if __name__ == "__main__":
    unittest.main()
