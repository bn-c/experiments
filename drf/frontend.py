"""
Dynamic Receptive Field Biomimetic Front-End (DRF-FE).

Biologically-inspired, non-spiking temporal-native visual front-end that accepts
arbitrary-resolution RGB streams (e.g., 720p, 1080p, 4K) and produces a fixed-size
(B, 12, 128, 128) feature tensor encoding 12 parallel retinal ganglion pathways.

Key properties:
- Parameter Count: 0 (completely deterministic, analytically parameterized).
- Gradients: None (requires_grad=False). Both forward() and forward_single_frame()
  run under @torch.no_grad().
- State Machine: Dual-regime causal state tracking (Static vs. Temporal video streams).
  State tensors are always detach().clone() to prevent parent-tensor memory leaks.
- Scale-Invariant: Dynamic stride S = max(1, round(min(H, W) / max(target_resolution)))
  and scale-adapted analytical filters. Rounding (rather than flooring) keeps the
  decimated grid within ±1/(2S) of the target on *both* sides, so the residual grid
  snap only ever crops/pads a small fraction of the frame. Flooring made one input
  pixel of size change swing the retained field of view by up to 2x (e.g. 255px ->
  S=1 -> 50% of the frame kept, 256px -> S=2 -> 100%).
- Zero Boundary Artifacts: Replicate-padded directional shifts without circular wrap-around.
- Polarity Preservation: Signed decimation for chromatic and transient bipolar channels.
- Anti-Aliasing Fidelity: Nyquist-scaled BlurPool filter matched to dynamic stride S
  (an identity kernel at S=1, where nothing is downsampled and blur could only soften).
- Fast Execution: Device-aware LRU filter cache; auto-cleared on .to() / .cuda() calls.
- Bounded Range: No DoG center-surround or divisive normalization stage, so every
  channel is a fixed, analytically-bounded function of the [0, 1] RGB input (a single
  Gaussian blur is a convex combination of in-range pixels, so it can't leave [0, 1];
  unlike DoG's difference-of-Gaussians, which has no fixed output range).
- Expansion-Selective Looming: OFF_T center-surround competition suppresses global darkening
  and a frame-wide OFF/ON mass balance suppresses rigid translation, so only expanding
  dark boundaries (genuine collision/approach) survive. Dark-expansion only, per
  Münch et al. 2009: an *approaching bright* object is not detected by this channel.
- Biphasic Transient: Two-timescale temporal kernel (fast excitation - slow inhibitory rebound)
  approximates the biphasic M-pathway bipolar cell response (Baccus & Meister 2002).
- OMS Delayed Surround: |T_{t-1}| used as surround reference for object-motion sensitivity
  matching polyaxonal amacrine cell synaptic delay (Baccus et al. 2008).
- Direction Selectivity: DSGCs use the signed spatiotemporal correlation
  (L_t * -L_x), i.e. the sign of the motion-energy term, rectified per preferred
  direction. A translating edge drives exactly one channel; a spatially uniform
  luminance change contributes nothing because L_x == 0 there, and the spatially
  uniform component of L_t is removed first.
- Raw Chromatic Opponency: RG and BY opponent channels are passed through unfiltered
  (no spatial filtering), unlike the achromatic pathway's single-Gaussian highpass.
  Both opponent axes are constructed zero-sum, so an achromatic (R=G=B) input
  cancels exactly and only genuine chromatic contrast reaches the channels.
- BT.709 Luminance: Updated from BT.601 to ITU-R BT.709/sRGB D65 luminance axis.

References:
- Barlow & Levick, 1965 (DSGC delay-inhibit mechanism)
- Baccus & Meister, 2002 (biphasic M-pathway temporal kernel)
- Ölveczky, Baccus & Meister, 2003 (OMS retinal detection)
- Baccus, Ölveczky, Manu & Meister, 2008 (OMS polyaxonal amacrine delay)
- Münch et al., 2009 (PV5 looming RGC, ON-surround suppression)
- Derrington, Krauskopf & Lennie, 1984 (DKL color opponency)
- Zhang, 2019 (BlurPool / anti-aliased CNNs)
- Dapello et al., 2020 (VOneNet fixed V1 front-end for adversarial robustness)
"""

from typing import Tuple, Dict, Optional, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernels import create_gaussian_kernel_2d, safe_spatial_shift


# ---------------------------------------------------------------------------
# Causal State Buffer
# ---------------------------------------------------------------------------

class DRFState(NamedTuple):
    """
    Causal temporal state buffer.

    All tensors are stored as detached clones to prevent parent-tensor storage
    leaks during long video streaming (fix for Issue #2 from code review).

    Fields:
        y_prev:    (B, 1, H, W) — previous-frame grayscale luminance.
        s_prev:    (B, 1, H, W) — sustained IIR low-pass state.
        abs_t_prev:(B, 1, H, W) — previous-frame |T_t| for OMS delayed surround
                   (Baccus et al. 2008); None on first frame.
    """
    y_prev: torch.Tensor
    s_prev: torch.Tensor
    abs_t_prev: Optional[torch.Tensor]


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class DRFBiomimeticFrontEnd(nn.Module):
    """
    Dynamic Receptive Field Biomimetic Front-End (DRF-FE).

    Transforms arbitrary-resolution RGB input (B, 3, H, W) into a standardised
    (B, 12, 128, 128) biological feature tensor across 12 canonical channels:

      0:  SUSTAINED_LUM  — Stationary luminance / structural forms (IIR sustained)
      1:  CHROMA_RG      — Red vs Green chromatic opponency (L - M), raw (unfiltered), signed
      2:  CHROMA_BY      — Blue vs Yellow opponency (S - (L+M)), raw (unfiltered), signed
      3:  SUBUNIT_ON     — Single-Gaussian highpass ON subunit (achromatic): ReLU(Y - blur(Y))
      4:  SUBUNIT_OFF    — Single-Gaussian highpass OFF subunit (achromatic): ReLU(blur(Y) - Y)
      5:  TRANSIENT      — Magnocellular biphasic temporal contrast, signed
      6:  DS_RIGHT       — ON-OFF Direction-Selective GC, preferred: right (0°)
      7:  DS_LEFT        — ON-OFF Direction-Selective GC, preferred: left (180°)
      8:  DS_UP          — ON-OFF Direction-Selective GC, preferred: up (90°)
      9:  DS_DOWN        — ON-OFF Direction-Selective GC, preferred: down (270°)
      10: DIFF_MOTION    — Object Motion Sensitivity (OMS) with delayed surround
      11: LOOMING        — Expansion-selective approach/collision detector

    Args:
        target_resolution:       Fixed spatial output grid (default: (128, 128)). The
                                 dynamic stride is derived from it, so a smaller grid
                                 down-samples the whole frame instead of crop-zooming.
        alpha:                   Sustained IIR low-pass coefficient (default: 0.8).
        beta:                    DSGC null-path attenuation. Preferred-path responses are
                                 gained by (1 + beta); the null side is rectified away
                                 (default: 0.9).
        gamma:                   OMS surround inhibition gain (default: 1.2).
        sigma_surround_ratio:    Blur sigma ratio (relative to scale) for the achromatic
                                 single-Gaussian highpass, and for the OMS/looming wide
                                 surround geometry (default: 3.0).
        snap_mode:               Grid snap strategy: 'crop_pad', 'adaptive_pool', 'interpolate'.
                                 With the rounded stride the residual is at most ~1/(2S)
                                 of the target, so 'crop_pad' trims/pads only a sliver.
        letterbox:               If True (default), pads the shorter spatial axis to a square
                                 (with a flat, clip-constant bar colour — not a replicated edge,
                                 and not a per-frame statistic) before Stage 0, so non-square
                                 inputs are never center-cropped or anisotropically squashed;
                                 the pipeline's uniform stride S then applies evenly to both
                                 axes. Set False to restore the legacy behaviour of scaling
                                 directly off (H, W) via snap_mode alone.
    """

    CHANNEL_NAMES: Tuple[str, ...] = (
        "SUSTAINED_LUM",
        "CHROMA_RG",
        "CHROMA_BY",
        "SUBUNIT_ON",
        "SUBUNIT_OFF",
        "TRANSIENT",
        "DS_RIGHT",
        "DS_LEFT",
        "DS_UP",
        "DS_DOWN",
        "DIFF_MOTION",
        "LOOMING",
    )

    def __init__(
        self,
        target_resolution: Tuple[int, int] = (128, 128),
        alpha: float = 0.8,
        beta: float = 0.9,
        gamma: float = 1.2,
        sigma_surround_ratio: float = 3.0,
        snap_mode: str = "crop_pad",
        letterbox: bool = True,
    ):
        super().__init__()
        target_h, target_w = int(target_resolution[0]), int(target_resolution[1])
        if target_h <= 0 or target_w <= 0:
            raise ValueError(f"target_resolution must be positive, got {target_resolution!r}")
        if not 0.0 <= float(alpha) < 1.0:
            raise ValueError(f"alpha must be in [0, 1) for a stable IIR filter, got {alpha!r}")
        if float(beta) < 0.0:
            raise ValueError(f"beta must be non-negative, got {beta!r}")
        if float(gamma) < 0.0:
            raise ValueError(f"gamma must be non-negative, got {gamma!r}")
        if float(sigma_surround_ratio) <= 0.0:
            raise ValueError(f"sigma_surround_ratio must be positive, got {sigma_surround_ratio!r}")
        if snap_mode not in ("crop_pad", "adaptive_pool", "interpolate"):
            raise ValueError(
                f"Unknown snap_mode {snap_mode!r}. "
                "Use 'crop_pad', 'adaptive_pool', or 'interpolate'."
            )

        self.target_resolution = (target_h, target_w)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.sigma_surround_ratio = float(sigma_surround_ratio)
        self.snap_mode = snap_mode
        self.letterbox = bool(letterbox)

        # ------------------------------------------------------------------
        # Color Opponency Matrix: [Y, RG, BY]^T = M @ [R, G, B]^T
        # Updated from BT.601 broadcast luma to BT.709/sRGB D65 luminance axis,
        # matching the ITU-R standard for linear sRGB and more closely aligning
        # with the photopic luminosity function (Derrington et al. 1984).
        #
        # Both opponent rows are made zero-sum (and unit-norm) from the raw
        # Stockman-Sharpe-style weights: an opponent axis must cancel for an
        # achromatic input, otherwise a flat grey frame leaks ~59% of its
        # luminance straight into CHROMA_BY, turning that channel into a
        # near-duplicate of SUSTAINED_LUM.
        # ------------------------------------------------------------------
        y_axis = torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float32)
        rg_axis = torch.tensor([0.6275, -0.6863, 0.0588], dtype=torch.float32)
        by_axis = torch.tensor([-0.1576, -0.1553, 0.9021], dtype=torch.float32)

        def _opponent(axis: torch.Tensor) -> torch.Tensor:
            """Remove the luminance component, then normalise to unit L2 norm."""
            axis = axis - axis.sum() * y_axis
            return axis / axis.norm().clamp_min(1e-8)

        opponency_matrix = torch.stack([y_axis, _opponent(rg_axis), _opponent(by_axis)])
        self.register_buffer("opponency_matrix", opponency_matrix)

        # Bipolar signed channel mask: Ch 1 (RG), Ch 2 (BY), Ch 5 (TRANSIENT) are signed.
        # Used in Stage 3 to preserve polarity through decimation.
        is_bipolar = torch.tensor(
            [False, True, True, False, False, True, False, False, False, False, False, False],
            dtype=torch.bool
        ).view(1, 12, 1, 1)
        self.register_buffer("is_bipolar", is_bipolar)

        # Causal temporal state buffer
        self._state: Optional[DRFState] = None

        # Letterbox fill colour, computed once per clip from the first frame's
        # border strips and reused for every later frame. Recomputing it per frame
        # (from each frame's own border) makes the synthetic bars change brightness
        # every frame, which injects transients/DSGC/looming responses into a region
        # that contains no image at all.
        self._pad_fill: Optional[torch.Tensor] = None

        # Device-aware filter kernel cache.
        # IMPORTANT: cleared on .to() / .cuda() via _apply() override (Issue #3 fix).
        self._kernel_cache: Dict[str, Tuple[torch.Tensor, int]] = {}

    # ------------------------------------------------------------------
    # Device migration hook — clears cache so kernels are rebuilt on
    # the new device. Fixes Issue #3: tensor device mismatch on .cuda().
    # The cached temporal state is dropped too: it holds tensors in the *old*
    # dtype/device, and a forward pass with the migrated input would otherwise
    # raise a RuntimeError from mixing them.
    # ------------------------------------------------------------------
    def _apply(self, fn):
        super()._apply(fn)
        self._kernel_cache.clear()
        self._state = None
        self._pad_fill = None
        return self

    # ------------------------------------------------------------------
    # Public state management API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clears causal temporal state (and the clip's letterbox fill) for a new clip."""
        self._state = None
        self._pad_fill = None

    def reset_state(self) -> None:
        """Alias for reset()."""
        self.reset()

    def clear_cache(self) -> None:
        """Clears precomputed filter kernel cache (useful after manual device changes)."""
        self._kernel_cache.clear()

    @property
    def has_state(self) -> bool:
        """True if a temporal history frame is cached in the state buffer."""
        return self._state is not None

    def __repr__(self) -> str:
        ch = ", ".join(f"{i}:{n}" for i, n in enumerate(self.CHANNEL_NAMES))
        return (
            f"DRFBiomimeticFrontEnd("
            f"target={self.target_resolution}, alpha={self.alpha}, beta={self.beta}, "
            f"gamma={self.gamma}, sigma_surround_ratio={self.sigma_surround_ratio}, "
            f"snap_mode={self.snap_mode!r}, letterbox={self.letterbox})\n"
            f"  Channels: [{ch}]"
        )

    # ------------------------------------------------------------------
    # Stage 0: Dynamic Receptive Field Geometry Adapter
    # ------------------------------------------------------------------

    def compute_geometry(self, h: int, w: int) -> Tuple[int, float, float, int, float, int]:
        """
        Computes the dynamic scale factor S and all filter parameters for the given input resolution.

        The stride is derived from `self.target_resolution`, not from a fixed 128,
        so a non-default output grid still down-samples the whole frame instead of
        center-cropping most of it away. It is *rounded* to the nearest integer scale:
        the decimated grid then lands within +-1/(2S) of the target on both sides, so
        the residual grid snap only ever trims/pads a small fraction of the frame.
        Flooring let a single pixel of input size change swing the retained field of
        view from 50% to 100%.

        Returns:
            s:          Stride S = max(1, round(min(H, W) / max(target_resolution))).
            sigma_c:    Achromatic center-Gaussian sigma (single-Gaussian highpass).
            sigma_s:    Achromatic surround sigma, drives the kernel footprint.
            k_center:   Odd kernel size for the center Gaussian.
            sigma_wide: Wide Gaussian sigma for OMS/looming surround (2 x sigma_s).
            k_wide:     Odd kernel size for wide Gaussian.
        """
        target = max(self.target_resolution)
        min_dim = min(h, w)
        s = max(1, int(math.floor(min_dim / float(target) + 0.5)))

        scale_ratio = s / 8.0
        sigma_c = max(0.5, 1.0 * scale_ratio)
        sigma_s = max(1.2, self.sigma_surround_ratio * scale_ratio)

        k_center = 2 * int(math.ceil(3.0 * sigma_s)) + 1
        k_center = max(3, k_center)

        sigma_wide = max(2.0, 2.0 * sigma_s)
        k_wide = 2 * int(math.ceil(3.0 * sigma_wide)) + 1
        k_wide = max(5, k_wide)

        return s, sigma_c, sigma_s, k_center, sigma_wide, k_wide

    @staticmethod
    def _border_fill(x: torch.Tensor) -> torch.Tensor:
        """
        Mean colour of the frame's four border strips, shape (B, C, 1, 1).

        Used as the letterbox bar colour for a whole clip. It is a flat fill rather
        than a replicated edge row/column (which would stripe a textured border into
        streaks), and it is computed once per clip rather than per frame (a per-frame
        fill changes brightness every frame, manufacturing transients and DSGC
        responses inside a region that contains no image).
        """
        h, w = x.shape[-2:]
        bh, bw = min(8, h), min(8, w)
        strips = [
            x[..., :bh, :].flatten(2),
            x[..., -bh:, :].flatten(2),
            x[..., :, :bw].flatten(2),
            x[..., :, -bw:].flatten(2),
        ]
        return torch.cat(strips, dim=-1).mean(dim=-1).view(x.shape[0], x.shape[1], 1, 1)

    @staticmethod
    def _letterbox_to_square(x: torch.Tensor, fill: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Pads the shorter spatial axis so (H, W) -> (max(H, W), max(H, W)).

        Run before Stage 0 so the pipeline's single isotropic stride S applies evenly
        to both axes downstream: non-square inputs are never center-cropped (losing
        peripheral content) nor anisotropically squashed (distorting shape/motion).
        Bars are filled with a flat colour rather than replicate-padded: replicating a
        textured edge row/column down the whole bar would stripe that texture into
        streaks instead of a flat letterbox bar.

        Args:
            x:    (B, C, H, W) input frame.
            fill: optional (B, C, 1, 1) bar colour, normally a single per-clip value
                  from `_border_fill` (see `forward_single_frame`). When omitted, the
                  fill is derived from *this* frame's own border strips, which is
                  only appropriate for one-shot still images.
        """
        h, w = x.shape[-2:]
        if h == w:
            return x
        if fill is not None:
            fill = fill.to(device=x.device, dtype=x.dtype)
        side = max(h, w)
        pad_h, pad_w = side - h, side - w
        if pad_h > 0:
            top, bottom = pad_h // 2, pad_h - pad_h // 2
            if fill is not None:
                top_fill = fill.expand(-1, -1, top, w)
                bottom_fill = fill.expand(-1, -1, bottom, w)
            else:
                border = min(8, h)
                top_fill = x[..., :border, :].mean(dim=(-2, -1), keepdim=True).expand(-1, -1, top, w)
                bottom_fill = x[..., -border:, :].mean(dim=(-2, -1), keepdim=True).expand(-1, -1, bottom, w)
            x = torch.cat([top_fill, x, bottom_fill], dim=-2)
        if pad_w > 0:
            left, right = pad_w // 2, pad_w - pad_w // 2
            if fill is not None:
                left_fill = fill.expand(-1, -1, x.shape[-2], left)
                right_fill = fill.expand(-1, -1, x.shape[-2], right)
            else:
                border = min(8, w)
                left_fill = x[..., :, :border].mean(dim=(-2, -1), keepdim=True).expand(-1, -1, x.shape[-2], left)
                right_fill = x[..., :, -border:].mean(dim=(-2, -1), keepdim=True).expand(-1, -1, x.shape[-2], right)
            x = torch.cat([left_fill, x, right_fill], dim=-1)
        return x

    # ------------------------------------------------------------------
    # Kernel cache accessors
    # ------------------------------------------------------------------

    def _get_wide_gaussian_kernel(
        self,
        k_wide: int,
        sigma_wide: float,
        device: torch.device,
        dtype: torch.dtype
    ) -> Tuple[torch.Tensor, int]:
        """Retrieves or creates cached wide Gaussian kernel (OMS surround)."""
        key = f"gauss_{k_wide}_{sigma_wide:.4f}_{device}_{dtype}"
        if key not in self._kernel_cache:
            g = create_gaussian_kernel_2d(k_wide, sigma_wide, device=device, dtype=dtype)
            self._kernel_cache[key] = (g, k_wide // 2)
        return self._kernel_cache[key]

    def _get_anti_aliasing_blur_kernel(
        self,
        stride_s: int,
        channels: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> Tuple[torch.Tensor, int]:
        """
        Retrieves or creates the Nyquist-scaled BlurPool kernel for Stage 3 decimation.

        S = 1 strides the identity: nothing is down-sampled, so any blur here could only
        soften the one resolution that is already at the target grid. S = 2 uses the 5x5
        Binomial filter (Zhang 2019); larger strides use a Gaussian BlurPool scaled to
        sigma = 0.4 * S to match the Nyquist criterion.
        """
        key = f"blur_s{stride_s}_c{channels}_{device}_{dtype}"
        if key not in self._kernel_cache:
            if stride_s <= 1:
                kernel = torch.ones(channels, 1, 1, 1, device=device, dtype=dtype)
                k_blur = 1
            elif stride_s <= 2:
                a = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=device, dtype=dtype)
                k2d = torch.outer(a, a) / 256.0
                k_blur = 5
                kernel = k2d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
            else:
                sigma_blur = max(1.0, 0.4 * stride_s)
                k_blur = 2 * int(math.ceil(3.0 * sigma_blur)) + 1
                g = create_gaussian_kernel_2d(k_blur, sigma_blur, device=device, dtype=dtype)
                kernel = g.repeat(channels, 1, 1, 1)
            self._kernel_cache[key] = (kernel, k_blur // 2)
        return self._kernel_cache[key]

    # ------------------------------------------------------------------
    # Stage 1: Photoreceptor & Chromatic Transduction
    # ------------------------------------------------------------------

    def _photoreceptor_transduction(
        self, x_rgb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Stage 1: Log dynamic range compression + BT.709 cone-opponent decomposition.

        1. I_log = ln(1 + 255 * I_RGB) / ln(256)     (Weber-Fechner compression)
        2. [Y, RG, BY]^T = M_opponency @ [R, G, B]^T  (BT.709 / DKL-inspired)
        """
        x_clamped = torch.clamp(x_rgb, 0.0, 1.0)
        i_log = torch.log1p(255.0 * x_clamped) / math.log(256.0)

        m = self.opponency_matrix.to(dtype=i_log.dtype, device=i_log.device)
        opp = torch.einsum("oc,bchw->bohw", m, i_log)

        y  = opp[:, 0:1, :, :]   # Luminance Y (L+M)
        rg = opp[:, 1:2, :, :]   # Red-Green (L-M)
        by = opp[:, 2:3, :, :]   # Blue-Yellow (S-(L+M))
        return y, rg, by

    # ------------------------------------------------------------------
    # Stage 2A: Spatial Form / Color Pathway
    # ------------------------------------------------------------------

    def _spatial_decomposition(
        self,
        y: torch.Tensor,
        rg: torch.Tensor,
        by: torch.Tensor,
        sigma_c: float,
        k_center: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Stage 2A: Single-Gaussian highpass for the achromatic pathway; chromatic
        pathway passed through unfiltered.

        Achromatic (M-pathway bipolars):
            blur(Y)     = Gauss(sigma_c) * Y     (a convex combination of Y, so
                          bounded in Y's own range — no DoG surround subtraction)
            SUBUNIT_ON  = ReLU( Y - blur(Y) )
            SUBUNIT_OFF = ReLU( blur(Y) - Y )
            Both bounded in [0, 1] when Y is bounded in [0, 1], since each term
            of the difference is itself in [0, 1].

        Chromatic (P-pathway midget bipolars):
            RG/BY passed through unfiltered — no spatial filtering stage.

        Returns:
            sub_on:  Achromatic ON subunit, bounded in [0, 1].
            sub_off: Achromatic OFF subunit, bounded in [0, 1].
            rg:      Raw (unfiltered) Red-Green channel (signed, bipolar).
            by:      Raw (unfiltered) Blue-Yellow channel (signed, bipolar).
        """
        # Replicate-pad before convolving (conv2d's own padding= is zero-padding)
        # so the frame border isn't read as a false dark edge.
        blur_kernel, pad = self._get_wide_gaussian_kernel(k_center, sigma_c, y.device, y.dtype)
        blurred_y = F.conv2d(F.pad(y, (pad, pad, pad, pad), mode="replicate"), blur_kernel)
        sub_on  = F.relu(y - blurred_y)
        sub_off = F.relu(blurred_y - y)

        return sub_on, sub_off, rg, by

    # ------------------------------------------------------------------
    # Stage 2B: Motion, Transient & Surround Inhibition Circuits
    # ------------------------------------------------------------------

    @staticmethod
    def _reconcile_state(state: DRFState, y_curr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Aligns a cached state buffer with the current frame's batch and spatial shape.

        - Resolution change mid-stream: bilinearly resample the cached maps.
        - Batch shrink: keep the leading samples.
        - Batch *growth*: the extra samples have no history, so their y_prev/s_prev are
          seeded with the current frame and their |T_prev| with zeros. That reproduces
          the first-frame regime (zero temporal response) for exactly those samples,
          instead of the previous behaviour of discarding the whole batch's history and
          silently reporting "no motion" for the samples that did have it.
        """
        y_prev = state.y_prev
        s_prev = state.s_prev
        abs_t_prev = state.abs_t_prev
        if abs_t_prev is None:
            abs_t_prev = torch.zeros_like(y_curr)

        cached_b, batch_b = y_prev.shape[0], y_curr.shape[0]
        if cached_b > batch_b:
            y_prev, s_prev = y_prev[:batch_b], s_prev[:batch_b]
            abs_t_prev = abs_t_prev[:batch_b]
        elif cached_b < batch_b:
            # The *new* samples are the trailing ones (indices cached_b..batch_b).
            new_frames = y_curr[cached_b:]
            y_prev = torch.cat([y_prev, new_frames], dim=0)
            s_prev = torch.cat([s_prev, new_frames], dim=0)
            abs_t_prev = torch.cat([abs_t_prev, torch.zeros_like(new_frames)], dim=0)

        if y_prev.shape[-2:] != y_curr.shape[-2:]:
            size = y_curr.shape[-2:]
            y_prev = F.interpolate(y_prev, size=size, mode="bilinear", align_corners=False)
            s_prev = F.interpolate(s_prev, size=size, mode="bilinear", align_corners=False)
            abs_t_prev = F.interpolate(abs_t_prev, size=size, mode="bilinear", align_corners=False)

        return y_prev, s_prev, abs_t_prev

    def _motion_decomposition(
        self,
        y_curr: torch.Tensor,
        state: Optional[DRFState],
        sigma_wide: float,
        k_wide: int
    ) -> Tuple[
        torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor,
        DRFState
    ]:
        """
        Stage 2B: Motion, Transient & Surround Inhibition.

        Implements (temporally):
        - Sustained stream (IIR low-pass, Parvocellular sustained bipolars).
        - Biphasic transient (two-timescale difference approximating M-pathway
          bipolar temporal kernel; Baccus & Meister 2002).
        - Direction-Selective Ganglion Cells: signed spatiotemporal correlation
          L_t * (-L_x), the local sign of the motion-energy term, rectified per
          preferred direction (see `_directional_correlations`).
        - OMS with delayed surround: |T_{t-1}| as surround reference matching
          polyaxonal amacrine cell ~1-frame synaptic delay (Baccus et al. 2008).
        - Looming via OFF_T center-advantage over its wide surround (suppresses
          global darkening) multiplied by the frame-wide OFF/ON mass balance
          (suppresses rigid translation), i.e. expansion only (Münch et al. 2009).

        Returns 8 feature maps + the updated DRFState.
        """
        is_first_frame = (state is None)

        if is_first_frame:
            # ----- Dual-Regime t=0: Static / First Frame -----
            s_curr = y_curr

            t_curr     = torch.zeros_like(y_curr)
            ds_right   = torch.zeros_like(y_curr)
            ds_left    = torch.zeros_like(y_curr)
            ds_up      = torch.zeros_like(y_curr)
            ds_down    = torch.zeros_like(y_curr)
            diff_motion = torch.zeros_like(y_curr)
            looming    = torch.zeros_like(y_curr)

            # abs_t_prev initialised to zero for first OMS delay slot
            abs_t_curr = torch.zeros_like(y_curr)

        else:
            y_prev, s_prev, abs_t_prev = self._reconcile_state(state, y_curr)

            # ---- Sustained Stream (IIR low-pass, Parvocellular tonic bipolars) ----
            s_curr = self.alpha * s_prev + (1.0 - self.alpha) * y_curr

            # ---- Biphasic Transient (Baccus & Meister 2002) ----
            # Real M-pathway bipolars have a biphasic temporal kernel:
            # fast excitatory lobe (Y_t - Y_{t-1}) followed by a slow inhibitory
            # rebound (~40-60 ms lag). s_prev approximates the slow rebound reference.
            t_fast = y_curr - y_prev
            t_slow = 0.5 * (y_curr - s_prev)     # delayed inhibitory rebound proxy
            t_curr = t_fast - 0.4 * t_slow        # biphasic response

            # ---- DSGC: signed spatiotemporal correlation (Barlow & Levick 1965) ----
            corr_across, corr_along = self._directional_correlations(y_curr, t_fast)
            # Each channel rectifies the correlation that matches its preferred
            # direction; beta is the null-path attenuation, so the preferred side is
            # boosted to (1 + beta) while the null side stays rectified away.
            gain = 1.0 + self.beta
            ds_right = gain * F.relu(-corr_across)   # preferred direction: +x
            ds_left  = gain * F.relu(corr_across)    # preferred direction: -x
            ds_down  = gain * F.relu(-corr_along)    # preferred direction: +y (row index)
            ds_up    = gain * F.relu(corr_along)     # preferred direction: -y

            # ---- OMS with Delayed Surround (Baccus et al. 2008) ----
            # Polyaxonal amacrine cells pool |T| over a wide spatial area with
            # ~1-frame synaptic delay. Using abs_t_prev as the surround reference
            # sharpens motion-onset selectivity vs. sustained global camera pan.
            abs_t_curr = torch.abs(t_curr)
            wide_kernel, pad_wide = self._get_wide_gaussian_kernel(k_wide, sigma_wide, y_curr.device, y_curr.dtype)
            abs_t_prev_padded = F.pad(abs_t_prev, (pad_wide,) * 4, mode="replicate")
            surround_motion = F.conv2d(abs_t_prev_padded, wide_kernel)
            diff_motion = F.relu(abs_t_curr - self.gamma * surround_motion)

            # ---- Looming: Expansion-selective OFF_T (Münch et al. 2009) ----
            off_transient = F.relu(-t_curr)
            on_transient  = F.relu(t_curr)
            off_transient_padded = F.pad(off_transient, (pad_wide,) * 4, mode="replicate")
            off_surround  = F.conv2d(off_transient_padded, wide_kernel)
            # Center-advantage: local darkening must exceed what the wide surround sees.
            # Suppresses global uniform darkening; preserves local darkening edges.
            center_adv = F.relu(off_transient - off_surround)
            looming    = F.max_pool2d(center_adv, kernel_size=5, stride=1, padding=2) * center_adv
            # Expansion vs. translation. An expanding dark region darkens its whole
            # boundary (net darkening); a rigidly translating one darkens the leading
            # side while brightening the trailing one (net ~0). No local operator can
            # separate the two for an object larger than its kernel, so this uses the
            # frame-wide OFF/ON mass balance as a per-sample expansion gate.
            off_mass = off_transient.mean(dim=(1, 2, 3), keepdim=True)
            on_mass  = on_transient.mean(dim=(1, 2, 3), keepdim=True)
            dark_balance = (off_mass - on_mass) / (off_mass + on_mass + 1e-6)
            looming = looming * F.relu(dark_balance)

        # FIX (Issue #2): detach().clone() breaks reference to parent tensor storage,
        # preventing memory leaks during long (B, T, 3, H, W) video sequences.
        new_state = DRFState(
            y_prev     = y_curr.detach().clone(),
            s_prev     = s_curr.detach().clone(),
            abs_t_prev = abs_t_curr.detach().clone(),
        )

        return s_curr, t_curr, ds_right, ds_left, ds_up, ds_down, diff_motion, looming, new_state

    @staticmethod
    def _directional_correlations(
        y_curr: torch.Tensor, delta_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Signed spatiotemporal correlations that carry the direction signal.

        For a luminance field advected by velocity v, the temporal derivative obeys
        the brightness-constancy constraint L_t = -(v . grad L). Hence
        `L_t * (-dL/dx) = v_x * (dL/dx)^2 >= 0` for +x motion and < 0 for -x motion,
        for both bright-on-dark and dark-on-bright edges (the sign of the gradient
        cancels out) — this is the sign of the classic motion-energy term.

        Two properties follow, and both were violated by the previous
        `relu(y_curr - beta * shift(y_prev))` formulation, which sampled the null side
        and therefore fired on *any* newly-occupied pixel: a translating edge drove all
        four channels equally, and a spatially uniform luminance change drove all four
        as well.

        Here a spatially uniform change contributes nothing, for two independent
        reasons: the spatial gradient of a uniform field is zero, and the spatially
        uniform component of L_t is removed before the correlation (so a global
        exposure change cannot masquerade as motion). The central-difference taps are
        +-0.5 and |L_t| <= 1 for inputs in [0, 1], so the correlation is bounded by
        ~1.0 and each rectified channel by (1 + beta) times that.

        Caveat: like any 1-px-stencil motion-energy model, this assumes a small
        inter-frame displacement. A rigid jump of tens of pixels per frame aliases the
        step edge, and the null channel then picks up a one-pixel-wide residue at the
        trailing edge; at realistic displacements (a few px) the orthogonal/null
        channels are exactly zero.

        Returns:
            corr_across: (B, 1, H, W), > 0 for leftward motion, < 0 for rightward.
            corr_along:  (B, 1, H, W), > 0 for upward motion, < 0 for downward.
        """
        # Remove the frame-wide mean of L_t: a global brightening/darkening is a
        # spatially constant temporal derivative and must not read as motion.
        delta_ac = delta_t - delta_t.mean(dim=(-2, -1), keepdim=True)
        grad_x = 0.5 * (safe_spatial_shift(y_curr, dx=-1, dy=0) - safe_spatial_shift(y_curr, dx=1, dy=0))
        grad_y = 0.5 * (safe_spatial_shift(y_curr, dx=0, dy=-1) - safe_spatial_shift(y_curr, dx=0, dy=1))
        return delta_ac * grad_x, delta_ac * grad_y

    # ------------------------------------------------------------------
    # Stage 3: Anti-Aliased Spatial Decimation & Grid Snap
    # ------------------------------------------------------------------

    def _decimate_and_snap(self, channels: torch.Tensor, stride_s: int) -> torch.Tensor:
        """
        Stage 3: Anti-aliased decimation (BlurPool, Zhang 2019) + adaptive grid snap.

        1. Rectify unipolar channels (magnitude); preserve the sign of bipolar channels.
        2. Nyquist-scaled depthwise BlurPool convolution with stride S.
        3. Center crop or symmetric replicate-pad to exact target_resolution.
        """
        # Polarity handling: bipolar channels (RG, BY, TRANSIENT) keep their sign,
        # everything else is rectified. This runs under no_grad and the front-end has
        # no parameters, so there is no sqrt-at-zero gradient to guard: a clamp would
        # only add a spurious 1e-6 floor to channels that are exactly zero (all seven
        # motion channels on a still frame).
        is_bipolar = self.is_bipolar.to(device=channels.device)
        energy = torch.where(is_bipolar, channels, channels.abs())

        # Nyquist-scaled BlurPool decimation. Replicate-pad (matching Zhang 2019's
        # reflect-padded reference) instead of conv2d's zero-padding to avoid
        # darkening the outermost decimated pixels toward the frame border.
        B, C, H, W = energy.shape
        blur_weight, pad = self._get_anti_aliasing_blur_kernel(stride_s, C, energy.device, energy.dtype)
        energy_padded = F.pad(energy, (pad, pad, pad, pad), mode="replicate")
        decimated = F.conv2d(energy_padded, blur_weight, stride=stride_s, groups=C)

        # Grid snap to target resolution
        target_h, target_w = self.target_resolution
        if self.snap_mode == "crop_pad":
            return self._spatial_crop_pad(decimated, target_h, target_w)
        elif self.snap_mode == "adaptive_pool":
            return F.adaptive_avg_pool2d(decimated, (target_h, target_w))
        elif self.snap_mode == "interpolate":
            return F.interpolate(decimated, size=(target_h, target_w), mode="bilinear", align_corners=False)
        else:
            raise ValueError(
                f"Unknown snap_mode '{self.snap_mode}'. "
                f"Use 'crop_pad', 'adaptive_pool', or 'interpolate'."
            )

    @staticmethod
    def _spatial_crop_pad(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """Center-crops or symmetrically replicate-pads to exactly (target_h, target_w)."""
        _, _, h, w = x.shape

        if h > target_h:
            top = (h - target_h) // 2
            x = x[:, :, top:top + target_h, :]
        elif h < target_h:
            ph  = target_h - h
            pt  = ph // 2
            pb  = ph - pt
            x = F.pad(x, (0, 0, pt, pb), mode="replicate")

        if w > target_w:
            left = (w - target_w) // 2
            x = x[:, :, :, left:left + target_w]
        elif w < target_w:
            pw   = target_w - w
            pl   = pw // 2
            pr   = pw - pl
            x = F.pad(x, (pl, pr, 0, 0), mode="replicate")

        return x

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_single_frame(
        self,
        x: torch.Tensor,
        state: Optional[DRFState] = None
    ) -> Tuple[torch.Tensor, DRFState]:
        """
        Processes one RGB frame through all 4 stages.

        FIX (Issue #8): decorated with @torch.no_grad() to match forward().
        Previously, calling this directly would silently build an autograd
        graph through all deterministic kernel operations.

        Args:
            x:     Input tensor (B, 3, H, W), values in [0, 1].
            state: Explicit causal state override; defaults to self._state.

        Returns:
            output:    (B, 12, target_h, target_w) feature tensor.
            new_state: Updated DRFState for the next frame.
        """
        B, C, H, W = x.shape
        if C != 3:
            raise ValueError(f"Expected 3-channel RGB input, got {C} channels.")
        if x.dim() != 4:
            raise ValueError(f"Expected a 4D (B, 3, H, W) frame, got shape {tuple(x.shape)}")

        if self.letterbox:
            # One flat bar colour per clip (not per frame) so the synthetic padding
            # cannot itself change brightness between frames.
            if self._pad_fill is None or self._pad_fill.shape[0] != B:
                self._pad_fill = self._border_fill(x)
            x = self._letterbox_to_square(x, fill=self._pad_fill)
            H, W = x.shape[-2:]

        # State selection: use explicit state if provided, else internal buffer.
        # Batch-size reconciliation (shrink/grow) happens in _reconcile_state, where the
        # current frame is available to seed the history of newly added samples.
        active_state = state if state is not None else self._state

        # Stage 0: Dynamic Receptive Field Geometry
        stride_s, sigma_c, _sigma_s, k_center, sigma_wide, k_wide = self.compute_geometry(H, W)

        # Stage 1: Photoreceptor & Chromatic Transduction
        y, rg, by = self._photoreceptor_transduction(x)

        # Stage 2A: Spatial Form & Chromatic Pathway
        sub_on, sub_off, rg_raw, by_raw = self._spatial_decomposition(
            y, rg, by, sigma_c, k_center
        )

        # Stage 2B: Motion, Transient & Surround Inhibition
        (
            sustained_lum,
            transient,
            ds_right,
            ds_left,
            ds_up,
            ds_down,
            diff_motion,
            looming,
            new_state,
        ) = self._motion_decomposition(y, active_state, sigma_wide, k_wide)

        # Concatenate 12 parallel channels
        channels = torch.cat([
            sustained_lum,   # Ch  0: SUSTAINED_LUM
            rg_raw,          # Ch  1: CHROMA_RG      (signed, raw/unfiltered)
            by_raw,          # Ch  2: CHROMA_BY      (signed, raw/unfiltered)
            sub_on,          # Ch  3: SUBUNIT_ON
            sub_off,         # Ch  4: SUBUNIT_OFF
            transient,       # Ch  5: TRANSIENT       (signed, biphasic)
            ds_right,        # Ch  6: DS_RIGHT
            ds_left,         # Ch  7: DS_LEFT
            ds_up,           # Ch  8: DS_UP
            ds_down,         # Ch  9: DS_DOWN
            diff_motion,     # Ch 10: DIFF_MOTION
            looming,         # Ch 11: LOOMING
        ], dim=1)

        # Stage 3: Anti-Aliased Decimation & Grid Snap
        output = self._decimate_and_snap(channels, stride_s)

        return output, new_state

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[DRFState] = None,
        return_state: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, DRFState]]:
        """
        Unified forward pass for single frames (B, 3, H, W) and video sequences (B, T, 3, H, W).

        Args:
            x:            4D (B, 3, H, W) single frame, or 5D (B, T, 3, H, W) video clip.
            state:        Optional explicit causal DRFState; uses internal self._state if None.
            return_state: If True, returns (features, final_state).

        Returns:
            features:    (B, 12, 128, 128)       for 4D input.
                         (B, T, 12, 128, 128)    for 5D input.
            final_state: (Optional) DRFState.
        """
        if x.dim() == 4:
            out, new_state = self.forward_single_frame(x, state=state)
            if state is None:
                self._state = new_state
            return (out, new_state) if return_state else out

        elif x.dim() == 5:
            B, T, C, H, W = x.shape
            outputs    = []
            curr_state = state if state is not None else self._state

            for t in range(T):
                out_t, curr_state = self.forward_single_frame(x[:, t], state=curr_state)
                outputs.append(out_t)

            if state is None:
                self._state = curr_state

            video_out = torch.stack(outputs, dim=1)   # (B, T, 12, target_h, target_w)
            return (video_out, curr_state) if return_state else video_out

        else:
            raise ValueError(
                f"Expected 4D (B, 3, H, W) or 5D (B, T, 3, H, W) input, got shape {x.shape}"
            )
