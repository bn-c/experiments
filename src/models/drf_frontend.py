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
- Scale-Invariant: Dynamic stride S = floor(min(H, W) / 128) and scale-adapted
  analytical filters. DSGC shift_px = max(1, S // 2) for half-output-pixel sensitivity.
- Zero Boundary Artifacts: Replicate-padded directional shifts without circular wrap-around.
- Polarity Preservation: Signed decimation for chromatic and transient bipolar channels.
- Anti-Aliasing Fidelity: Nyquist-scaled BlurPool filter matched to dynamic stride S.
- Fast Execution: Device-aware LRU filter cache; auto-cleared on .to() / .cuda() calls.
- Correct Normalization: Separate divisive normalization pools for parvocellular (Ch 0-4)
  and magnocellular (Ch 5-11) pathways matching IPL laminar segregation.
- Expansion-Selective Looming: OFF_T center-surround competition suppresses global darkening
  while preserving genuine collision/approach responses.
- Biphasic Transient: Two-timescale temporal kernel (fast excitation - slow inhibitory rebound)
  approximates the biphasic M-pathway bipolar cell response (Baccus & Meister 2002).
- OMS Delayed Surround: |T_{t-1}| used as surround reference for object-motion sensitivity
  matching polyaxonal amacrine cell synaptic delay (Baccus et al. 2008).
- Chromatic DoG: Mild spatial center-surround applied to RG and BY opponent channels,
  matching parvocellular midget bipolar cell spatial filtering.
- BT.709 Luminance: Updated from BT.601 to ITU-R BT.709/sRGB D65 luminance axis.

References:
- Enroth-Cugell & Robson, 1966 (DoG receptive fields)
- Barlow & Levick, 1965 (DSGC delay-inhibit mechanism)
- Baccus & Meister, 2002 (biphasic M-pathway temporal kernel)
- Ölveczky, Baccus & Meister, 2003 (OMS retinal detection)
- Baccus, Ölveczky, Manu & Meister, 2008 (OMS polyaxonal amacrine delay)
- Münch et al., 2009 (PV5 looming RGC, ON-surround suppression)
- Derrington, Krauskopf & Lennie, 1984 (DKL color opponency)
- Heeger, 1992 (divisive normalization)
- Carandini & Heeger, 2012 (canonical neural normalization)
- Zhang, 2019 (BlurPool / anti-aliased CNNs)
- Dapello et al., 2020 (VOneNet fixed V1 front-end for adversarial robustness)
"""

from typing import Tuple, Dict, Optional, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
# Kernel factory helpers
# ---------------------------------------------------------------------------

def create_gaussian_kernel_2d(
    kernel_size: int,
    sigma: float,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Creates a normalised 2D Gaussian kernel of shape (1, 1, K, K)."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    half = kernel_size // 2
    coords = torch.linspace(-half, half, kernel_size, device=device, dtype=dtype or torch.float32)
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    r_sq = x ** 2 + y ** 2
    sigma_sq = max(float(sigma) ** 2, 1e-6)
    kernel = torch.exp(-0.5 * r_sq / sigma_sq)
    kernel = kernel / (kernel.sum() + 1e-8)
    return kernel.unsqueeze(0).unsqueeze(0)


def create_dog_kernel_2d(
    kernel_size: int,
    sigma_center: float,
    sigma_surround: float,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Creates a zero-mean normalised Difference-of-Gaussians kernel (1, 1, K, K)."""
    g_c = create_gaussian_kernel_2d(kernel_size, sigma_center, device=device, dtype=dtype)
    g_s = create_gaussian_kernel_2d(kernel_size, sigma_surround, device=device, dtype=dtype)
    dog = g_c - g_s
    dog = dog - dog.mean()
    norm = torch.sqrt(torch.sum(dog ** 2) + 1e-8)
    return dog / norm


def safe_spatial_shift(
    x: torch.Tensor,
    dx: int,
    dy: int,
    mode: str = "replicate"
) -> torch.Tensor:
    """
    Spatially shifts a tensor by (dx, dy) with replicate padding.
    Avoids circular wrap-around artifacts that torch.roll introduces at frame borders.

    Convention:
        dx > 0: content shifts right  (position x receives value from x - dx).
        dx < 0: content shifts left   (position x receives value from x + |dx|).
        dy > 0: content shifts down   (position y receives value from y - dy).
        dy < 0: content shifts up     (position y receives value from y + |dy|).
    """
    if dx == 0 and dy == 0:
        return x

    pad_l = max(0, dx)
    pad_r = max(0, -dx)
    pad_t = max(0, dy)
    pad_b = max(0, -dy)

    padded = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode=mode)
    h, w = x.shape[-2:]
    start_x = max(0, -dx)
    start_y = max(0, -dy)
    return padded[..., start_y:start_y + h, start_x:start_x + w]


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class DRFBiomimeticFrontEnd(nn.Module):
    """
    Dynamic Receptive Field Biomimetic Front-End (DRF-FE).

    Transforms arbitrary-resolution RGB input (B, 3, H, W) into a standardised
    (B, 12, 128, 128) biological feature tensor across 12 canonical channels:

      0:  SUSTAINED_LUM  — Stationary luminance / structural forms (IIR sustained)
      1:  CHROMA_RG      — Red vs Green chromatic opponency (L - M), DoG filtered, signed
      2:  CHROMA_BY      — Blue vs Yellow opponency (S - (L+M)), DoG filtered, signed
      3:  SUBUNIT_ON     — High-frequency center-surround ON subunit (achromatic)
      4:  SUBUNIT_OFF    — High-frequency center-surround OFF subunit (achromatic)
      5:  TRANSIENT      — Magnocellular biphasic temporal contrast, signed
      6:  DS_RIGHT       — ON-OFF Direction-Selective GC, preferred: right (0°)
      7:  DS_LEFT        — ON-OFF Direction-Selective GC, preferred: left (180°)
      8:  DS_UP          — ON-OFF Direction-Selective GC, preferred: up (90°)
      9:  DS_DOWN        — ON-OFF Direction-Selective GC, preferred: down (270°)
      10: DIFF_MOTION    — Object Motion Sensitivity (OMS) with delayed surround
      11: LOOMING        — Expansion-selective approach/collision detector

    Args:
        target_resolution:       Fixed spatial output grid (default: (128, 128)).
        alpha:                   Sustained IIR low-pass coefficient (default: 0.8).
        beta:                    DSGC delay-inhibit attenuation factor (default: 0.9).
        gamma:                   OMS surround inhibition gain (default: 1.2).
        sigma_0:                 Divisive norm baseline saturation constant (default: 0.1).
        norm_pool_kernel_size:   Local pooling kernel for divisive norm (default: 5).
        sigma_surround_ratio:    Surround/center sigma ratio for achromatic DoG (default: 3.0).
        chroma_dog_scale:        Chromatic DoG sigma relative to achromatic (default: 0.6).
                                 Parvocellular midget bipolars have smaller RFs than M-pathway.
        snap_mode:               Grid snap strategy: 'crop_pad', 'adaptive_pool', 'interpolate'.
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
        sigma_0: float = 0.1,
        norm_pool_kernel_size: int = 5,
        sigma_surround_ratio: float = 3.0,
        chroma_dog_scale: float = 0.6,
        snap_mode: str = "crop_pad",
    ):
        super().__init__()
        self.target_resolution = target_resolution
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.sigma_0 = float(sigma_0)
        self.norm_pool_kernel_size = norm_pool_kernel_size
        self.sigma_surround_ratio = float(sigma_surround_ratio)
        self.chroma_dog_scale = float(chroma_dog_scale)
        self.snap_mode = snap_mode

        # ------------------------------------------------------------------
        # Color Opponency Matrix: [Y, RG, BY]^T = M @ [R, G, B]^T
        # Updated from BT.601 broadcast luma to BT.709/sRGB D65 luminance axis,
        # matching the ITU-R standard for linear sRGB and more closely aligning
        # with the photopic luminosity function (Derrington et al. 1984).
        # ------------------------------------------------------------------
        opponency_matrix = torch.tensor([
            [ 0.2126,  0.7152,  0.0722],   # BT.709 D65 luminance (L+M)
            [ 0.6275, -0.6863,  0.0588],   # Red-Green (L - M), normalised
            [-0.1576, -0.1553,  0.9021],   # Blue-Yellow (S - (L+M)), Stockman-Sharpe basis
        ], dtype=torch.float32)
        self.register_buffer("opponency_matrix", opponency_matrix)

        # Bipolar signed channel mask: Ch 1 (RG), Ch 2 (BY), Ch 5 (TRANSIENT) are signed.
        # Used in Stage 4 to preserve polarity through decimation.
        is_bipolar = torch.tensor(
            [False, True, True, False, False, True, False, False, False, False, False, False],
            dtype=torch.bool
        ).view(1, 12, 1, 1)
        self.register_buffer("is_bipolar", is_bipolar)

        # Causal temporal state buffer
        self._state: Optional[DRFState] = None

        # Device-aware filter kernel cache.
        # IMPORTANT: cleared on .to() / .cuda() via _apply() override (Issue #3 fix).
        self._kernel_cache: Dict[str, Tuple[torch.Tensor, int]] = {}

    # ------------------------------------------------------------------
    # Device migration hook — clears cache so kernels are rebuilt on
    # the new device. Fixes Issue #3: tensor device mismatch on .cuda().
    # ------------------------------------------------------------------
    def _apply(self, fn):
        super()._apply(fn)
        self._kernel_cache.clear()
        return self

    # ------------------------------------------------------------------
    # Public state management API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clears causal temporal state for a new independent video clip."""
        self._state = None

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
            f"gamma={self.gamma}, sigma_0={self.sigma_0}, snap_mode={self.snap_mode!r})\n"
            f"  Channels: [{ch}]"
        )

    # ------------------------------------------------------------------
    # Stage 0: Dynamic Receptive Field Geometry Adapter
    # ------------------------------------------------------------------

    def compute_geometry(self, h: int, w: int) -> Tuple[int, float, float, int, float, int]:
        """
        Computes the dynamic scale factor S and all filter parameters for the given input resolution.

        Returns:
            s:          Stride S = max(1, floor(min(H, W) / 128)).
            sigma_c:    Achromatic DoG center sigma.
            sigma_s:    Achromatic DoG surround sigma.
            k_dog:      Odd kernel size for DoG (covers 3-sigma of surround).
            sigma_wide: Wide Gaussian sigma for OMS surround (2 × sigma_s).
            k_wide:     Odd kernel size for wide Gaussian.
        """
        min_dim = min(h, w)
        s = max(1, min_dim // 128)

        scale_ratio = s / 8.0
        sigma_c = max(0.5, 1.0 * scale_ratio)
        sigma_s = max(1.2, self.sigma_surround_ratio * scale_ratio)

        k_dog = 2 * int(math.ceil(3.0 * sigma_s)) + 1
        k_dog = max(3, k_dog)

        sigma_wide = max(2.0, 2.0 * sigma_s)
        k_wide = 2 * int(math.ceil(3.0 * sigma_wide)) + 1
        k_wide = max(5, k_wide)

        return s, sigma_c, sigma_s, k_dog, sigma_wide, k_wide

    # ------------------------------------------------------------------
    # Kernel cache accessors
    # ------------------------------------------------------------------

    def _get_dog_kernel(
        self,
        k_dog: int,
        sigma_c: float,
        sigma_s: float,
        device: torch.device,
        dtype: torch.dtype,
        tag: str = ""
    ) -> torch.Tensor:
        """Retrieves or creates a cached DoG kernel. `tag` disambiguates chromatic vs achromatic."""
        key = f"dog{tag}_{k_dog}_{sigma_c:.4f}_{sigma_s:.4f}_{device}_{dtype}"
        if key not in self._kernel_cache:
            k = create_dog_kernel_2d(k_dog, sigma_c, sigma_s, device=device, dtype=dtype)
            self._kernel_cache[key] = (k, k_dog // 2)
        return self._kernel_cache[key][0]

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
        Retrieves or creates the Nyquist-scaled BlurPool kernel for Stage 4 decimation.
        Uses the 5×5 Binomial filter for S ≤ 2 (Zhang 2019), and a Gaussian BlurPool
        scaled to sigma = 0.4 * S for larger strides to match the Nyquist criterion.
        """
        key = f"blur_s{stride_s}_c{channels}_{device}_{dtype}"
        if key not in self._kernel_cache:
            if stride_s <= 2:
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
        sigma_s: float,
        k_dog: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Stage 2A: Center-surround DoG filtering for achromatic and chromatic pathways.

        Achromatic (M-pathway bipolars):
            DoG_Y = Gauss(sigma_c) - Gauss(sigma_s)
            SUBUNIT_ON  = ReLU( DoG_Y * Y )
            SUBUNIT_OFF = ReLU( -DoG_Y * Y )

        Chromatic (P-pathway midget bipolars):
            DoG_chroma applied to RG and BY with scaled (smaller) sigmas,
            matching the narrower parvocellular RF bandwidth vs M-pathway.

        Returns:
            sub_on:      Achromatic ON subunit.
            sub_off:     Achromatic OFF subunit.
            rg_filtered: Spatially filtered Red-Green channel (signed, bipolar).
            by_filtered: Spatially filtered Blue-Yellow channel (signed, bipolar).
        """
        # Achromatic DoG
        dog_kernel = self._get_dog_kernel(k_dog, sigma_c, sigma_s, y.device, y.dtype, tag="")
        pad = k_dog // 2
        dog_resp = F.conv2d(y, dog_kernel, padding=pad)
        sub_on  = F.relu(dog_resp)
        sub_off = F.relu(-dog_resp)

        # Chromatic DoG (scaled sigmas for parvocellular midget bipolar bandwidth)
        sc_c = max(0.4, sigma_c * self.chroma_dog_scale)
        sc_s = max(0.8, sigma_s * self.chroma_dog_scale)
        chroma_kernel = self._get_dog_kernel(k_dog, sc_c, sc_s, y.device, y.dtype, tag="_chroma")
        rg_filtered = F.conv2d(rg, chroma_kernel, padding=pad)
        by_filtered = F.conv2d(by, chroma_kernel, padding=pad)

        return sub_on, sub_off, rg_filtered, by_filtered

    # ------------------------------------------------------------------
    # Stage 2B: Motion, Transient & Surround Inhibition Circuits
    # ------------------------------------------------------------------

    def _motion_decomposition(
        self,
        y_curr: torch.Tensor,
        state: Optional[DRFState],
        stride_s: int,
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
        - Direction-Selective Ganglion Cells: Barlow-Levick delay-and-inhibit
          with shift_px = max(1, S // 2) for half-output-pixel sensitivity.
        - OMS with delayed surround: |T_{t-1}| as surround reference matching
          polyaxonal amacrine cell ~1-frame synaptic delay (Baccus et al. 2008).
        - Looming via OFF_T center-advantage over its wide surround: suppresses
          global darkening, preserves genuine spatial expansion (Münch et al. 2009).

        Returns 8 feature maps + the updated DRFState.
        """
        is_first_frame = (state is None)

        if is_first_frame:
            # ----- Dual-Regime t=0: Static / First Frame -----
            y_prev = y_curr
            s_prev = y_curr
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
            y_prev     = state.y_prev
            s_prev     = state.s_prev
            abs_t_prev = state.abs_t_prev  # may be None on very first temporal frame
            if abs_t_prev is None:
                abs_t_prev = torch.zeros_like(y_curr)

            # Adapt spatial dims if resolution changed mid-stream
            if y_prev.shape[-2:] != y_curr.shape[-2:]:
                y_prev     = F.interpolate(y_prev,     size=y_curr.shape[-2:], mode="bilinear", align_corners=False)
                s_prev     = F.interpolate(s_prev,     size=y_curr.shape[-2:], mode="bilinear", align_corners=False)
                abs_t_prev = F.interpolate(abs_t_prev, size=y_curr.shape[-2:], mode="bilinear", align_corners=False)

            # ---- Sustained Stream (IIR low-pass, Parvocellular tonic bipolars) ----
            s_curr = self.alpha * s_prev + (1.0 - self.alpha) * y_curr

            # ---- Biphasic Transient (Baccus & Meister 2002) ----
            # Real M-pathway bipolars have a biphasic temporal kernel:
            # fast excitatory lobe (Y_t - Y_{t-1}) followed by a slow inhibitory
            # rebound (~40-60 ms lag). s_prev approximates the slow rebound reference.
            t_fast = y_curr - y_prev
            t_slow = 0.5 * (y_curr - s_prev)     # delayed inhibitory rebound proxy
            t_curr = t_fast - 0.4 * t_slow        # biphasic response

            # ---- DSGC: Barlow-Levick delay-and-inhibit (Barlow & Levick 1965) ----
            # shift_px = max(1, S // 2) gives approx. half-output-pixel sensitivity.
            # FIX (Issue #1): old round(S/8) collapsed to 1 px at all resolutions.
            shift_px   = max(1, stride_s // 2)
            beta       = self.beta
            motion_gate = torch.tanh(torch.abs(t_curr) * 50.0)

            # Null inhibition is sampled from the null (opposite) side of preferred direction.
            # Right (0°): null side is +x, inhibit from the right.
            y_null_right = safe_spatial_shift(y_prev, dx=-shift_px, dy=0)
            ds_right = F.relu(y_curr - beta * y_null_right) * motion_gate

            # Left (180°): null side is -x.
            y_null_left  = safe_spatial_shift(y_prev, dx=shift_px, dy=0)
            ds_left  = F.relu(y_curr - beta * y_null_left) * motion_gate

            # Up (90°): null side is -y.
            y_null_up    = safe_spatial_shift(y_prev, dx=0, dy=shift_px)
            ds_up    = F.relu(y_curr - beta * y_null_up) * motion_gate

            # Down (270°): null side is +y.
            y_null_down  = safe_spatial_shift(y_prev, dx=0, dy=-shift_px)
            ds_down  = F.relu(y_curr - beta * y_null_down) * motion_gate

            # ---- OMS with Delayed Surround (Baccus et al. 2008) ----
            # Polyaxonal amacrine cells pool |T| over a wide spatial area with
            # ~1-frame synaptic delay. Using abs_t_prev as the surround reference
            # sharpens motion-onset selectivity vs. sustained global camera pan.
            abs_t_curr = torch.abs(t_curr)
            wide_kernel, pad_wide = self._get_wide_gaussian_kernel(k_wide, sigma_wide, y_curr.device, y_curr.dtype)
            surround_motion = F.conv2d(abs_t_prev, wide_kernel, padding=pad_wide)
            diff_motion = F.relu(abs_t_curr - self.gamma * surround_motion)

            # ---- Looming: Expansion-selective OFF_T (Münch et al. 2009) ----
            # Real PV5/approach RGC uses ON-surround suppression to distinguish
            # expanding dark boundaries from global darkening (light-off, shadow).
            # Implementation: center-advantage competition of OFF_T against its
            # own wide surround — fires only when local OFF_T > surround OFF_T.
            off_transient = F.relu(-t_curr)
            off_surround  = F.conv2d(off_transient, wide_kernel, padding=pad_wide)
            # Center-advantage: local darkening must exceed what the wide surround sees.
            # Suppresses global uniform darkening; preserves local expanding edges.
            center_adv = F.relu(off_transient - off_surround)
            looming    = F.max_pool2d(center_adv, kernel_size=5, stride=1, padding=2) * center_adv

            abs_t_curr = abs_t_curr  # already computed above

        # FIX (Issue #2): detach().clone() breaks reference to parent tensor storage,
        # preventing memory leaks during long (B, T, 3, H, W) video sequences.
        new_state = DRFState(
            y_prev     = y_curr.detach().clone(),
            s_prev     = s_curr.detach().clone(),
            abs_t_prev = abs_t_curr.detach().clone(),
        )

        return s_curr, t_curr, ds_right, ds_left, ds_up, ds_down, diff_motion, looming, new_state

    # ------------------------------------------------------------------
    # Stage 3: Divisive Normalization (Inner Plexiform Layer)
    # ------------------------------------------------------------------

    def _normalize_group(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies local divisive normalization within a single functional pathway group
        (Heeger 1992; Carandini & Heeger 2012).

        Formula: Norm_c(x,y) = X_c(x,y) / (sigma_0 + Pool_local(mean_c(|X|^2)))
        """
        k   = self.norm_pool_kernel_size
        pad = k // 2
        # Population energy within this group (mean across channels keeps scale stable)
        pop_sq     = torch.mean(x ** 2, dim=1, keepdim=True)
        local_eng  = F.avg_pool2d(pop_sq, kernel_size=k, stride=1, padding=pad)
        return x / (self.sigma_0 + local_eng)

    def _divisive_normalization(self, channels: torch.Tensor) -> torch.Tensor:
        """
        Stage 3: Separate normalization pools for Parvocellular (Ch 0–4) and
        Magnocellular (Ch 5–11) pathways, matching IPL laminar stratification.

        FIX (Issue #4): previous implementation used a single cross-pathway
        population pool, coupling spatial and motion channel gains.
        """
        spatial  = self._normalize_group(channels[:, :5,  :, :])   # P-pathway
        temporal = self._normalize_group(channels[:, 5:,  :, :])   # M-pathway
        return torch.cat([spatial, temporal], dim=1)

    # ------------------------------------------------------------------
    # Stage 4: Anti-Aliased Spatial Decimation & Grid Snap
    # ------------------------------------------------------------------

    def _decimate_and_snap(self, normalized_features: torch.Tensor, stride_s: int) -> torch.Tensor:
        """
        Stage 4: Anti-aliased decimation (BlurPool, Zhang 2019) + adaptive grid snap.

        1. L2 Energy magnitude for unipolar channels; polarity preserved for bipolar channels.
        2. Nyquist-scaled depthwise BlurPool convolution with stride S.
        3. Center crop or symmetric replicate-pad to exact target_resolution.
        """
        # Energy detection: preserve sign for bipolar channels (RG, BY, TRANSIENT)
        energy_mag = torch.sqrt(torch.clamp(normalized_features ** 2, min=1e-12))
        is_bipolar = self.is_bipolar.to(device=normalized_features.device)
        energy = torch.where(is_bipolar, normalized_features, energy_mag)

        # Nyquist-scaled BlurPool decimation
        B, C, H, W = energy.shape
        blur_weight, pad = self._get_anti_aliasing_blur_kernel(stride_s, C, energy.device, energy.dtype)
        decimated = F.conv2d(energy, blur_weight, stride=stride_s, padding=pad, groups=C)

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
        Processes one RGB frame through all 5 stages.

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

        # State selection: use explicit state if provided, else internal buffer.
        active_state = state if state is not None else self._state

        # Batch-size adaptation (streaming with changing B)
        if active_state is not None:
            cached_b = active_state.y_prev.shape[0]
            if cached_b != B:
                if cached_b > B:
                    # Trim to current batch size
                    abs_t_sliced = (
                        active_state.abs_t_prev[:B] if active_state.abs_t_prev is not None else None
                    )
                    active_state = DRFState(
                        y_prev     = active_state.y_prev[:B],
                        s_prev     = active_state.s_prev[:B],
                        abs_t_prev = abs_t_sliced,
                    )
                else:
                    # New batch is larger than cached — reset to avoid shape errors
                    active_state = None

        # Stage 0: Dynamic Receptive Field Geometry
        stride_s, sigma_c, sigma_s, k_dog, sigma_wide, k_wide = self.compute_geometry(H, W)

        # Stage 1: Photoreceptor & Chromatic Transduction
        y, rg, by = self._photoreceptor_transduction(x)

        # Stage 2A: Spatial Form & Chromatic Pathway
        sub_on, sub_off, rg_filtered, by_filtered = self._spatial_decomposition(
            y, rg, by, sigma_c, sigma_s, k_dog
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
        ) = self._motion_decomposition(y, active_state, stride_s, sigma_wide, k_wide)

        # Concatenate 12 parallel channels
        channels = torch.cat([
            sustained_lum,   # Ch  0: SUSTAINED_LUM
            rg_filtered,     # Ch  1: CHROMA_RG      (signed, DoG-filtered)
            by_filtered,     # Ch  2: CHROMA_BY      (signed, DoG-filtered)
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

        # Stage 3: Separate P/M Divisive Normalization
        normalized = self._divisive_normalization(channels)

        # Stage 4: Anti-Aliased Decimation & Grid Snap
        output = self._decimate_and_snap(normalized, stride_s)

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
