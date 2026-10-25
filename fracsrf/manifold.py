"""
FOVI-style foveated sensing, with FracSRF structured filters mapped onto it.

Prototype for combining the two published ideas in this repo's neighbourhood:

* **FOVI** (Blauch, Alvarez & Konkle, ICML 2026, arXiv:2602.03766) samples visual space
  through a cortical magnification function ``M(r) = k / (r + a)``, lifts the samples onto
  a 3-D cortical sensor manifold, and defines receptive fields as k-nearest-neighbourhoods
  *on the manifold*. Weight sharing is recovered by a kernel mapping that rewrites each
  neighbourhood into a shared reference frame -- see ``fovi/arch/knn.py`` in
  ``github.com/nblauch/fovi`` (``compute_local_rf``), which snaps every neighbour to the
  *nearest* point of a square reference grid.
* **FracSRF** (Saldanha et al., BMVC 2021) represents a kernel analytically, as a single
  weighted separable Gaussian derivative of fractional order, so a kernel costs three
  scalars instead of ``k*k``.

Why they compose
----------------
FOVI's mapping step exists because a learned reference kernel is a *dense grid* and can
only be read at grid points, so sub-grid neighbour positions are rounded away -- the
reason their ``doubleres`` variant (2x reference grid) buys ~3% ImageNet-1K at 4x the
parameters. A FracSRF kernel is a function, not a table: it can be evaluated at the
neighbour positions themselves, which makes the mapping exact and the reference grid
unnecessary. `StructuredKNNConv` implements both, so the difference is measurable
rather than argued.

`mode` selects the arm:

===============  ==================================================  =====================
mode             what is learned                                     mapping
===============  ==================================================  =====================
``dense_ref``    one free tap per reference grid point (FOVI)        one-hot, grid-limited
``structured_ref``  (alpha, nu_x, nu_y) per (out, in), shared sigma  one-hot, grid-limited
``analytic``     same parameters                                     exact, no grid
===============  ==================================================  =====================

``dense_ref`` is the FOVI baseline; ``structured_ref`` and ``analytic`` differ only in
whether the kernel is sampled or evaluated. Nothing here imports `fovi`: the coordinate
spaces, the one-hot mapping and the padding convention are reimplemented to match it
so that the comparison can be run before installing anything.

Coordinates
-----------
Visual positions are normalised to ``[-1, 1]`` (FOVI's convention), so that ``-1``/``1``
is ``fov / 2`` degrees from fixation; pass a `SensorArray` or build one with
`isotropic_sensor`.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .basis import hermite_gaussian_basis, order_coefficients

__all__ = [
    "SensorArray",
    "isotropic_sensor",
    "cortical_coords",
    "knn",
    "neighbourhood_frames",
    "reference_grid",
    "one_hot_mapping",
    "mapping_residual",
    "neighbourhood_anisotropy",
    "neighbourhood_coordinates",
    "StructuredKNNConv",
]

#: CMF scale (mm/deg). Uniform across a sensor, so it cancels in every ratio computed
#: here; kept explicit because the manifold is otherwise unitless.
DEFAULT_CMF_SCALE = 10.0

#: Manifold coordinates are mm-scale and foveal neighbourhoods are tiny, so the
#: default matmul-based cdist loses more precision than the distances being measured
#: (its diagonal is only zero to ~1e-3). Compute differences directly instead.
EXACT_EUCLIDEAN = "donot_use_mm_for_euclid_dist"


# --------------------------------------------------------------------------------------
# Sensor manifold
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SensorArray:
    """
    A foveated sensor: where the samples are in the visual field and on the manifold.

    Attributes:
        visual: ``(n, 2)`` sample positions normalised to ``[-1, 1]``.
        cortical: ``(n, 3)`` positions on the Rovamo-Virsu cortical manifold in mm, laid
            out as cortical Cartesian ``(x_c, y_c, z)`` -- the space FOVI measures
            neighbourhood distances in. A neighbourhood is kNN *here*, not in the image.
        radii: ``(n_r,)`` eccentricity of each sampling ring, in degrees.
        counts: ``(n_r,)`` number of samples drawn on each ring.
        fov: Field of view in degrees.
        cmf_a: The ``a`` of ``M(r) = k / (r + a)`` in degrees; smaller is stronger foveation.
    """

    visual: torch.Tensor
    cortical: torch.Tensor
    radii: torch.Tensor
    counts: torch.Tensor
    fov: float
    cmf_a: float

    def __len__(self) -> int:
        return self.visual.shape[0]

    @property
    def eccentricity(self) -> torch.Tensor:
        """Per-sample eccentricity in degrees, derived from `visual`."""
        return self.visual.norm(dim=-1) * (self.fov / 2.0)


def cortical_coords(
    visual: torch.Tensor,
    cmf_a: float,
    fov: float,
    cmf_scale: float = DEFAULT_CMF_SCALE,
) -> torch.Tensor:
    """
    Map normalised visual coordinates onto the Rovamo-Virsu cortical manifold.

    Follows ``fovi.sensing.manifold``'s planar geometry, where the manifold is the surface
    that makes ``M(r)`` isotropic::

        M(r)   = k / (r + a)                          [mm / deg]
        rho(r) = r * M(r)                             [mm]
        z(r)   = k * (asinh(t) - t / sqrt(1 + t^2)),  t = sqrt((r/a + 1)^2 - 1)

    with ``z`` chosen so that ``dz/dr = sqrt(M^2 - (d rho/dr)^2)``, i.e. so the surface
    element is isotropic everywhere -- the property that keeps receptive fields circular.

    Args:
        visual: ``(..., 2)`` normalised visual coordinates.
        cmf_a: CMF ``a`` in degrees.
        fov: Field of view in degrees.
        cmf_scale: CMF ``k`` in mm/deg.

    Returns:
        ``(..., 3)`` cortical Cartesian coordinates in mm.
    """
    r_deg = visual.norm(dim=-1) * (fov / 2.0)
    theta = torch.atan2(visual[..., 1], visual[..., 0])

    rho = cmf_scale * r_deg / (cmf_a + r_deg)
    t = ((r_deg / cmf_a + 1.0) ** 2 - 1.0).clamp_min(0.0).sqrt()
    z = cmf_scale * (torch.asinh(t) - t / (1.0 + t * t).sqrt())

    return torch.stack([rho * torch.cos(theta), rho * torch.sin(theta), z], dim=-1)


def isotropic_sensor(
    cmf_a: float,
    fov: float = 16.0,
    target_samples: int = 1024,
    n_radii: int | None = None,
    cmf_scale: float = DEFAULT_CMF_SCALE,
    max_theta: int = 720,
) -> SensorArray:
    """
    Build an isotropically sampled foveated sensor.

    Radii are placed at equal steps of the integrated CMF, ``log(r + a)``, so they are
    evenly spaced along the manifold's radial axis; the number of angular samples on each
    ring is then whatever makes the arc between neighbouring angles equal the gap between
    neighbouring radii (local isotropy). The angular count is therefore *derived*, never
    chosen -- as in FOVI Sec. 4.1. Concretely, with ``du`` the log-radius step,
    ``n_theta(r) = 2*pi*r / ((r + a) * du)``.

    Args:
        cmf_a: CMF ``a`` in degrees. ``a -> inf`` recovers uniform sampling; small ``a``
            is strong foveation.
        fov: Field of view in degrees.
        target_samples: Number of samples to aim for. Ignored when `n_radii` is given.
        n_radii: Number of sampling rings. When None, searched to best match
            `target_samples` (FOVI searches over the same quantity).
        cmf_scale: CMF ``k`` in mm/deg.
        max_theta: Cap on samples per ring, to bound pathological sensors.

    Returns:
        A `SensorArray`.
    """
    if cmf_a <= 0:
        raise ValueError(f"cmf_a must be positive, got {cmf_a}")
    if fov <= 0:
        raise ValueError(f"fov must be positive, got {fov}")

    half = fov / 2.0

    def rings(n_r: int) -> tuple[torch.Tensor, torch.Tensor]:
        u = torch.linspace(0.0, math.log((half + cmf_a) / cmf_a), n_r, dtype=torch.float64)
        radii = cmf_a * torch.expm1(u)  # r = a * (exp(u) - 1); uniform steps in log(r + a)
        d_radius = (radii + cmf_a) * (u[1] - u[0]) if n_r > 1 else torch.full_like(radii, half)
        n_theta = (2.0 * math.pi * radii / d_radius).round().clamp(1, max_theta).long()
        n_theta[0] = 1  # the innermost "ring" is the fixation point itself
        return radii, n_theta

    if n_radii is None:
        best: tuple[float, int] | None = None
        for candidate in range(2, 400):
            _, n_theta = rings(candidate)
            error = abs(int(n_theta.sum().item()) - target_samples)
            if best is None or error < best[0]:
                best = (float(error), candidate)
            if error == 0:
                break
        assert best is not None
        n_radii = best[1]

    radii, n_theta = rings(n_radii)

    points = []
    for radius, count in zip(radii.tolist(), n_theta.tolist(), strict=True):
        if count == 1:
            points.append((0.0, 0.0))
            continue
        for j in range(count):
            angle = 2.0 * math.pi * j / count
            points.append((radius * math.cos(angle), radius * math.sin(angle)))

    visual = torch.tensor(points, dtype=torch.float32) / half
    return SensorArray(
        visual=visual,
        cortical=cortical_coords(visual, cmf_a, fov, cmf_scale),
        radii=radii.float(),
        counts=n_theta,
        fov=fov,
        cmf_a=cmf_a,
    )


# --------------------------------------------------------------------------------------
# Neighbourhoods and the kernel mapping
# --------------------------------------------------------------------------------------


def knn(
    query: torch.Tensor,
    key: torch.Tensor,
    k: int,
    chunk: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Indices and distances of the `k` nearest `key` points for each `query` point.

    Args:
        query: ``(n_query, d)``.
        key: ``(n_key, d)``.
        k: Neighbours to return.
        chunk: Rows of `query` to process at once, to bound the distance matrix.

    Returns:
        ``(indices (n_query, k) long, distances (n_query, k))``, both sorted by distance.
    """
    if k > key.shape[0]:
        raise ValueError(f"k={k} exceeds the number of key points ({key.shape[0]})")

    idx_out, dist_out = [], []
    for start in range(0, query.shape[0], chunk):
        d = torch.cdist(query[start : start + chunk], key, compute_mode=EXACT_EUCLIDEAN)
        dist, idx = d.topk(k, dim=1, largest=False)
        idx_out.append(idx)
        dist_out.append(dist)
    return torch.cat(idx_out), torch.cat(dist_out)


def neighbourhood_coordinates(
    out_visual: torch.Tensor,
    in_visual: torch.Tensor,
    out_cortical: torch.Tensor,
    in_cortical: torch.Tensor,
    knn_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Express each raw neighbourhood in the shared reference frame (FOVI's kernel mapping).

    Per neighbourhood, and per neighbour: take the geodesic distance on the manifold as
    the radius, the polar angle *in visual space* as the angle, and project back to
    Cartesian -- ``(r cos(theta), r sin(theta))``. Radius on the manifold plus angle in
    the visual field is what makes a vertical filter detect vertical edges everywhere:
    the frame is uniform in the manifold while staying aligned with the picture.

    Mirrors the first half of ``fovi.arch.knn.KNNConvLayer.compute_local_rf``; the
    remaining per-axis rescaling to ``[-1, 1]`` is `neighbourhood_frames`.

    Args:
        out_visual: ``(n_out, 2)`` normalised visual coordinates of the output units.
        in_visual: ``(n_in, 2)`` normalised visual coordinates of the inputs.
        out_cortical: ``(n_out, 3)`` manifold coordinates of the output units.
        in_cortical: ``(n_in, 3)`` manifold coordinates of the inputs.
        knn_indices: ``(n_out, k)`` indices into the input arrays, ``-1`` where missing.

    Returns:
        ``(coords (n_out, k, 2), valid (n_out, k) bool)`` in reference-frame units but
        *un*-normalised -- the shape each neighbourhood actually has on the manifold.
        Coordinates of invalid neighbours are zero and must be masked by the caller.
    """
    valid = knn_indices >= 0
    safe = knn_indices.clamp_min(0)

    in_c = in_cortical[safe]  # (n_out, k, 3)
    in_v = in_visual[safe]  # (n_out, k, 2)

    geodesic = (out_cortical[:, None, :] - in_c).norm(dim=-1)  # (n_out, k)
    delta = in_v - out_visual[:, None, :]
    theta = torch.atan2(delta[..., 1], delta[..., 0])

    coords = torch.stack([geodesic * torch.cos(theta), geodesic * torch.sin(theta)], dim=-1)
    return torch.where(valid.unsqueeze(-1), coords, torch.zeros_like(coords)), valid


def neighbourhood_frames(
    out_visual: torch.Tensor,
    in_visual: torch.Tensor,
    out_cortical: torch.Tensor,
    in_cortical: torch.Tensor,
    knn_indices: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    `neighbourhood_coordinates`, rescaled per axis into ``[-1, 1]``.

    This is the per-neighbourhood affine stretch that lets every neighbourhood fill the
    shared reference frame. Because it is per axis, it also *hides* the neighbourhood's
    own shape -- which is why `neighbourhood_coordinates` and `neighbourhood_anisotropy`
    exist as separate handles on the geometry underneath.

    Returns:
        ``(frames (n_out, k, 2) in [-1, 1], valid (n_out, k) bool)``. Invalid neighbours
        are zeroed *after* the rescaling (and excluded from it), so a reserved slot can
        neither masquerade as a real point nor drag the extent.
    """
    coords, valid = neighbourhood_coordinates(
        out_visual, in_visual, out_cortical, in_cortical, knn_indices
    )
    keep = valid.unsqueeze(-1)
    lo = torch.where(keep, coords, torch.full_like(coords, float("inf"))).amin(dim=1, keepdim=True)
    hi = torch.where(keep, coords, torch.full_like(coords, float("-inf"))).amax(dim=1, keepdim=True)
    lo = torch.where(torch.isfinite(lo), lo, torch.zeros_like(lo))
    hi = torch.where(torch.isfinite(hi), hi, torch.zeros_like(hi))
    frames = 2.0 * (coords - lo) / (hi - lo + eps) - 1.0
    return torch.where(keep, frames, torch.zeros_like(frames)), valid


def reference_grid(k: int, doubleres: bool = False) -> torch.Tensor:
    """
    The square reference grid a neighbourhood's kernel lives on, in ``[-1, 1]``.

    Matches ``KNNConvLayer.compute_reference_coords``: side ``ceil(sqrt(k))``, doubled when
    `doubleres` is set. `doubleres` is FOVI's default because nearest-grid-point rounding
    costs accuracy at the base resolution.

    Args:
        k: Neighbourhood size.
        doubleres: Use a grid with twice the side length.

    Returns:
        ``(side * side, 2)`` reference coordinates.
    """
    side = int(math.ceil(math.sqrt(k)))
    if doubleres:
        side *= 2
    axis = torch.linspace(-1.0, 1.0, side)
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack([x.reshape(-1), y.reshape(-1)], dim=-1)


def one_hot_mapping(frames: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Map every neighbour to its nearest reference-grid point (FOVI's one-hot mapping).

    Args:
        frames: ``(n_out, k, 2)`` neighbourhood coordinates in ``[-1, 1]``.
        ref: ``(n_ref, 2)`` reference coordinates.

    Returns:
        ``(n_out, k, n_ref)`` with exactly one 1 per ``(n_out, k)``.
    """
    n_out, k, _ = frames.shape
    mapping = torch.zeros(n_out, k, ref.shape[0], dtype=frames.dtype, device=frames.device)
    for start in range(0, n_out, 256):
        block = frames[start : start + 256]
        nearest = torch.cdist(block.reshape(-1, 2), ref, compute_mode=EXACT_EUCLIDEAN).argmin(dim=-1)
        nearest = nearest.reshape(block.shape[0], k)
        mapping[start : start + block.shape[0]].scatter_(2, nearest.unsqueeze(-1), 1.0)
    return mapping


def mapping_residual(frames: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Distance from each neighbour to its snapped grid point, per output unit.

    This is exactly the information `one_hot_mapping` throws away, so it upper-bounds the
    kernel-mapping error of the ``*_ref`` arms (in reference-frame units, where one grid
    cell spans ``2 / (side - 1)``). The ``analytic`` arm is zero here by construction.

    Args:
        frames: ``(n_out, k, 2)``.
        ref: ``(n_ref, 2)``.

    Returns:
        ``(n_out,)`` mean over the neighbourhood.
    """
    n_out, k, _ = frames.shape
    out = []
    for start in range(0, n_out, 256):
        block = frames[start : start + 256].reshape(-1, 2)
        nearest = torch.cdist(block, ref, compute_mode=EXACT_EUCLIDEAN).min(dim=-1).values
        out.append(nearest.reshape(-1, k).mean(dim=-1))
    return torch.cat(out)


def neighbourhood_anisotropy(coords: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Shape of each neighbourhood *before* the per-axis rescaling: the aspect ratio of the
    fitted ellipse, ``sqrt(lambda_max / lambda_min)`` of the coordinate covariance.

    Rotation-invariant (unlike a per-axis std ratio) and the same index FOVI computes in
    Appendix S4 to condemn the warp of log-polar and warped-Cartesian sensors. It is the
    right thing to measure *here* because FOVI's per-axis stretch to ``[-1, 1]`` forces
    every neighbourhood to look square afterwards, hiding its true shape -- so a
    structured kernel, which inherits the stretch rather than absorbing it into free
    taps, needs the pre-stretch geometry to be isotropic for "one filter, everywhere"
    (FOVI Fig. 3) to actually hold.

    Args:
        coords: ``(n_out, k, 2)`` un-normalised neighbourhood coordinates, from
            `neighbourhood_coordinates`. Invalid (zeroed) slots bias the estimate and
            should be dropped by the caller.

    Returns:
        ``(n_out,)`` ratios, ``1.0`` for a circular neighbourhood.
    """
    centred = coords - coords.mean(dim=1, keepdim=True)
    cov = torch.einsum("nki,nkj->nij", centred, centred) / max(coords.shape[1] - 1, 1)
    # Closed-form eigenvalues of the symmetric 2x2 covariance.
    trace = cov[:, 0, 0] + cov[:, 1, 1]
    gap = (cov[:, 0, 0] - cov[:, 1, 1]) / 2.0
    disc = (gap * gap + cov[:, 0, 1] * cov[:, 1, 0]).clamp_min(0.0).sqrt()
    lam_max = (trace / 2.0 + disc).clamp_min(eps)
    lam_min = (trace / 2.0 - disc).clamp_min(eps)
    return (lam_max / lam_min).sqrt()


# --------------------------------------------------------------------------------------
# The layer
# --------------------------------------------------------------------------------------


class StructuredKNNConv(nn.Module):
    """
    kNN convolution on a sensor manifold, with dense or structured reference kernels.

    Drop-in counterpart of ``fovi.arch.knn.KNNConvLayer``: input and output are
    ``(batch, channels, n_points)`` feature maps on the manifold. The three `mode` arms
    exist so that the structured parameterisation can be compared against FOVI's own
    mapping with everything else held fixed.

    Args:
        in_channels: Input features.
        out_channels: Output features.
        k: Neighbourhood size.
        in_coords: ``(n_in, 2)`` normalised visual coordinates of the inputs.
        out_coords: ``(n_out, 2)`` normalised visual coordinates of the outputs.
        mode: ``"dense_ref"``, ``"structured_ref"`` or ``"analytic"``.
        fov: Field of view in degrees, used to build the manifold.
        cmf_a: CMF ``a`` in degrees, used to build the manifold.
        doubleres: Double the reference grid (``dense_ref``/``structured_ref`` only).
        sigma_init: Initial Gaussian scale, in reference-frame units.
        order_init_range: Uniform init range for the fractional orders.
        learn_sigma: Whether ``sigma`` is learned or fixed.
        bias: Whether to add a bias term.
        geodesic: Neighbours on the manifold (True) or in visual space (False).
        knn_indices: Precomputed ``(n_out, k)`` neighbourhoods, to share across layers.
    """

    MODES = ("dense_ref", "structured_ref", "analytic")

    #: Same bounds as `fracsrf.layers.FracSRFConv2d`: the Hermite recursion depth, hence
    #: the cost, grows with the order, and a runaway sigma is invisible in the parameter
    #: count but not in the compute.
    MAX_ORDER = 10
    MIN_SIGMA = 0.05
    MAX_SIGMA = 8.0

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int,
        in_coords: torch.Tensor,
        out_coords: torch.Tensor,
        mode: str = "analytic",
        fov: float = 16.0,
        cmf_a: float = 0.5,
        doubleres: bool = False,
        sigma_init: float = 1.0,
        order_init_range: tuple[float, float] = (1.0, 3.0),
        learn_sigma: bool = True,
        bias: bool = True,
        geodesic: bool = True,
        knn_indices: torch.Tensor | None = None,
        cmf_scale: float = DEFAULT_CMF_SCALE,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mode = mode
        self.k = k
        self.fov = fov
        self.cmf_a = cmf_a

        self.register_buffer("in_visual", in_coords.clone(), persistent=False)
        self.register_buffer("out_visual", out_coords.clone(), persistent=False)
        self.register_buffer(
            "in_cortical", cortical_coords(in_coords, cmf_a, fov, cmf_scale), persistent=False
        )
        self.register_buffer(
            "out_cortical", cortical_coords(out_coords, cmf_a, fov, cmf_scale), persistent=False
        )

        if knn_indices is None:
            space = self.in_cortical if geodesic else in_coords
            query = self.out_cortical if geodesic else out_coords
            knn_indices, _ = knn(query, space, k)
        self.register_buffer("knn_indices", knn_indices.clone(), persistent=False)

        frames, valid = neighbourhood_frames(
            out_coords, in_coords, self.out_cortical, self.in_cortical, self.knn_indices
        )
        self.register_buffer("frames", frames, persistent=False)
        self.register_buffer("valid", valid, persistent=False)

        if mode != "analytic":
            self.register_buffer("ref_coords", reference_grid(k, doubleres), persistent=False)
            self.register_buffer("local_rf", one_hot_mapping(frames, self.ref_coords), persistent=False)
        else:
            self.register_buffer("ref_coords", torch.empty(0, 2), persistent=False)
            self.register_buffer("local_rf", torch.empty(0), persistent=False)
        self.ref_side = int(math.ceil(math.sqrt(k))) * (2 if doubleres else 1)

        self.log_sigma = nn.Parameter(
            torch.tensor(
                math.log(min(max(sigma_init, self.MIN_SIGMA), self.MAX_SIGMA)), dtype=torch.float32
            ),
            requires_grad=learn_sigma,
        )

        if mode == "dense_ref":
            self.weight = nn.Parameter(
                torch.empty(out_channels, in_channels * self.ref_coords.shape[0])
            )
            self.register_parameter("alpha", None)
            self.register_parameter("nu_x", None)
            self.register_parameter("nu_y", None)
        else:
            lo, hi = order_init_range
            self.alpha = nn.Parameter(torch.empty(out_channels, in_channels))
            self.nu_x = nn.Parameter(torch.empty(out_channels, in_channels).uniform_(lo, hi))
            self.nu_y = nn.Parameter(torch.empty(out_channels, in_channels).uniform_(lo, hi))
            self.register_parameter("weight", None)

        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.reset_parameters()

    # -- parameters --------------------------------------------------------------------

    @property
    def n_ref(self) -> int:
        """Reference-grid size. Defined for `analytic` too, which has no grid: it is the
        grid the structured kernel *would* live on, so the arms stay comparable."""
        return self.ref_side * self.ref_side

    @property
    def sigma(self) -> torch.Tensor:
        """Effective Gaussian scale, clipped to ``[MIN_SIGMA, MAX_SIGMA]``."""
        return self.log_sigma.clamp(math.log(self.MIN_SIGMA), math.log(self.MAX_SIGMA)).exp()

    def reset_parameters(self) -> None:
        """Initialise the reference kernel (dense arm) or the structured parameters."""
        if self.mode == "dense_ref":
            nn.init.kaiming_normal_(self.weight)
            return

        # Variance-preserving init like `FracSRFConv2d._init_alpha`: a Gaussian-derivative
        # basis carries far less energy than a dense kernel, so scaling alpha by the
        # basis energy keeps a fresh layer comparable to a Kaiming-initialised dense conv.
        with torch.no_grad():
            self.alpha.fill_(1.0)
            kernel = self.dense_kernel()  # (out, in, side, side)
            energy = kernel.pow(2).sum(dim=(-1, -2)).clamp_min(1e-12)
            std = torch.sqrt((2.0 / max(self.in_channels, 1)) / energy)
            self.alpha.copy_(torch.randn_like(self.alpha) * std)

    def clip_params_(self) -> None:
        """Clamp orders and sigma in place; call after ``optimizer.step()``."""
        with torch.no_grad():
            if self.nu_x is not None:
                self.nu_x.clamp_(0.0, self.MAX_ORDER)
                self.nu_y.clamp_(0.0, self.MAX_ORDER)
            self.log_sigma.clamp_(
                math.log(self.MIN_SIGMA), math.log(self.MAX_SIGMA)
            )

    def _max_order(self) -> int:
        if self.nu_x is None:
            return 1
        with torch.no_grad():
            order = torch.stack(
                [
                    self.nu_x.max().clamp_max(float(self.MAX_ORDER)),
                    self.nu_y.max().clamp_max(float(self.MAX_ORDER)),
                ]
            ).max()
        return max(1, min(self.MAX_ORDER, int(order.ceil().item()) + 1))

    # -- kernels -----------------------------------------------------------------------

    def dense_kernel(self) -> torch.Tensor:
        """
        The structured reference kernel as a dense ``(out, in, side, side)`` tensor.

        Reference-coordinate columns are the reference frame's ``(x, y)``; the grid is a
        separable product, so each 1-D basis is evaluated once on the axis and the kernel
        is their outer product. This is the arm that makes the structured filter
        comparable to FOVI's learned reference kernel, kernel for kernel.

        Returns:
            Tensor of shape ``(out, in, side, side)``, with ``[..., i, j]`` the tap at
            reference coordinate ``(axis[i], axis[j])`` -- the ``(x, y)`` convention
            `reference_grid` and `one_hot_mapping` use.
        """
        side = self.ref_side
        axis = torch.linspace(-1.0, 1.0, side, dtype=self.alpha.dtype, device=self.alpha.device)
        max_order = self._max_order()
        packed = hermite_gaussian_basis(axis, self.sigma, max_order)  # (M+1, side)

        lo_x, hi_x, frac_x = order_coefficients(self.nu_x, max_order)
        lo_y, hi_y, frac_y = order_coefficients(self.nu_y, max_order)
        gx = (1.0 - frac_x).unsqueeze(-1) * packed[lo_x] + frac_x.unsqueeze(-1) * packed[hi_x]
        gy = (1.0 - frac_y).unsqueeze(-1) * packed[lo_y] + frac_y.unsqueeze(-1) * packed[hi_y]

        kernel = gx.unsqueeze(-1) * gy.unsqueeze(-2)  # (out, in, side, side)
        return kernel * self.alpha.view(self.out_channels, self.in_channels, 1, 1)

    def reference_kernel(self) -> torch.Tensor:
        """Reference kernel flattened to the dense arm's ``(out, in * n_ref)`` layout."""
        kernel = self.dense_kernel().reshape(self.out_channels, self.in_channels * self.n_ref)
        return kernel

    # -- forward -----------------------------------------------------------------------

    def _gather(self, x: torch.Tensor) -> torch.Tensor:
        """Neighbourhood features ``(batch, in, n_out, k)``, zero at invalid neighbours."""
        batch, channels, _ = x.shape
        n_out = self.knn_indices.shape[0]
        safe = self.knn_indices.clamp_min(0)  # -1 marks an unsampled neighbour
        flat = safe.reshape(1, 1, -1).expand(batch, channels, -1)
        gathered = torch.gather(x, 2, flat).reshape(batch, channels, n_out, self.k)
        mask = self.valid.reshape(1, 1, n_out, self.k).to(gathered.dtype)
        return gathered * mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the kNN convolution.

        Args:
            x: ``(batch, in_channels, n_points)`` features on the manifold.

        Returns:
            ``(batch, out_channels, n_out)``.
        """
        if x.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {x.shape[1]}")

        neighbours = self._gather(x)  # (B, in, n_out, k)

        if self.mode == "analytic":
            out = self._forward_analytic(neighbours)
        else:
            weight = self.weight if self.mode == "dense_ref" else self.reference_kernel()
            # Fold the reference-grid axis into the channel axis, as FOVI does, so the
            # reference kernel is one (out, in * n_ref) matrix. The (in, n_ref) ordering
            # here must match `reference_kernel`'s flattening.
            mapped = torch.einsum("bcqk,qkv->bcvq", neighbours, self.local_rf)
            mapped = mapped.reshape(x.shape[0], -1, mapped.shape[-1])
            out = torch.einsum("oi,bin->bon", weight, mapped)

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1)
        return out

    def _forward_analytic(self, neighbours: torch.Tensor) -> torch.Tensor:
        """
        Exact mapping: evaluate the kernel at the neighbour positions, grid-free.

        The separable kernel in a neighbourhood is ``alpha * G_nux(u) * G_nuy(v)``, and a
        fractional order is a mix of exactly two integer orders, so the mapped filter
        decomposes into at most four terms per ``(out, in)`` pair, keyed by an
        integer-order pair ``(m_x, m_y)``. Terms that share a key collapse into one
        ``(out, in)`` matrix, leaving one basis-product reduction and one matmul -- so the
        work stays dense-conv-shaped instead of becoming a per-neighbourhood kernel.

        Args:
            neighbours: ``(batch, in, n_out, k)`` gathered features.

        Returns:
            ``(batch, out, n_out)``.
        """
        u = self.frames[..., 0]  # (n_out, k)
        v = self.frames[..., 1]
        max_order = self._max_order()

        basis_u = hermite_gaussian_basis(u, self.sigma, max_order)  # (M+1, n_out, k)
        basis_v = hermite_gaussian_basis(v, self.sigma, max_order)

        lo_x, hi_x, frac_x = order_coefficients(self.nu_x, max_order)
        lo_y, hi_y, frac_y = order_coefficients(self.nu_y, max_order)

        strides = max_order + 1
        codes, coefficients = [], []
        for m_x, w_x in ((lo_x, 1.0 - frac_x), (hi_x, frac_x)):
            for m_y, w_y in ((lo_y, 1.0 - frac_y), (hi_y, frac_y)):
                codes.append(m_x * strides + m_y)  # which pair of basis slabs
                coefficients.append(self.alpha * w_x * w_y)  # and how much of it

        # Collapse the (at most four) terms onto the integer-order pairs they refer to.
        # Each term carries a coefficient for every (out, in) pair, and different pairs
        # land in different slots, so this is an elementwise accumulation, not a row add.
        n_terms = len(codes)
        n_pairs = self.out_channels * self.in_channels
        flat_codes = torch.cat([code.reshape(-1) for code in codes])
        values = torch.cat([c.reshape(-1) for c in coefficients])
        slots, inverse = torch.unique(flat_codes, return_inverse=True)
        columns = torch.arange(n_pairs, device=values.device).repeat(n_terms)
        weights = torch.zeros(
            (slots.numel(), n_pairs), dtype=values.dtype, device=values.device
        ).index_put_((inverse, columns), values, accumulate=True)
        weights = weights.view(slots.numel(), self.out_channels, self.in_channels)

        products = basis_u[slots // strides] * basis_v[slots % strides]  # (slots, n_out, k)
        summary = torch.einsum("bcqk,mqk->bmcq", neighbours, products)
        return torch.einsum("moc,bmcq->boq", weights, summary)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, k={self.k}, "
            f"mode={self.mode}, n_ref={self.n_ref}, sigma={self.sigma.item():.3f}, "
            f"cmf_a={self.cmf_a}"
        )
