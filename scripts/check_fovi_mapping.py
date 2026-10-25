#!/usr/bin/env python
"""
Measure the kernel-mapping error a structured reference kernel removes, without training.

The three arms in `fracsrf.manifold.StructuredKNNConv` carry the *same* filter:

    dense_ref  (rf1)      FOVI's baseline: one learned tap per reference-grid point, and
                          every neighbour snapped to its nearest grid point.
    dense_ref  (rf2)      FOVI's `doubleres` default: 4x the taps, half the snap distance.
    structured_ref        same snapped mapping, kernel generated from (alpha, nu, sigma).
    analytic              same parameters, kernel evaluated *at* the neighbour positions.

Comparing them isolates the mapping, so the difference is the error FOVI's finer grid pays
for in parameters -- and, read against eccentricity, whether that error is uniform or
concentrated where the neighbourhoods are least square.

Usage:
    python scripts/check_fovi_mapping.py                      # table only
    python scripts/check_fovi_mapping.py --figure outputs/fovi_mapping.png
"""

import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fracsrf.basis import hermite_gaussian_basis, interpolate_order  # noqa: E402
from fracsrf.manifold import (  # noqa: E402
    StructuredKNNConv,
    isotropic_sensor,
    knn,
    mapping_residual,
    neighbourhood_anisotropy,
    neighbourhood_coordinates,
)


def build_arms(sensor, k, in_channels, out_channels, indices, cmf_scale=10.0):
    """Four layers holding the same filter, differing only in how it reaches the pixels."""
    common = dict(
        in_channels=in_channels,
        out_channels=out_channels,
        k=k,
        in_coords=sensor.visual,
        out_coords=sensor.visual,
        fov=sensor.fov,
        cmf_a=sensor.cmf_a,
        cmf_scale=cmf_scale,
        knn_indices=indices,
    )
    analytic = StructuredKNNConv(mode="analytic", **common)
    structured = StructuredKNNConv(mode="structured_ref", **common)
    dense = StructuredKNNConv(mode="dense_ref", **common)
    dense_fine = StructuredKNNConv(mode="dense_ref", doubleres=True, **common)

    # One filter everywhere: copy the structured parameters, then give each dense arm the
    # kernel those parameters produce *on its own grid*. Nothing else differs.
    with torch.no_grad():
        structured.alpha.copy_(analytic.alpha)
        structured.nu_x.copy_(analytic.nu_x)
        structured.nu_y.copy_(analytic.nu_y)
        structured.log_sigma.copy_(analytic.log_sigma)
        dense.weight.copy_(structured.reference_kernel())
        dense.log_sigma.copy_(analytic.log_sigma)

        side = dense_fine.ref_side
        axis = torch.linspace(-1.0, 1.0, side, dtype=analytic.alpha.dtype)
        packed = hermite_gaussian_basis(axis, analytic.sigma, analytic._max_order())
        gx = interpolate_order(packed, analytic.nu_x)
        gy = interpolate_order(packed, analytic.nu_y)
        kernel = gx.unsqueeze(-1) * gy.unsqueeze(-2) * analytic.alpha.view(out_channels, in_channels, 1, 1)
        dense_fine.weight.copy_(kernel.reshape(out_channels, -1))
        dense_fine.log_sigma.copy_(analytic.log_sigma)

    return {"rf1": dense, "rf2": dense_fine, "structured": structured, "analytic": analytic}


def parameter_count(layer):
    """Learnable tensors only, so the four arms are comparable head to head."""
    return sum(p.numel() for p in layer.parameters() if p.requires_grad)


def forward_ms(layer, x, repeats=15):
    with torch.no_grad():
        for _ in range(3):
            layer(x)
        start = time.perf_counter()
        for _ in range(repeats):
            layer(x)
        return (time.perf_counter() - start) * 1000.0 / repeats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", type=float, default=0.5, help="CMF `a` in degrees; smaller is stronger foveation")
    parser.add_argument("--fov", type=float, default=16.0, help="field of view in degrees")
    parser.add_argument("--samples", type=int, default=1024, help="target number of sensor samples")
    parser.add_argument("--k", type=int, default=25, help="neighbourhood size")
    parser.add_argument("--channels", type=int, default=8, help="in channel count; out is 2x")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--figure", type=Path, default=None, help="write a PNG of sensor + projected kernels")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    sensor = isotropic_sensor(cmf_a=args.a, fov=args.fov, target_samples=args.samples)
    indices, _ = knn(sensor.cortical, sensor.cortical, k=args.k)
    in_channels, out_channels = args.channels, args.channels * 2

    print(f"sensor: n={len(sensor)} samples, {len(sensor.radii)} rings, fov={sensor.fov:g} deg, a={sensor.cmf_a}")
    print(
        f"        eccentricity {float(sensor.eccentricity.min()):.2f}..{float(sensor.eccentricity.max()):.2f} deg, "
        f"k={args.k} on the cortical manifold "
        f"({sensor.cortical.shape[1]}-d, {sensor.cortical.norm(dim=-1).max():.1f} mm max radius)"
    )

    arms = build_arms(sensor, args.k, in_channels, out_channels, indices)
    frames = arms["analytic"].frames
    raw, valid = neighbourhood_coordinates(
        sensor.visual, sensor.visual, sensor.cortical, sensor.cortical, indices
    )
    x = torch.randn(args.batch, in_channels, len(sensor))

    with torch.no_grad():
        reference = arms["analytic"](x)
    scale = float(reference.pow(2).mean().sqrt())

    header = f"\n{'arm':<12}{'ref side':>9}{'taps':>7}{'params':>10}{'snap err':>10}{'rel RMS vs analytic':>21}{'fwd ms':>8}"
    print(header)
    print("-" * len(header))
    errors = {}
    for name in ("rf1", "rf2", "structured", "analytic"):
        layer = arms[name]
        if layer.ref_coords.numel():
            residual = float(mapping_residual(frames, layer.ref_coords).mean())
        else:
            residual = 0.0
        with torch.no_grad():
            out = layer(x)
        errors[name] = out - reference
        rel = float((out - reference).pow(2).mean().sqrt()) / scale
        side = layer.ref_side if layer.mode != "analytic" else 0
        print(
            f"{name:<12}{side:>9}{layer.n_ref:>7}{parameter_count(layer):>10}"
            f"{residual:>10.4f}{rel:>21.2%}{forward_ms(layer, x):>8.2f}"
        )
    print("\nsnap err = mean distance from a neighbour to its nearest reference point, in [-1,1] frame units")
    print("rel RMS  = RMS(output - analytic output) / RMS(analytic output), same filter in every arm")

    # -- does the mapping error depend on eccentricity? ----------------------------------
    ecc = sensor.eccentricity
    edges = torch.quantile(ecc, torch.linspace(0, 1, 7)).unique()
    anisotropy = neighbourhood_anisotropy(raw)

    print(f"\n{'ecc (deg)':>18}{'n':>6}{'snap err rf1':>14}{'rf2':>8}{'aniso':>7}{'rel RMS rf1':>13}{'rf2':>8}")
    print("-" * 75)
    residual_rf1 = mapping_residual(frames, arms["rf1"].ref_coords)
    residual_rf2 = mapping_residual(frames, arms["rf2"].ref_coords)
    for lo, hi in zip(edges[:-1].tolist(), edges[1:].tolist(), strict=True):
        mask = (ecc >= lo) & (ecc <= hi)
        if not bool(mask.any()):
            continue
        err_rf1 = errors["rf1"][:, :, mask].pow(2).sum().sqrt()
        err_rf2 = errors["rf2"][:, :, mask].pow(2).sum().sqrt()
        ref = reference[:, :, mask].pow(2).sum().sqrt()
        print(
            f"{lo:8.2f} - {hi:6.2f}{int(mask.sum()):>6}"
            f"{float(residual_rf1[mask].mean()):>14.4f}{float(residual_rf2[mask].mean()):>8.4f}"
            f"{float(anisotropy[mask].mean()):>7.2f}"
            f"{float(err_rf1 / ref):>13.2%}{float(err_rf2 / ref):>8.2%}"
        )

    print(
        "\nReading: `aniso` is the pre-stretch aspect ratio of each neighbourhood (sqrt of the\n"
        "covariance eigenvalue ratio) -- the same index FOVI computes in Appendix S4. It bounds\n"
        "how far 'the same filter across the visual field' (FOVI Fig. 3) can hold for a\n"
        "structured kernel: a dense kernel absorbs the per-axis stretch, an analytic one\n"
        "inherits it. It does NOT rise with eccentricity here, so the shared-sigma\n"
        "assumption is safe at this foveation strength; check --a 0.03 before generalising."
    )

    if args.figure is not None:
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        render(sensor, arms["analytic"], frames, indices, args.figure)
        print(f"\nwrote {args.figure}")


def render(sensor, analytic, frames, indices, path, size=384):
    """Left: the foveated sample cloud. Right: one filter projected back into visual space
    at three eccentricities, which is the picture FOVI's Fig. 3 uses to argue that the
    filter is constant across the field."""
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (size * 2 + 12, size), "white")
    draw = ImageDraw.Draw(canvas)

    # Panel 1: sample positions, in normalised visual coordinates.
    for u, v in sensor.visual.tolist():
        px = size / 2 + u * size / 2 * 0.98
        py = size / 2 + v * size / 2 * 0.98
        draw.ellipse([px - 1.1, py - 1.1, px + 1.1, py + 1.1], fill=(40, 40, 40))
    draw.ellipse([size / 2 - 2, size / 2 - 2, size / 2 + 2, size / 2 + 2], fill=(200, 30, 30))
    draw.text((8, 8), f"foveated samples (n={len(sensor)})", fill=(0, 0, 0))

    # Panel 2: three stacked views of one filter, each centred on a different eccentricity.
    units = []
    for target in (1.0, 4.0, 7.5):
        ecc = sensor.eccentricity
        units.append(int((ecc - target).abs().argmin()))

    with torch.no_grad():
        max_order = analytic._max_order()
        basis_u = hermite_gaussian_basis(frames[..., 0], analytic.sigma, max_order)
        basis_v = hermite_gaussian_basis(frames[..., 1], analytic.sigma, max_order)
        gx = interpolate_order(basis_u, analytic.nu_x)  # (out, in, n_out, k)
        gy = interpolate_order(basis_v, analytic.nu_y)
        weights = analytic.alpha.view(*analytic.alpha.shape, 1, 1) * gx * gy

    for row, unit in enumerate(units):
        top = row * size / 3
        centre_x, centre_y = size + 6 + size / 2, top + size / 6
        radius_px = size / 6 * 0.85
        offset = (sensor.visual[indices[unit]] - sensor.visual[unit]) * (sensor.fov / 2)
        extent = float(offset.abs().max()) + 1e-6
        for j, (dx, dy) in enumerate(offset.tolist()):
            wv = float(weights[0, 0, unit, j])
            mag = min(abs(wv) / (float(weights[0, 0].abs().max()) + 1e-9), 1.0)
            px = centre_x + dx / extent * radius_px
            py = centre_y + dy / extent * radius_px
            r = 2.0 + 4.0 * mag
            colour = (int(200 * mag), 40, int(200 * (1 - mag))) if wv >= 0 else (40, int(200 * mag), 40)
            draw.ellipse([px - r, py - r, px + r, py + r], fill=colour)
        ecc_deg = float(sensor.eccentricity[unit])
        draw.text((size + 12, top + 4), f"unit at {ecc_deg:.1f} deg", fill=(0, 0, 0))

    draw.text((size + 12, size - 16), "one filter, projected into visual space", fill=(0, 0, 0))
    canvas.save(path)


if __name__ == "__main__":
    main()
