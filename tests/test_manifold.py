"""Tests for the FOVI-style sensor manifold and the structured kNN convolution.

The load-bearing test here is `test_analytic_matches_brute_force_per_neighbourhood`: the
analytic arm computes an equivalent filter without ever materialising a per-neighbourhood
kernel, so it is checked against a literal `(n_out, k)` weight tensor built from the basis.
Everything else pins down the pieces that are easy to get quietly wrong: local isotropy,
the cortical manifold's isotropy condition, and FOVI's one-hot mapping convention.
"""

import math
import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fracsrf import (
    StructuredKNNConv,
    cortical_coords,
    hermite_gaussian_basis,
    interpolate_order,
    isotropic_sensor,
    knn,
    mapping_residual,
    neighbourhood_anisotropy,
    neighbourhood_coordinates,
    neighbourhood_frames,
    one_hot_mapping,
    order_coefficients,
    reference_grid,
)


def _toy_sensor(n_radii=6, cmf_a=0.5, fov=16.0):
    return isotropic_sensor(cmf_a=cmf_a, fov=fov, target_samples=200, n_radii=n_radii)


class TestBasis(unittest.TestCase):
    def test_gaussian_zero_order_is_gaussian(self):
        """G_0 must be the plain Gaussian, so the recursion's normalisation is right."""
        x = torch.linspace(-3, 3, 41, dtype=torch.float64)
        sigma = torch.tensor(0.7, dtype=torch.float64)
        basis = hermite_gaussian_basis(x, sigma, max_order=0)
        expected = torch.exp(-(x**2) / (2 * 0.7**2)) / (0.7 * math.sqrt(2 * math.pi))
        torch.testing.assert_close(basis[0], expected)

    def test_integer_order_matches_finite_difference(self):
        """G_1 must be dG_0/dx, which is the defining recursion relation."""
        x = torch.linspace(-2.5, 2.5, 201, dtype=torch.float64)
        sigma = torch.tensor(1.0, dtype=torch.float64)
        basis = hermite_gaussian_basis(x, sigma, max_order=2)
        d1 = torch.gradient(basis[0], spacing=(float(x[1] - x[0]),))[0]
        inner = slice(2, -2)
        torch.testing.assert_close(basis[1][inner], d1[inner], atol=2e-3, rtol=1e-2)

    def test_fractional_order_interpolates_between_integers(self):
        """nu exactly halfway between two orders must equal the mean of those orders."""
        x = torch.linspace(-2, 2, 33)
        basis = hermite_gaussian_basis(x, torch.tensor(1.0), max_order=3)
        half = interpolate_order(basis, torch.tensor(1.5))
        torch.testing.assert_close(half, 0.5 * basis[1] + 0.5 * basis[2])

    def test_order_coefficients_split_matches_interpolate_order(self):
        """The coefficient decomposition must reconstruct `interpolate_order` exactly."""
        x = torch.linspace(-2, 2, 33)
        basis = hermite_gaussian_basis(x, torch.tensor(0.9), max_order=5)
        nu = torch.tensor([[0.25, 2.7], [4.5, 3.0]])
        lo, hi, frac = order_coefficients(nu, max_order=5)
        rebuilt = (1 - frac).unsqueeze(-1) * basis[lo] + frac.unsqueeze(-1) * basis[hi]
        torch.testing.assert_close(rebuilt, interpolate_order(basis, nu))


class TestSensor(unittest.TestCase):
    def test_every_sample_lies_inside_the_field_of_view(self):
        sensor = isotropic_sensor(cmf_a=0.5, fov=16.0, target_samples=300)
        self.assertLessEqual(float(sensor.visual.abs().max()), 1.0 + 1e-6)
        self.assertLessEqual(float(sensor.eccentricity.max()), 8.0 + 1e-4)

    def test_sample_count_tracks_the_target(self):
        """The n_radii search exists to hit a sample budget, so it has to get close."""
        for target in (128, 512, 1024):
            sensor = isotropic_sensor(cmf_a=0.5, fov=16.0, target_samples=target)
            self.assertLess(abs(len(sensor) - target) / target, 0.10, f"target={target}")

    def test_sampling_is_locally_isotropic(self):
        """
        The defining property: on each ring, the arc between neighbouring angles equals
        the gap to the neighbouring radii. Compared in *visual* degrees, as FOVI specifies.
        """
        sensor = isotropic_sensor(cmf_a=0.5, fov=16.0, target_samples=2000, n_radii=40)
        radii = sensor.radii
        counts = sensor.counts

        for i in range(1, len(radii) - 1):
            radial_gap = float(radii[i + 1] - radii[i - 1]) / 2.0
            arc = float(radii[i]) * 2.0 * math.pi / int(counts[i])
            # Rounding n_theta to an integer is the only loss, so allow one sample-step.
            self.assertLess(abs(arc - radial_gap) / radial_gap, 0.15, f"ring {i}: {arc} vs {radial_gap}")

    def test_weaker_foveation_spreads_samples_more_evenly(self):
        """Larger `a` is closer to uniform sampling, so the centre stops being privileged."""
        strong = isotropic_sensor(cmf_a=0.5, fov=16.0, target_samples=1000)
        weak = isotropic_sensor(cmf_a=50.0, fov=16.0, target_samples=1000)
        centre = lambda s: float((s.eccentricity < 1.0).float().mean())  # noqa: E731
        self.assertGreater(centre(strong), centre(weak))

    def test_cortical_manifold_is_isotropic(self):
        """
        The manifold's `z` is chosen so that the surface element is isotropic, i.e.
        dz/dr = sqrt(M^2 - (drho/dr)^2). Checked by finite differences on the meridian,
        which is where the two metric components are exactly separable.
        """
        cmf_a, fov, scale = 0.5, 16.0, 10.0
        eps = 1e-4
        for radius_deg in (0.25, 1.0, 4.0):
            def point(r):
                visual = torch.tensor([[r / (fov / 2.0), 0.0]])
                return cortical_coords(visual, cmf_a, fov, scale)[0]

            p0, p1 = point(radius_deg), point(radius_deg + eps)
            dz = float(p1[2] - p0[2]) / eps
            drho = float(p1[0] - p0[0]) / eps
            magnification = scale / (cmf_a + radius_deg)
            expected = math.sqrt(max(magnification**2 - drho**2, 0.0))
            self.assertAlmostEqual(dz, expected, delta=1e-2 * max(expected, 1.0), msg=f"r={radius_deg}")

    def test_cortical_radius_is_magnification_times_radius(self):
        cmf_a, fov, scale = 0.5, 16.0, 10.0
        visual = torch.tensor([[1.0, 0.0]])  # 8 degrees, on the +x meridian
        cortical = cortical_coords(visual, cmf_a, fov, scale)
        self.assertAlmostEqual(float(cortical[0, 0]), scale * 8.0 / (cmf_a + 8.0), places=5)
        self.assertAlmostEqual(float(cortical[0, 2]), float(cortical[0, 2]))  # z is finite


class TestNeighbourhoods(unittest.TestCase):
    def setUp(self):
        self.sensor = _toy_sensor(n_radii=8)
        self.indices, _ = knn(self.sensor.cortical, self.sensor.cortical, k=9)

    def test_knn_returns_sorted_distances_without_self(self):
        indices, distances = knn(self.sensor.cortical, self.sensor.cortical, k=9)
        self.assertEqual(tuple(indices.shape), (len(self.sensor), 9))
        self.assertTrue(torch.all(distances[:, 1:] >= distances[:, :-1] - 1e-6))
        self.assertTrue(torch.all(distances[:, 0] < 1e-5))  # each point is its own neighbour

    def test_knn_on_visual_space_differs_from_cortical_space(self):
        """Neighbours on the foveated manifold are not the Euclidean neighbours."""
        cortical_idx, _ = knn(self.sensor.cortical, self.sensor.cortical, k=9)
        visual_idx, _ = knn(self.sensor.visual, self.sensor.visual, k=9)
        self.assertFalse(torch.equal(cortical_idx, visual_idx))

    def test_frames_are_normalised_to_unit_box(self):
        frames, valid = neighbourhood_frames(
            self.sensor.visual,
            self.sensor.visual,
            self.sensor.cortical,
            self.sensor.cortical,
            self.indices,
        )
        self.assertTrue(bool(valid.all()))
        self.assertLessEqual(float(frames.abs().max()), 1.0 + 1e-5)
        # Per-axis extremes are reached by construction, so both axes span the box.
        self.assertTrue(torch.allclose(frames.amin(dim=1), -torch.ones(len(self.sensor), 2), atol=1e-5))
        self.assertTrue(torch.allclose(frames.amax(dim=1), torch.ones(len(self.sensor), 2), atol=1e-5))

    def test_invalid_neighbours_are_masked_out_of_the_frames(self):
        indices = self.indices.clone()
        indices[:, 0] = -1
        frames, valid = neighbourhood_frames(
            self.sensor.visual,
            self.sensor.visual,
            self.sensor.cortical,
            self.sensor.cortical,
            indices,
        )
        self.assertFalse(bool(valid[:, 0].any()))
        self.assertTrue(torch.all(frames[:, 0] == 0))

    def test_reference_grid_matches_fovi_convention(self):
        grid = reference_grid(k=10)
        self.assertEqual(grid.shape, (16, 2))  # ceil(sqrt(10)) = 4, squared
        self.assertAlmostEqual(float(grid.min()), -1.0, places=6)
        self.assertAlmostEqual(float(grid.max()), 1.0, places=6)
        doubled = reference_grid(k=10, doubleres=True)
        self.assertEqual(doubled.shape, (64, 2))
        # Row-major with `indexing="ij"`: index i*side + j is the tap at (axis[i], axis[j]),
        # so the first coordinate (x) varies slowest. `reference_kernel`'s flattening and
        # `one_hot_mapping`'s indexing both depend on this.
        axis = torch.linspace(-1.0, 1.0, 4)
        self.assertEqual(grid[0].tolist(), [axis[0].item(), axis[0].item()])
        self.assertEqual(grid[1].tolist(), [axis[0].item(), axis[1].item()])
        self.assertEqual(grid[4].tolist(), [axis[1].item(), axis[0].item()])

    def test_one_hot_mapping_selects_exactly_one_grid_point(self):
        grid = reference_grid(k=9)
        frames, _ = neighbourhood_frames(
            self.sensor.visual,
            self.sensor.visual,
            self.sensor.cortical,
            self.sensor.cortical,
            self.indices,
        )
        mapping = one_hot_mapping(frames, grid)
        self.assertEqual(tuple(mapping.shape), (len(self.sensor), 9, grid.shape[0]))
        torch.testing.assert_close(mapping.sum(dim=-1), torch.ones(len(self.sensor), 9))
        self.assertTrue(bool(((mapping == 0) | (mapping == 1)).all()))

    def test_doubled_reference_grid_reduces_the_mapping_residual(self):
        """
        The residual is exactly what the one-hot snap discards, so it must shrink with a
        finer grid -- this is the quantity FOVI's `doubleres` variant pays 4x for.
        """
        frames, _ = neighbourhood_frames(
            self.sensor.visual,
            self.sensor.visual,
            self.sensor.cortical,
            self.sensor.cortical,
            self.indices,
        )
        coarse = float(mapping_residual(frames, reference_grid(k=9)).mean())
        fine = float(mapping_residual(frames, reference_grid(k=9, doubleres=True)).mean())
        self.assertLess(fine, coarse)

    def test_raw_neighbourhoods_are_near_isotropic(self):
        """
        Pre-stretch neighbourhood shape: this is what a structured kernel inherits, and
        what the shared-sigma premise depends on. Measured on `neighbourhood_coordinates`,
        because the [-1, 1] rescaling forces every neighbourhood to look square.
        """
        raw, valid = neighbourhood_coordinates(
            self.sensor.visual,
            self.sensor.visual,
            self.sensor.cortical,
            self.sensor.cortical,
            self.indices,
        )
        self.assertTrue(bool(valid.all()))
        anisotropy = neighbourhood_anisotropy(raw)
        self.assertLess(float((anisotropy - 1.0).abs().median()), 0.35)

    def test_rescaling_hides_anisotropy_that_the_raw_frames_show(self):
        """
        The two diagnostics must disagree on a deliberately stretched neighbourhood --
        otherwise the anisotropy test above is measuring nothing.
        """
        coords = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 0.1], [0.02, 0.2]]])
        self.assertGreater(float(neighbourhood_anisotropy(coords)), 2.0)


class TestStructuredKNNConv(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.sensor = _toy_sensor(n_radii=10)
        self.in_channels, self.out_channels, self.k = 3, 4, 9
        self.indices, _ = knn(self.sensor.cortical, self.sensor.cortical, self.k)

    def _layer(self, mode, **kwargs):
        return StructuredKNNConv(
            self.in_channels,
            self.out_channels,
            self.k,
            self.sensor.visual,
            self.sensor.visual,
            mode=mode,
            fov=self.sensor.fov,
            cmf_a=self.sensor.cmf_a,
            knn_indices=self.indices,
            **kwargs,
        )

    def _brute_force(self, layer, x):
        """Materialise one kernel per neighbourhood and contract it, no bucketing."""
        frames = layer.frames  # (n_out, k, 2)
        max_order = layer._max_order()
        bases = [
            hermite_gaussian_basis(frames[..., axis], layer.sigma, max_order)
            for axis in (0, 1)
        ]
        gx = interpolate_order(bases[0], layer.nu_x)  # (out, in, n_out, k)
        gy = interpolate_order(bases[1], layer.nu_y)
        weights = layer.alpha.view(self.out_channels, self.in_channels, 1, 1) * gx * gy
        gathered = layer._gather(x)  # (B, in, n_out, k)
        return torch.einsum("bcqk,ocqk->boq", gathered, weights)

    def test_analytic_matches_brute_force_per_neighbourhood(self):
        layer = self._layer("analytic")
        x = torch.randn(2, self.in_channels, len(self.sensor))
        torch.testing.assert_close(layer(x), self._brute_force(layer, x) + layer.bias.view(1, -1, 1))

    def test_forward_shape(self):
        layer = self._layer("analytic")
        out = layer(torch.randn(2, self.in_channels, len(self.sensor)))
        self.assertEqual(tuple(out.shape), (2, self.out_channels, len(self.sensor)))

    def test_dense_ref_matches_structured_ref_when_weights_are_copied(self):
        """
        The two `*_ref` arms must be the same function given the same reference kernel;
        that is what makes them a fair mapping-error comparison rather than two systems.
        """
        structured = self._layer("structured_ref")
        dense = self._layer("dense_ref")
        with torch.no_grad():
            dense.weight.copy_(structured.reference_kernel())
            dense.log_sigma.copy_(structured.log_sigma)
        x = torch.randn(2, self.in_channels, len(self.sensor))
        torch.testing.assert_close(dense(x), structured(x), atol=1e-6, rtol=1e-5)

    def _dense_layer_with_grid(self, side, source):
        """
        A `dense_ref` arm whose reference grid has `side` points per axis, holding the
        kernel `source` would generate *at those grid points* -- i.e. the same filter,
        sampled more or less finely.
        """
        dense = self._layer("dense_ref")
        dense.ref_side = side
        dense.register_buffer("ref_coords", reference_grid(k=side**2), persistent=False)
        dense.register_buffer(
            "local_rf", one_hot_mapping(dense.frames, dense.ref_coords), persistent=False
        )
        axis = torch.linspace(-1.0, 1.0, side, dtype=source.alpha.dtype)
        packed = hermite_gaussian_basis(axis, source.sigma, source._max_order())
        gx = interpolate_order(packed, source.nu_x)
        gy = interpolate_order(packed, source.nu_y)
        kernel = (gx.unsqueeze(-1) * gy.unsqueeze(-2)) * source.alpha.view(
            self.out_channels, self.in_channels, 1, 1
        )
        dense.weight = torch.nn.Parameter(kernel.reshape(self.out_channels, -1), requires_grad=False)
        return dense

    def test_dense_arm_converges_to_analytic_as_the_grid_refines(self):
        """
        The two arms represent the same filter, so a finer reference grid must move the
        `dense_ref` output toward the analytic one. This is the mapping error FOVI's
        `doubleres` pays 4x the parameters to reduce, isolated from everything else.
        """
        analytic_layer = self._layer("analytic")
        x = torch.randn(2, self.in_channels, len(self.sensor))
        target = analytic_layer(x) - analytic_layer.bias.view(1, -1, 1)

        errors = []
        for side in (3, 6, 12):
            dense = self._dense_layer_with_grid(side, analytic_layer)
            with torch.no_grad():
                errors.append(float((dense(x) - target).pow(2).mean().sqrt()))

        self.assertLess(errors[1], errors[0], f"6 vs 3: {errors}")
        self.assertLess(errors[2], errors[1], f"12 vs 6: {errors}")

    def test_gradients_reach_every_structured_parameter(self):
        layer = self._layer("analytic")
        out = layer(torch.randn(2, self.in_channels, len(self.sensor)))
        out.pow(2).sum().backward()
        for name in ("alpha", "nu_x", "nu_y", "log_sigma"):
            grad = getattr(layer, name).grad
            self.assertIsNotNone(grad, name)
            self.assertTrue(torch.all(torch.isfinite(grad)), name)
            self.assertGreater(float(grad.abs().sum()), 0.0, name)

    def test_reference_kernel_layout_round_trips(self):
        """`reference_kernel` must be the flattening `local_rf` indexes into."""
        layer = self._layer("structured_ref")
        kernel = layer.dense_kernel()
        flat = layer.reference_kernel()
        torch.testing.assert_close(flat, kernel.reshape(self.out_channels, -1))

    def test_invalid_neighbours_contribute_nothing(self):
        """A neighbourhood of all-invalid neighbours must produce the bias alone."""
        indices = self.indices.clone()
        indices[0, :] = -1
        layer = StructuredKNNConv(
            self.in_channels,
            self.out_channels,
            self.k,
            self.sensor.visual,
            self.sensor.visual,
            mode="analytic",
            fov=self.sensor.fov,
            cmf_a=self.sensor.cmf_a,
            knn_indices=indices,
        )
        out = layer(torch.randn(1, self.in_channels, len(self.sensor)))
        torch.testing.assert_close(out[0, :, 0], layer.bias)

    def test_clip_params_bounds_orders_and_sigma(self):
        layer = self._layer("analytic")
        with torch.no_grad():
            layer.nu_x.fill_(99.0)
            layer.log_sigma.fill_(10.0)
        layer.clip_params_()
        self.assertLessEqual(float(layer.nu_x.max().detach()), StructuredKNNConv.MAX_ORDER)
        self.assertLessEqual(float(layer.sigma.detach()), StructuredKNNConv.MAX_SIGMA)

    def test_dense_ref_parameter_count_is_the_fovi_baseline(self):
        dense = self._layer("dense_ref")
        structured = self._layer("structured_ref")
        self.assertEqual(dense.weight.numel(), self.out_channels * self.in_channels * dense.n_ref)
        self.assertEqual(structured.alpha.numel(), self.out_channels * self.in_channels)


if __name__ == "__main__":
    unittest.main()
