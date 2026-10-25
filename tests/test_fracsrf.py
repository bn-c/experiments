"""Sanity tests for the FracSRF layer and FracSRF-ResNet-32."""

import math
import unittest

import torch

from fracsrf import FracSRFConv2d, FracSRFResNet32


class TestFracSRFConv2d(unittest.TestCase):
    def test_forward_shape_preserved_with_stride_1(self):
        layer = FracSRFConv2d(in_channels=3, out_channels=8, stride=1)
        x = torch.randn(2, 3, 16, 16)
        out = layer(x)
        self.assertEqual(out.shape, (2, 8, 16, 16))

    def test_forward_downsamples_with_stride_2(self):
        layer = FracSRFConv2d(in_channels=4, out_channels=4, stride=2)
        x = torch.randn(2, 4, 16, 16)
        out = layer(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)
        self.assertEqual(out.shape[2], 8)
        self.assertEqual(out.shape[3], 8)

    def test_gradients_flow_to_alpha_nu_and_sigma(self):
        layer = FracSRFConv2d(in_channels=3, out_channels=4, stride=1, learn_sigma=True)
        x = torch.randn(2, 3, 8, 8, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        self.assertIsNotNone(layer.alpha.grad)
        self.assertIsNotNone(layer.nu_x.grad)
        self.assertIsNotNone(layer.nu_y.grad)
        self.assertIsNotNone(layer.log_sigma.grad)
        self.assertTrue(torch.all(torch.isfinite(layer.alpha.grad)))

    def test_clip_orders_bounds_parameters(self):
        layer = FracSRFConv2d(in_channels=2, out_channels=2, stride=1)
        with torch.no_grad():
            layer.nu_x.fill_(100.0)
            layer.nu_y.fill_(-5.0)
        layer.clip_orders_()
        self.assertTrue(torch.all(layer.nu_x <= layer.MAX_ORDER))
        self.assertTrue(torch.all(layer.nu_y >= 0.0))

    def test_sigma_is_bounded(self):
        """
        Sigma bounds the structured kernel's footprint but does not show up in the
        parameter count, so an unclamped learned sigma could turn a cheap layer into a
        123x123 dense convolution from 49 parameters.
        """
        layer = FracSRFConv2d(in_channels=4, out_channels=4, stride=1)
        with torch.no_grad():
            layer.log_sigma.fill_(math.log(30.0))
        self.assertAlmostEqual(layer.sigma.item(), layer.MAX_SIGMA, places=5)
        kernel, radius = layer._build_kernel()
        self.assertLessEqual(kernel.shape[-1], 2 * int(math.ceil(2 * layer.MAX_SIGMA)) + 1)
        self.assertLessEqual(radius, int(math.ceil(2 * layer.MAX_SIGMA)))

        with torch.no_grad():
            layer.log_sigma.fill_(math.log(1e-3))
        self.assertAlmostEqual(layer.sigma.item(), layer.MIN_SIGMA, places=5)

        # clip_sigma_ writes the bound back into the parameter itself
        with torch.no_grad():
            layer.log_sigma.fill_(math.log(30.0))
        layer.clip_sigma_()
        self.assertAlmostEqual(layer.log_sigma.item(), math.log(layer.MAX_SIGMA), places=5)

    def test_init_is_variance_preserving(self):
        """
        A fresh layer must not start an order of magnitude quieter than a dense conv:
        the Gaussian-derivative basis has a far smaller L2 norm than its alpha implies.
        """
        torch.manual_seed(0)
        x = torch.randn(64, 16, 32, 32)
        frac = FracSRFConv2d(16, 16, stride=1)
        frac.eval()
        dense = torch.nn.Conv2d(16, 16, 3, padding=1)
        torch.nn.init.kaiming_normal_(dense.weight)
        dense.eval()
        with torch.no_grad():
            std_frac = frac(x).std().item()
            std_dense = dense(x).std().item()
        self.assertGreater(std_frac, 0.5 * std_dense,
                           f"FracSRF init too quiet: {std_frac:.3f} vs dense {std_dense:.3f}")
        self.assertLess(std_frac, 2.0 * std_dense,
                        f"FracSRF init too loud: {std_frac:.3f} vs dense {std_dense:.3f}")

    def test_kernel_is_deterministic_for_fixed_parameters(self):
        """Same parameters -> same kernel (extent/order derivation must not drift)."""
        layer = FracSRFConv2d(in_channels=2, out_channels=2, stride=1)
        k1, r1 = layer._build_kernel()
        k2, r2 = layer._build_kernel()
        self.assertEqual(r1, r2)
        self.assertTrue(torch.equal(k1, k2))


class TestFracSRFResNet32(unittest.TestCase):
    def test_forward_output_shape(self):
        model = FracSRFResNet32(num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        logits = model(x)
        self.assertEqual(logits.shape, (2, 10))

    def test_has_32_weight_layers(self):
        model = FracSRFResNet32(num_classes=10, n_blocks=5)
        from fracsrf.layers import FracSRFConv2d as FC
        conv_count = sum(1 for m in model.modules() if isinstance(m, FC))
        # conv1 + 3 stages * 5 blocks * 2 convs = 31 structured convs, plus the final
        # FC layer = the 6n+2 = 32 "weight layers" of the ResNet-32 family.
        self.assertEqual(conv_count, 31)

    def test_param_count_smaller_than_baseline_resnet32(self):
        model = FracSRFResNet32(num_classes=10)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # Baseline (non-structured) CIFAR ResNet-32 has ~0.47M params (paper Table 3).
        # NOTE: this compares *parameters* only. At sigma_init=1.0 each FracSRF kernel
        # spans 5x5 (radius = ceil(2*sigma)), so FLOPs are ~2.8x a 3x3 baseline's.
        self.assertLess(n_params, 470_000)

    def test_clip_orders_runs_on_full_model_and_bounds_sigma(self):
        model = FracSRFResNet32(num_classes=10)
        for m in model.modules():
            if isinstance(m, FracSRFConv2d):
                with torch.no_grad():
                    m.log_sigma.fill_(math.log(50.0))
        model.clip_orders_()  # should not raise
        for m in model.modules():
            if isinstance(m, FracSRFConv2d):
                self.assertLessEqual(m.sigma.item(), m.MAX_SIGMA + 1e-6)

    def test_sigma_param_groups_partition_all_params(self):
        model = FracSRFResNet32(num_classes=10, learn_sigma=True)
        sigma_ids = {id(p) for p in model.sigma_parameters()}
        non_sigma_ids = {id(p) for p in model.non_sigma_parameters()}
        all_ids = {id(p) for p in model.parameters()}
        self.assertEqual(sigma_ids | non_sigma_ids, all_ids)
        self.assertEqual(sigma_ids & non_sigma_ids, set())

    def test_step_reduces_loss(self):
        """A short optimisation run must actually learn (guards the init/scale changes)."""
        torch.manual_seed(0)
        model = FracSRFResNet32(num_classes=10)
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        loss_fn = torch.nn.CrossEntropyLoss()
        x = torch.randn(32, 3, 32, 32)
        y = torch.randint(0, 10, (32,))
        losses = []
        for _ in range(12):
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            model.clip_orders_()
            losses.append(loss.item())
        self.assertTrue(all(torch.isfinite(torch.tensor(losses))))
        self.assertLess(sum(losses[-3:]) / 3, sum(losses[:3]) / 3)


if __name__ == "__main__":
    unittest.main()
