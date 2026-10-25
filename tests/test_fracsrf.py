"""Sanity tests for the FracSRF layer and FracSRF-ResNet-32."""

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
        # conv1 + 3 stages * 5 blocks * 2 convs = 31, plus the final FC layer = 32 total.
        self.assertEqual(conv_count, 31)

    def test_param_count_smaller_than_baseline_resnet32(self):
        model = FracSRFResNet32(num_classes=10)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # Baseline (non-structured) CIFAR ResNet-32 has ~0.47M params (paper Table 3).
        self.assertLess(n_params, 470_000)

    def test_clip_orders_runs_on_full_model(self):
        model = FracSRFResNet32(num_classes=10)
        model.clip_orders_()  # should not raise

    def test_sigma_param_groups_partition_all_params(self):
        model = FracSRFResNet32(num_classes=10, learn_sigma=True)
        sigma_ids = {id(p) for p in model.sigma_parameters()}
        non_sigma_ids = {id(p) for p in model.non_sigma_parameters()}
        all_ids = {id(p) for p in model.parameters()}
        self.assertEqual(sigma_ids | non_sigma_ids, all_ids)
        self.assertEqual(sigma_ids & non_sigma_ids, set())


if __name__ == "__main__":
    unittest.main()
