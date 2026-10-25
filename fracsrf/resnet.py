"""
FracSRF-ResNet-32: the CIFAR-style ResNet-32 (He et al. 2016; 3 stages of
n=5 BasicBlocks with 16/32/64 channels, 6n+2=32 weight layers) with every
non-1x1 convolution replaced by a FracSRFConv2d structured filter, following
Saldanha et al. (BMVC 2021). 1x1 downsample shortcuts and the final linear
classifier are left as standard layers, matching the paper's Sec. 4.1 setup.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import FracSRFConv2d


class FracSRFBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        sigma_init: float = 1.0,
        order_init_range: Tuple[float, float] = (1.0, 3.0),
        learn_sigma: bool = True,
    ):
        super().__init__()
        common = dict(sigma_init=sigma_init, order_init_range=order_init_range, learn_sigma=learn_sigma)

        self.conv1 = FracSRFConv2d(in_planes, planes, stride=stride, bias=False, **common)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = FracSRFConv2d(planes, planes, stride=1, bias=False, **common)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class FracSRFResNet32(nn.Module):
    """FracSRF-ResNet-32 image classifier for 32x32 inputs (e.g. CIFAR-10/100)."""

    def __init__(
        self,
        num_classes: int = 10,
        n_blocks: int = 5,
        sigma_init: float = 1.0,
        order_init_range: Tuple[float, float] = (1.0, 3.0),
        learn_sigma: bool = True,
    ):
        super().__init__()
        self.in_planes = 16
        common = dict(sigma_init=sigma_init, order_init_range=order_init_range, learn_sigma=learn_sigma)

        self.conv1 = FracSRFConv2d(3, 16, stride=1, bias=False, **common)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(16, n_blocks, stride=1, **common)
        self.layer2 = self._make_layer(32, n_blocks, stride=2, **common)
        self.layer3 = self._make_layer(64, n_blocks, stride=2, **common)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64 * FracSRFBasicBlock.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _make_layer(self, planes: int, n_blocks: int, stride: int, **common) -> nn.Sequential:
        strides = [stride] + [1] * (n_blocks - 1)
        layers = []
        for s in strides:
            layers.append(FracSRFBasicBlock(self.in_planes, planes, stride=s, **common))
            self.in_planes = planes * FracSRFBasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out).flatten(1)
        return self.fc(out)

    def clip_orders_(self) -> None:
        """Clamp all FracSRF layers' fractional orders in-place; call after optimizer.step()."""
        for m in self.modules():
            if isinstance(m, FracSRFConv2d):
                m.clip_orders_()

    def sigma_parameters(self):
        """Yields the per-layer learnable `sigma` (log_sigma) parameters, for a separate
        optimizer param-group (the paper uses a distinct, smaller LR/weight-decay for sigma)."""
        for m in self.modules():
            if isinstance(m, FracSRFConv2d) and m.log_sigma.requires_grad:
                yield m.log_sigma

    def non_sigma_parameters(self):
        """Yields all other learnable parameters (alpha, nu_x, nu_y, BN, shortcut convs, FC)."""
        sigma_ids = {id(p) for p in self.sigma_parameters()}
        for p in self.parameters():
            if id(p) not in sigma_ids:
                yield p
