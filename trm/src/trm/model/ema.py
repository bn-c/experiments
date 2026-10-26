"""Exponential moving average of model weights."""

from __future__ import annotations

import copy

import torch
from torch import nn


class EMAModel:
    """Maintain a shadow copy of ``model`` updated with EMA (paper decay 0.999)."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for param in self.shadow.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for shadow_param, model_param in zip(self.shadow.parameters(), model.parameters(), strict=True):
            shadow_param.lerp_(model_param, 1.0 - self.decay)

    def state_dict(self) -> dict:
        return {'decay': self.decay, 'shadow': self.shadow.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state['decay']
        self.shadow.load_state_dict(state['shadow'])
