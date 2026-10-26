"""Standalone Muon optimizer (Newton–Schulz orthogonalized momentum).

Faithful to Keller Jordan's Muon (https://kellerjordan.github.io/posts/muon/).
"""

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


def zeropower_via_newtonschulz5(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate matrix zeroth-power / orthogonalization via quintic Newton–Schulz."""
    assert g.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g.to(dtype=torch.bfloat16, copy=True)
    transposed = g.size(0) > g.size(1)
    if transposed:
        x = x.T
    x = x / (x.norm() + eps)
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * a_mat @ a_mat
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x


class Muon(Optimizer):
    """MomentUm Orthogonalized by Newton–Schulz for 2D matrix parameters."""

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError(f"Muon expects 2D params; got ndim={p.ndim}")

                grad = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                buf.lerp_(grad, 1 - momentum)
                update = grad.lerp(buf, momentum) if nesterov else buf

                update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                # Scale like Jordan Muon so wide/tall matrices share a similar effective step.
                update = update * max(1.0, update.size(-2) / update.size(-1)) ** 0.5
                update = update.to(dtype=p.dtype)

                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)
                p.add_(update, alpha=-lr)

        return loss
