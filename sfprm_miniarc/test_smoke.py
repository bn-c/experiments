#!/usr/bin/env python3
"""Cheap unit/smoke checks (no GPU, no full train)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.sfprm_core import SFPRM
from models.spherical_ops import geodesic_damping, project_to_sphere, slerp
from optim.muon import Muon, zeropower_via_newtonschulz5


def test_spherical_ops_shapes():
    u = project_to_sphere(torch.randn(2, 5, 8))
    v = project_to_sphere(torch.randn(2, 5, 8))
    lam = torch.tensor(0.3)
    out = slerp(u, v, lam)
    assert out.shape == u.shape
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    damped = geodesic_damping(u, v, torch.tensor(0.5))
    assert damped.shape == u.shape


def test_model_forward_and_param_count():
    model = SFPRM(vocab_size=11, d_model=64, max_seq_len=32)
    # Tiny d_model for speed; still exercises step/head.
    x = torch.randint(0, 11, (2, 32))
    z_in = model.get_input_embedding(x)
    z = model.step(z_in, z_in)
    logits = model.head(z)
    assert logits.shape == (2, 32, 10)
    assert model.count_parameters() > 0


def test_full_width_param_ballpark():
    model = SFPRM(vocab_size=11, d_model=512, max_seq_len=128)
    n = model.count_parameters()
    # Guide quotes ~6.83M, but the listed dims (d=512, d_ff=1536, weight-tied)
    # yield ~2.70M; assert the architecture we actually ship.
    assert 2.5e6 < n < 3.0e6, n


def test_muon_step():
    p = torch.nn.Parameter(torch.randn(16, 8))
    opt = Muon([p], lr=0.02, momentum=0.95)
    loss = (p**2).sum()
    loss.backward()
    opt.step()
    g = torch.randn(8, 8)
    o = zeropower_via_newtonschulz5(g)
    assert o.shape == g.shape


def main() -> int:
    test_spherical_ops_shapes()
    test_model_forward_and_param_count()
    test_full_width_param_ballpark()
    test_muon_step()
    print("test_smoke: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
