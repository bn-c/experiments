# DRF — Dynamic Receptive Field Biomimetic Front-End

A zero-parameter, non-spiking visual front-end that maps arbitrary-resolution RGB
streams `(B, 3, H, W)` or video clips `(B, T, 3, H, W)` onto a fixed
`(B, 12, 128, 128)` tensor of 12 parallel retinal ganglion pathways
(sustained luminance, chromatic opponency, ON/OFF subunits, biphasic transient,
four direction-selective channels, object-motion sensitivity, looming).

## Layout

| Path | Contents |
| --- | --- |
| [drf/frontend.py](drf/frontend.py) | `DRFBiomimeticFrontEnd`, `DRFState` — the 5-stage model and its causal state machine |
| [drf/kernels.py](drf/kernels.py) | Analytic Gaussian kernels and replicate-padded spatial shift |
| [drf/visualization.py](drf/visualization.py) | Channel colorization and the 4×4 dashboard renderer (needs `numpy`, `Pillow`) |
| [fracsrf/basis.py](fracsrf/basis.py) | Shared separable Gaussian-derivative basis, used by the Cartesian and manifold filters |
| [fracsrf/manifold.py](fracsrf/manifold.py) | FOVI-style foveated sensor and `StructuredKNNConv`: dense, structured and analytic kernel mapping |
| [scripts/visualize_drf.py](scripts/visualize_drf.py) | CLI that renders channel decompositions for an image or video |
| [scripts/check_fovi_mapping.py](scripts/check_fovi_mapping.py) | CLI measuring the kernel-mapping error a structured reference kernel removes |
| [tests/test_frontend.py](tests/test_frontend.py) | Unit tests: geometry, resolution invariance, dual regime, DSGC tuning, looming, OMS, device migration |
| [tests/test_manifold.py](tests/test_manifold.py) | Unit tests: isotropic sampling, manifold isotropy, kernel mapping, structured/analytic parity |

## Usage

```python
import torch
from drf import DRFBiomimeticFrontEnd

frontend = DRFBiomimeticFrontEnd().eval()
frontend.reset()                                  # start a new clip

features = frontend(torch.rand(1, 3, 1080, 1920)) # (1, 12, 128, 128)
clip = frontend(torch.rand(1, 8, 3, 720, 1280))   # (1, 8, 12, 128, 128)
```

The module holds no trainable parameters and runs entirely under `torch.no_grad()`.
Temporal state is kept internally; pass an explicit `DRFState` (and `return_state=True`)
to manage streaming state yourself.

Non-square inputs are letterboxed to a square (`letterbox=True`, the default) before
Stage 0, using a flat bar colour taken once per clip from the first frame's border
strips. Wide/tall frames therefore keep their full field of view and aspect ratio
instead of being center-cropped or anisotropically squashed to fit the grid, and the
padding itself cannot inject motion signals into the temporal channels.

### Geometry

The stride is `S = max(1, round(min(H, W) / max(target_resolution)))`, i.e. it is
derived from the *configured output grid*, not from a fixed 128 — so a `(64, 64)`
front-end (what `DRFViT` uses) still down-samples the whole frame rather than
crop-zooming into its centre. Because `S` is rounded rather than floored, the
decimated grid lands within ±1/(2S) of the target, so the final grid snap only trims
or pads a sliver; the retained field of view is continuous in the input size.

## Running

```bash
python -m unittest discover tests
python scripts/visualize_drf.py --image sample.jpg --video clip.mp4
```

## ViT training (ROCm)

[drf/vit.py](drf/vit.py) feeds the DRF front-end's `(B, 12, 64, 64)` output into a small
ViT (`DRFViT`) and [scripts/train_vit_imagenet100.py](scripts/train_vit_imagenet100.py)
trains it on the Hugging Face `clane9/imagenet-100` subset. Dependencies (including an
AMD ROCm nightly build of PyTorch, installed from `download.pytorch.org`'s wheel index
rather than built from source) are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run python scripts/train_vit_imagenet100.py --epochs 30 --batch-size 128
```

## Structured filters on a foveated manifold (prototype)

[fracsrf/manifold.py](fracsrf/manifold.py) combines FracSRF's analytic kernels with
FOVI's foveated interface (Blauch, Alvarez & Konkle, ICML 2026,
[arXiv:2602.03766](https://arxiv.org/abs/2602.03766)). FOVI samples visual space through
a cortical magnification function `M(r) = k/(r+a)`, puts the samples on a 3-D cortical
manifold, and defines receptive fields as kNNs *on the manifold*; weight sharing comes
from a kernel mapping that rewrites each neighbourhood into a shared reference frame. In
the reference implementation (`fovi/arch/knn.py`, `compute_local_rf`) that mapping snaps
every neighbour to its **nearest** point of a square reference grid, which is why FOVI's
`doubleres` variant — 2× the grid, 4× the taps — is worth ~3% ImageNet-1K.

A FracSRF kernel is a function rather than a table, so it can be evaluated *at* the
neighbour positions instead of sampled from a grid. `StructuredKNNConv` exposes three
arms that differ only in that respect, so the cost of the snap can be measured:

```bash
python scripts/check_fovi_mapping.py                          # comparison table
python scripts/check_fovi_mapping.py --figure outputs/fovi_mapping.png
python scripts/check_fovi_mapping.py --a 0.03 --samples 600 --k 16   # stronger foveation
```

With `k=121` on a 396-sample sensor (a=0.5, 16°), the same filter in every arm:

| arm | ref side | taps | params | snap err | rel RMS vs analytic | fwd ms |
| --- | --- | --- | --- | --- | --- | --- |
| `dense_ref` (rf1) | 11 | 121 | 991 361 | 0.076 | 12.7% | 7.5 |
| `dense_ref` (rf2, FOVI default) | 22 | 484 | 3 965 057 | 0.036 | 6.1% | 28.1 |
| `structured_ref` (rf1 mapping) | 11 | 121 | 24 705 | 0.076 | 12.7% | 8.0 |
| `analytic` | — | — | 24 705 | 0 | 0% | 4.5 |

So the analytic arm removes the mapping error entirely at 1/40th of the rf1 parameters,
while being 1.7× faster than rf1 and 6× faster than the doubled-res grid that only halves
that error. Two caveats: the parameterisation caps expressiveness at one separable
fractional-order pair per kernel (a real risk in high-data regimes — FOVI Appendix 10.3
cites Jacobsen et al. 2016 on exactly this), and the kNN gather cost is untouched, which
is FOVI's own stated bottleneck. `scripts/check_fovi_mapping.py` also reports the
pre-stretch anisotropy of each neighbourhood, i.e. whether "one filter across the visual
field" (FOVI Fig. 3) is geometrically admissible at a given foveation strength.

Nothing here imports `fovi`; the coordinate spaces, grid convention, one-hot snapping and
padding behaviour are reimplemented and pinned by
[tests/test_manifold.py](tests/test_manifold.py), including a brute-force check of the
analytic forward against one explicitly built kernel per neighbourhood.

## ViT training (ROCm)

[drf/vit.py](drf/vit.py) feeds the DRF front-end's `(B, 12, 64, 64)` output into a small
ViT (`DRFViT`) and [scripts/train_vit_imagenet100.py](scripts/train_vit_imagenet100.py)
trains it on the Hugging Face `clane9/imagenet-100` subset. Dependencies (including an
AMD ROCm nightly build of PyTorch, installed from `download.pytorch.org`'s wheel index
rather than built from source) are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run python scripts/train_vit_imagenet100.py --epochs 30 --batch-size 128
```

The script fits `DRFViT`'s per-pathway feature range on the first
`--calibrate-batches` training batches before epoch 1 (resuming restores it from the
checkpoint), and seeds python/torch from `--seed`.

## Tests

`python -m unittest discover tests` (or `pytest`, if installed) runs the suite. The
regression tests cover the properties that are easy to break silently: field-of-view
continuity across strides, per-direction selectivity (including *orthogonal* channel
suppression), expansion-vs-translation looming, zero-sum colour opponency, state
migration through `.to()`, batch growth mid-stream, and the ViT/calibration wiring.
