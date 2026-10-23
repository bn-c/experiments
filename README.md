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
| [drf/kernels.py](drf/kernels.py) | Analytic Gaussian / DoG kernels and replicate-padded spatial shift |
| [drf/visualization.py](drf/visualization.py) | Channel colorization and the 4×4 dashboard renderer (needs `numpy`, `Pillow`) |
| [scripts/visualize_drf.py](scripts/visualize_drf.py) | CLI that renders channel decompositions for an image or video |
| [tests/test_frontend.py](tests/test_frontend.py) | Unit tests: geometry, resolution invariance, dual regime, DSGC tuning, looming, OMS, device migration |

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

## Running

```bash
python -m unittest discover tests
python scripts/visualize_drf.py --image sample.jpg --video clip.mp4
```
