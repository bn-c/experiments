"""
FracSRF: Frequency learning for structured CNN filters with Gaussian fractional
derivatives (Saldanha, Pintea, van Gemert & Tomen, BMVC 2021, arXiv:2111.06660).
"""

from .basis import hermite_gaussian_basis, interpolate_order, order_coefficients
from .layers import FracSRFConv2d
from .manifold import (
    SensorArray,
    StructuredKNNConv,
    cortical_coords,
    isotropic_sensor,
    knn,
    mapping_residual,
    neighbourhood_anisotropy,
    neighbourhood_coordinates,
    neighbourhood_frames,
    one_hot_mapping,
    reference_grid,
)
from .resnet import FracSRFBasicBlock, FracSRFResNet32

__all__ = [
    "FracSRFConv2d",
    "FracSRFBasicBlock",
    "FracSRFResNet32",
    "SensorArray",
    "StructuredKNNConv",
    "cortical_coords",
    "hermite_gaussian_basis",
    "interpolate_order",
    "isotropic_sensor",
    "knn",
    "mapping_residual",
    "neighbourhood_anisotropy",
    "neighbourhood_coordinates",
    "neighbourhood_frames",
    "one_hot_mapping",
    "order_coefficients",
    "reference_grid",
]
