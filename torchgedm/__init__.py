"""
TorchGEDM: PyTorch implementation of Galactic Electron Density Models

A pure PyTorch implementation of the NE2001 electron density model,
featuring GPU acceleration, batched operations, and differentiability.
"""

__version__ = "0.1.0"

from . import ne2001
from . import ne2001_wrapper
from .api import (
    dist_to_dm,
    dm_to_dist,
    calculate_electron_density_xyz,
    calculate_electron_density_lbr,
    clear_cache
)

__all__ = [
    "ne2001",
    "ne2001_wrapper",
    "__version__",
    "dist_to_dm",
    "dm_to_dist",
    "calculate_electron_density_xyz",
    "calculate_electron_density_lbr",
    "clear_cache",
]
