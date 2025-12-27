"""
NE2001 electron density model implementation in PyTorch
"""

from .data_loader import NE2001Data
from .density import density_2001, DensityResult
from .utils import sech2
from .spline import compute_spline_coefficients, evaluate_spline, cspline, CubicSpline
from .dmdsm import dist_to_dm, dm_to_dist, calculate_sm
from .scattering import (
    tauiss, scintbw, scintime, specbroad,
    theta_xgal, theta_gal, emission_measure, transition_frequency
)

__all__ = [
    'NE2001Data',
    'density_2001',
    'DensityResult',
    'sech2',
    'compute_spline_coefficients',
    'evaluate_spline',
    'cspline',
    'CubicSpline',
    'dist_to_dm',
    'dm_to_dist',
    'calculate_sm',
    'tauiss',
    'scintbw',
    'scintime',
    'specbroad',
    'theta_xgal',
    'theta_gal',
    'emission_measure',
    'transition_frequency',
]
