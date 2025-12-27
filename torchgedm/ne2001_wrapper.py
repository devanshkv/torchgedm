"""
Python wrapper for TorchGEDM NE2001 implementation.

This module provides a pygedm-compatible ne2001_wrapper interface.
It wraps the TorchGEDM implementation to match pygedm's API.

References:
    [1] `Cordes, J. M., & Lazio, T. J. W. (2002)`
    *NE2001.I. A New Model for the Galactic Distribution of Free Electrons*
    [2] `Cordes, J. M., & Lazio, T. J. W. (2003)`
    *NE2001. II. Using Radio Propagation Data*
"""

from typing import Union, Dict, Tuple

# Try to import astropy for units support
try:
    from astropy.units import Quantity
    import astropy.units as u
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False

from .api import (
    dist_to_dm as _dist_to_dm_api,
    dm_to_dist as _dm_to_dist_api,
    calculate_electron_density_xyz as _calc_ne_xyz_api
)


def dm_to_dist(
    l: float,
    b: float,
    dm: float,
    nu: float = 1.0,
    full_output: bool = False
) -> Union[Tuple, Dict]:
    """
    Convert DM to distance and compute scattering timescale.

    Args:
        l: Galactic longitude in degrees
        b: Galactic latitude in degrees
        dm: Dispersion measure (pc/cm3)
        nu: Observing frequency (GHz)
        full_output: Return full raw output (dict) from NE2001 if set to True

    Returns:
        If full_output=False:
            Tuple of (dist, tau_sc):
            - dist: Distance (astropy Quantity in pc if astropy available)
            - tau_sc: Scattering timescale (astropy Quantity in s if astropy available)
        If full_output=True:
            Dictionary with all outputs
    """
    # Handle zero DM edge case
    if abs(dm) < 1e-10:
        if not full_output:
            if HAS_ASTROPY:
                return 0.0 * u.pc, 0.0 * u.s
            return 0.0, 0.0
        else:
            result = {
                'dist': 0.0,
                'dm': 0.0,
                'limit': ' ',
                'sm': 0.0,
                'smtau': 0.0,
                'smtheta': 0.0,
                'tau_sc': 0.0,
                'nu': nu
            }
            if HAS_ASTROPY:
                result['dist'] = result['dist'] * u.kpc
                result['tau_sc'] = result['tau_sc'] * u.s
            return result

    # Call the main API function
    if full_output:
        result = _dm_to_dist_api(l, b, dm, method='ne2001', nu=nu, full_output=True)
        # Convert dist from pc to kpc for wrapper compatibility
        if HAS_ASTROPY:
            result['dist'] = result['dist'].to('kpc')
        else:
            result['dist'] = result['dist'] / 1000.0
        return result
    else:
        dist_pc, tau_sc = _dm_to_dist_api(l, b, dm, method='ne2001', nu=nu, full_output=False)
        # Convert to kpc for wrapper compatibility
        if HAS_ASTROPY:
            return dist_pc.to('kpc'), tau_sc
        else:
            return dist_pc / 1000.0, tau_sc


def dist_to_dm(
    l: float,
    b: float,
    dist: float,
    nu: float = 1.0,
    full_output: bool = False
) -> Union[Tuple, Dict]:
    """
    Convert distance to DM and compute scattering timescale.

    Args:
        l: Galactic longitude in degrees
        b: Galactic latitude in degrees
        dist: Distance in kpc
        nu: Observing frequency (GHz)
        full_output: Return full raw output (dict) from NE2001 if set to True

    Returns:
        If full_output=False:
            Tuple of (dm, tau_sc):
            - dm: Dispersion measure (astropy Quantity in pc/cm3 if astropy available)
            - tau_sc: Scattering timescale (astropy Quantity in s if astropy available)
        If full_output=True:
            Dictionary with all outputs
    """
    # Handle zero distance edge case
    if abs(dist) < 1e-10:
        if not full_output:
            if HAS_ASTROPY:
                return 0.0 * u.pc / u.cm**3, 0.0 * u.s
            return 0.0, 0.0
        else:
            result = {
                'dm': 0.0,
                'dist': 0.0,
                'sm': 0.0,
                'smtau': 0.0,
                'smtheta': 0.0,
                'tau_sc': 0.0,
                'nu': nu
            }
            if HAS_ASTROPY:
                result['dm'] = result['dm'] * u.pc / u.cm**3
                result['tau_sc'] = result['tau_sc'] * u.s
            return result

    # Convert kpc to pc for the API call
    dist_pc = dist * 1000.0

    # Call the main API function
    if full_output:
        result = _dist_to_dm_api(l, b, dist_pc, method='ne2001', nu=nu, full_output=True)
        # dist is already in pc from the API
        return result
    else:
        return _dist_to_dm_api(l, b, dist_pc, method='ne2001', nu=nu, full_output=False)


def calculate_electron_density_xyz(
    x: float,
    y: float,
    z: float
):
    """
    Compute electron density at Galactocentric X, Y, Z coordinates.

    x, y, z are Galactocentric Cartesian coordinates, measured in kpc
    with the axes parallel to (l, b) = (90, 0), (180, 0), and (0, 90) degrees

    Args:
        x: Galactocentric coordinates in kpc
        y: Galactocentric coordinates in kpc
        z: Galactocentric coordinates in kpc

    Returns:
        ne_out: Electron density (astropy Quantity in cm^-3 if astropy available)
    """
    # Convert kpc to pc for the API call
    x_pc = x * 1000.0
    y_pc = y * 1000.0
    z_pc = z * 1000.0

    return _calc_ne_xyz_api(x_pc, y_pc, z_pc, method='ne2001')
