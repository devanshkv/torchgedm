"""
High-level API for TorchGEDM - PyTorch implementation of NE2001

This module provides a pygedm-compatible API for the TorchGEDM package.
Functions follow the same interface as pygedm for drop-in replacement.

Example:
    >>> import torchgedm
    >>> # Calculate DM to a distance of 1 kpc towards Galactic center
    >>> dm, tau_sc = torchgedm.dist_to_dm(0.0, 0.0, 1.0)
    >>> print(f"DM: {dm:.2f} pc/cm3, Scattering time: {tau_sc:.3f} s")
    >>>
    >>> # Calculate distance from DM
    >>> dist, tau_sc = torchgedm.dm_to_dist(0.0, 0.0, 100.0)
    >>> print(f"Distance: {dist:.2f} kpc")
"""

import torch
from typing import Union, Tuple, Optional, Dict, Any
from .ne2001 import (
    NE2001Data,
    dist_to_dm as _dist_to_dm_internal,
    dm_to_dist as _dm_to_dist_internal,
    density_2001,
    tauiss as _tauiss
)
from .ne2001.utils import galactic_to_galactocentric, galactocentric_to_galactic

# Try to import astropy for units support
try:
    from astropy.coordinates import Angle
    from astropy.units import Quantity, Unit
    import astropy.units as u
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False


# Singleton data loader
_data_cache = {}


def _get_data(device: str = 'cpu') -> NE2001Data:
    """Get or create NE2001Data instance for given device."""
    if device not in _data_cache:
        _data_cache[device] = NE2001Data(device=device)
    return _data_cache[device]


def _gl_gb_convert(gl, gb, unit='deg'):
    """Convert gl, gb astropy.Angle to floats with correct units.

    If not an astropy quantity, returns value unchanged.
    """
    if HAS_ASTROPY:
        if isinstance(gl, (Angle, Quantity)):
            gl = gl.to(unit).value
        if isinstance(gb, (Angle, Quantity)):
            gb = gb.to(unit).value
    return gl, gb


def _unit_convert(q, unit_str):
    """Convert astropy.Quantity to float with given unit.

    If not an astropy quantity, returns value unchanged.
    """
    if HAS_ASTROPY and isinstance(q, Quantity):
        q = q.to(unit_str).value
    return q


def _to_tensor(
    value: Union[float, int, torch.Tensor],
    device: str = 'cpu'
) -> torch.Tensor:
    """Convert value to torch.Tensor on specified device."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return torch.tensor(float(value), device=device, dtype=torch.float32)


def _to_float(value: torch.Tensor) -> float:
    """Convert tensor to Python float."""
    if value.numel() == 1:
        return float(value.item())
    else:
        raise ValueError("Cannot convert multi-element tensor to float")


def dist_to_dm(
    l: Union[float, torch.Tensor],
    b: Union[float, torch.Tensor],
    dist: Union[float, torch.Tensor],
    mode: str = 'gal',
    method: str = 'ne2001',
    nu: Union[float, torch.Tensor] = 1.0,
    device: str = 'cpu',
    full_output: bool = False
) -> Union[Tuple[float, float], Dict[str, float]]:
    """
    Convert distance to dispersion measure and scattering timescale.

    This function provides a pygedm-compatible interface for TorchGEDM.

    Args:
        l: Galactic longitude (degrees) or astropy Angle
        b: Galactic latitude (degrees) or astropy Angle
        dist: Distance from Sun (pc) or astropy Quantity
        mode: Calculation mode, only 'gal' supported for NE2001
        method: Model to use, only 'ne2001' supported
        nu: Radio frequency (GHz), default 1.0
        device: Device to run computation on ('cpu' or 'cuda')
        full_output: If True, return full dictionary with all outputs

    Returns:
        If full_output=False (default):
            Tuple of (dm, tau_sc) where:
            - dm: Dispersion measure (astropy Quantity in pc/cm3 if astropy available, else float)
            - tau_sc: Scattering timescale (astropy Quantity in seconds if astropy available, else float)
        If full_output=True:
            Dictionary with keys (all as astropy Quantities if available, else floats)

    Examples:
        >>> # Basic usage
        >>> dm, tau_sc = dist_to_dm(0.0, 0.0, 1000.0)  # distance in pc
        >>> print(f"DM: {dm:.2f} pc/cm3")
        >>>
        >>> # With astropy units
        >>> import astropy.units as u
        >>> dm, tau_sc = dist_to_dm(0.0, 0.0, 1.0 * u.kpc)
        >>> print(f"DM: {dm}")
    """
    # Validate method
    if method.lower() != 'ne2001':
        raise RuntimeError("Only 'ne2001' method is supported by TorchGEDM.")

    # Validate mode
    if mode != 'gal':
        raise RuntimeError("NE2001 only supports Galactic (gal) mode.")

    # Convert astropy angles to degrees
    l, b = _gl_gb_convert(l, b, 'deg')

    # Convert distance from pc to kpc (torchgedm works in kpc internally)
    dist = _unit_convert(dist, 'pc')
    dist_kpc = dist / 1000.0

    # Convert frequency to GHz
    nu = _unit_convert(nu, 'GHz')

    # Handle zero distance edge case
    if abs(dist_kpc) < 1e-10:
        if not full_output:
            dm_val = 0.0
            tau_val = 0.0
            if HAS_ASTROPY:
                return dm_val * u.pc / u.cm**3, tau_val * u.s
            return dm_val, tau_val
        else:
            result = {'dm': 0.0, 'dist': 0.0, 'sm': 0.0, 'smtau': 0.0,
                     'smtheta': 0.0, 'tau_sc': 0.0, 'nu': nu}
            if HAS_ASTROPY:
                result['dm'] = result['dm'] * u.pc / u.cm**3
                result['dist'] = result['dist'] * u.pc
                result['tau_sc'] = result['tau_sc'] * u.s
            return result

    # Convert inputs to tensors
    l_t = _to_tensor(l, device)
    b_t = _to_tensor(b, device)
    dist_t = _to_tensor(dist_kpc, device)
    nu_t = _to_tensor(nu, device)

    # Get model data
    data = _get_data(device)

    # Call internal function
    dm_t, sm_t, smtau_t, smtheta_t = _dist_to_dm_internal(
        l_t, b_t, dist_t, data, full_output=True
    )

    # Calculate scattering timescale using smtau (for pulse broadening)
    tau_sc_t = _tauiss(dist_t, smtau_t, nu_t)

    # Convert results to Python floats
    dm = _to_float(dm_t)
    tau_sc = _to_float(tau_sc_t)

    if not full_output:
        if HAS_ASTROPY:
            return dm * u.pc / u.cm**3, tau_sc * u.s
        return dm, tau_sc

    # Return full output dictionary
    result = {
        'dm': dm,
        'dist': dist,  # Return in pc as per pygedm convention
        'sm': _to_float(sm_t),
        'smtau': _to_float(smtau_t),
        'smtheta': _to_float(smtheta_t),
        'tau_sc': tau_sc,
        'nu': nu
    }

    # Add astropy units if available
    if HAS_ASTROPY:
        result['dm'] = result['dm'] * u.pc / u.cm**3
        result['dist'] = result['dist'] * u.pc
        result['tau_sc'] = result['tau_sc'] * u.s

    return result


def dm_to_dist(
    l: Union[float, torch.Tensor],
    b: Union[float, torch.Tensor],
    dm: Union[float, torch.Tensor],
    dm_host: float = 0.0,
    mode: str = 'gal',
    method: str = 'ne2001',
    nu: Union[float, torch.Tensor] = 1.0,
    device: str = 'cpu',
    full_output: bool = False
) -> Union[Tuple[float, float], Dict[str, Any]]:
    """
    Convert dispersion measure to distance and scattering timescale.

    This function provides a pygedm-compatible interface for TorchGEDM.

    Args:
        l: Galactic longitude (degrees) or astropy Angle
        b: Galactic latitude (degrees) or astropy Angle
        dm: Dispersion measure (pc/cm3) or astropy Quantity
        dm_host: Host galaxy DM contribution to subtract (default 0)
        mode: Calculation mode, only 'gal' supported for NE2001
        method: Model to use, only 'ne2001' supported
        nu: Radio frequency (GHz), default 1.0
        device: Device to run computation on ('cpu' or 'cuda')
        full_output: If True, return full dictionary with all outputs

    Returns:
        If full_output=False (default):
            Tuple of (dist, tau_sc) where:
            - dist: Distance (astropy Quantity in pc if astropy available, else float in kpc)
            - tau_sc: Scattering timescale (astropy Quantity in s if astropy available, else float)
        If full_output=True:
            Dictionary with keys (all as astropy Quantities if available, else floats)

    Examples:
        >>> # Basic usage
        >>> dist, tau_sc = dm_to_dist(0.0, 0.0, 100.0)
        >>> print(f"Distance: {dist:.2f}")
        >>>
        >>> # With astropy units
        >>> import astropy.units as u
        >>> dist, tau_sc = dm_to_dist(0.0, 0.0, 100.0 * u.pc / u.cm**3)
        >>> print(f"Distance: {dist}")
    """
    # Validate method
    if method.lower() != 'ne2001':
        raise RuntimeError("Only 'ne2001' method is supported by TorchGEDM.")

    # Validate mode
    if mode != 'gal':
        raise RuntimeError("NE2001 only supports Galactic (gal) mode.")

    # Convert astropy angles to degrees
    l, b = _gl_gb_convert(l, b, 'deg')

    # Convert DM to pc/cm3 and subtract host contribution
    dm = _unit_convert(dm, 'pc cm^(-3)')
    dm = dm - dm_host

    # Convert frequency to GHz
    nu = _unit_convert(nu, 'GHz')

    # Handle zero DM edge case
    if abs(dm) < 1e-10:
        if not full_output:
            dist_val = 0.0
            tau_val = 0.0
            if HAS_ASTROPY:
                return dist_val * u.pc, tau_val * u.s
            return dist_val, tau_val
        else:
            result = {'dist': 0.0, 'dm': 0.0, 'limit': ' ', 'sm': 0.0,
                     'smtau': 0.0, 'smtheta': 0.0, 'tau_sc': 0.0, 'nu': nu}
            if HAS_ASTROPY:
                result['dist'] = result['dist'] * u.pc
                result['dm'] = result['dm'] * u.pc / u.cm**3
                result['tau_sc'] = result['tau_sc'] * u.s
            return result

    # Convert inputs to tensors
    l_t = _to_tensor(l, device)
    b_t = _to_tensor(b, device)
    dm_t = _to_tensor(dm, device)
    nu_t = _to_tensor(nu, device)

    # Get model data
    data = _get_data(device)

    # Call internal function
    dist_t, is_limit, sm_t, smtau_t, smtheta_t = _dm_to_dist_internal(
        l_t, b_t, dm_t, data, full_output=True
    )

    # Calculate scattering timescale using smtau (for pulse broadening)
    tau_sc_t = _tauiss(dist_t, smtau_t, nu_t)

    # Convert results to Python floats
    dist_kpc = _to_float(dist_t)
    dist_pc = dist_kpc * 1000.0  # Convert to pc for pygedm compatibility
    tau_sc = _to_float(tau_sc_t)

    if not full_output:
        if HAS_ASTROPY:
            return dist_pc * u.pc, tau_sc * u.s
        return dist_kpc, tau_sc  # Return kpc for backward compatibility

    # Return full output dictionary
    limit_str = '>' if is_limit else ' '
    result = {
        'dist': dist_pc,  # Return in pc as per pygedm convention
        'dm': dm,
        'limit': limit_str,
        'sm': _to_float(sm_t),
        'smtau': _to_float(smtau_t),
        'smtheta': _to_float(smtheta_t),
        'tau_sc': tau_sc,
        'nu': nu
    }

    # Add astropy units if available
    if HAS_ASTROPY:
        result['dist'] = result['dist'] * u.pc
        result['dm'] = result['dm'] * u.pc / u.cm**3
        result['tau_sc'] = result['tau_sc'] * u.s

    return result


def calculate_electron_density_xyz(
    x: Union[float, torch.Tensor],
    y: Union[float, torch.Tensor],
    z: Union[float, torch.Tensor],
    method: str = 'ne2001',
    device: str = 'cpu'
) -> float:
    """
    Calculate electron density at Galactocentric Cartesian coordinates.

    Coordinates are Galactocentric Cartesian (NOT heliocentric), measured in pc (or astropy Quantity)
    with axes parallel to (l, b) = (90°, 0°), (180°, 0°), and (0°, 90°).

    Args:
        x: Galactocentric X coordinate (pc) or astropy Quantity
        y: Galactocentric Y coordinate (pc) or astropy Quantity
        z: Galactocentric Z coordinate (pc) or astropy Quantity
        method: Model to use, only 'ne2001' supported
        device: Device to run computation on ('cpu' or 'cuda')

    Returns:
        Electron density (astropy Quantity in cm^-3 if astropy available, else float)

    Examples:
        >>> # Density at Sun's position (approximately)
        >>> ne = calculate_electron_density_xyz(8500.0, 0.0, 0.0)
        >>> print(f"ne at Sun: {ne:.3f}")
        >>>
        >>> # Density at Galactic center
        >>> ne = calculate_electron_density_xyz(0.0, 0.0, 0.0)
        >>> print(f"ne at GC: {ne:.3f}")
    """
    # Validate method
    if method.lower() != 'ne2001':
        raise RuntimeError("Only 'ne2001' method is supported by TorchGEDM.")

    # Convert from pc to kpc (torchgedm works in kpc internally)
    x = _unit_convert(x, 'pc')
    y = _unit_convert(y, 'pc')
    z = _unit_convert(z, 'pc')
    x_kpc = x / 1000.0
    y_kpc = y / 1000.0
    z_kpc = z / 1000.0

    # Convert inputs to tensors
    x_t = _to_tensor(x_kpc, device)
    y_t = _to_tensor(y_kpc, device)
    z_t = _to_tensor(z_kpc, device)

    # Get model data
    data = _get_data(device)

    # Calculate density
    result = density_2001(x_t, y_t, z_t, data)
    ne = result.total_density(data)

    ne_val = _to_float(ne)

    if HAS_ASTROPY:
        return ne_val / u.cm**3
    return ne_val


def calculate_electron_density_lbr(
    l: Union[float, torch.Tensor],
    b: Union[float, torch.Tensor],
    r: Union[float, torch.Tensor],
    method: str = 'ne2001',
    device: str = 'cpu'
) -> float:
    """
    Calculate electron density at Galactic coordinates.

    Args:
        l: Galactic longitude (degrees) or astropy Angle
        b: Galactic latitude (degrees) or astropy Angle
        r: Distance from Sun (pc) or astropy Quantity
        method: Model to use, only 'ne2001' supported
        device: Device to run computation on ('cpu' or 'cuda')

    Returns:
        Electron density (astropy Quantity in cm^-3 if astropy available, else float)

    Examples:
        >>> # Density at 1 kpc towards Galactic center
        >>> ne = calculate_electron_density_lbr(0.0, 0.0, 1000.0)
        >>> print(f"ne: {ne:.3f}")
    """
    # Validate method
    if method.lower() != 'ne2001':
        raise RuntimeError("Only 'ne2001' method is supported by TorchGEDM.")

    # Convert astropy angles to degrees
    l, b = _gl_gb_convert(l, b, 'deg')

    # Convert distance from pc to kpc (torchgedm works in kpc internally)
    r = _unit_convert(r, 'pc')
    r_kpc = r / 1000.0

    # Convert inputs to tensors
    l_t = _to_tensor(l, device)
    b_t = _to_tensor(b, device)
    r_t = _to_tensor(r_kpc, device)

    # Convert to Galactocentric coordinates
    x, y, z = galactic_to_galactocentric(l_t, b_t, r_t)

    # Get model data
    data = _get_data(device)

    # Calculate density
    result = density_2001(x, y, z, data)
    ne = result.total_density(data)

    ne_val = _to_float(ne)

    if HAS_ASTROPY:
        return ne_val / u.cm**3
    return ne_val


def clear_cache():
    """
    Clear the cached NE2001Data instances.

    Useful for freeing GPU memory or forcing data reload.

    Example:
        >>> import torchgedm
        >>> # ... do some computations ...
        >>> torchgedm.clear_cache()  # Free GPU memory
    """
    global _data_cache
    _data_cache.clear()
