"""
Galactic center component of NE2001 model

This component represents the Galactic center as an ellipsoidal Gaussian
structure with constant density within the ellipsoid.
"""

import torch


def ne_gc(x, y, z, xgc, ygc, zgc, rgc, hgc, negc0, Fgc0=None):
    """
    Calculate electron density for Galactic center component.

    This implements the NE2001 Galactic center component, which has a
    cylindrical/ellipsoidal form centered at (xgc, ygc, zgc) with radius
    rgc and half-height hgc. The density is constant (negc0) inside the
    ellipsoid and zero outside.

    Formula:
        rr = sqrt((x-xgc)² + (y-ygc)²)
        zz = |z-zgc|
        If rr > rgc or zz > hgc: negc = 0
        Else if (rr/rgc)² + (zz/hgc)² <= 1: negc = negc0
        Else: negc = 0

    Args:
        x: Galactocentric x coordinate [kpc], shape (..., )
        y: Galactocentric y coordinate [kpc], shape (..., )
        z: Galactocentric z coordinate [kpc], shape (..., )
        xgc: x coordinate of GC center [kpc]
        ygc: y coordinate of GC center [kpc]
        zgc: z coordinate of GC center [kpc]
        rgc: Radial extent (1/e radius) [kpc]
        hgc: Vertical extent (1/e half-height) [kpc]
        negc0: Electron density inside ellipsoid [cm^-3]
        Fgc0: Fluctuation parameter (optional), returned unchanged if provided

    Returns:
        negc: Electron density [cm^-3], shape (..., )
        Fgc0: Fluctuation parameter (if Fgc0 input was not None)

    Notes:
        - All inputs are vectorized - x, y, z can have arbitrary leading dimensions
        - The density is constant inside the ellipsoid, zero outside
        - Component is differentiable with respect to all coordinate inputs
        - First checks if point is within bounding cylinder (rr <= rgc, zz <= hgc)
        - Then checks ellipsoidal condition: (rr/rgc)² + (zz/hgc)² <= 1

    Reference:
        NE2001 model (Cordes & Lazio 2002, 2003)
        Original FORTRAN: density.NE2001.f, function NE_GC
    """
    # Calculate cylindrical radius from GC center
    rr = torch.sqrt((x - xgc)**2 + (y - ygc)**2)

    # Calculate vertical distance from GC center
    zz = torch.abs(z - zgc)

    # Check bounding cylinder and ellipsoidal condition
    # Original FORTRAN truncates at rgc and hgc first, then checks ellipsoid
    within_cylinder = (rr <= rgc) & (zz <= hgc)

    # Ellipsoidal condition: (rr/rgc)² + (zz/hgc)² <= 1
    arg = (rr / rgc)**2 + (zz / hgc)**2
    within_ellipsoid = arg <= 1.0

    # Density is negc0 if within both cylinder and ellipsoid, else 0
    # Use torch.where with zero tensor that depends on coordinates to maintain gradients
    negc = torch.where(within_cylinder & within_ellipsoid,
                       negc0 + 0.0 * rr + 0.0 * zz,  # Constant but preserves gradients
                       0.0 * rr + 0.0 * zz)  # Zero but preserves gradients

    if Fgc0 is not None:
        return negc, Fgc0
    return negc
