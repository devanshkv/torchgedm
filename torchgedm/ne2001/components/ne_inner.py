"""
Inner thin disk (annular) component of NE2001 model

This component represents the inner Galaxy thin disk as an annular (ring-shaped)
structure centered at radius A2, with a Gaussian radial profile and sech² vertical
profile. This is referred to as component 2 in the NE2001 model.
"""

import torch
from ..utils import sech2


def ne_inner(x, y, z, n2, h2, A2, F2=None):
    """
    Calculate electron density for inner thin disk (annular) component.

    This implements the NE2001 inner Galaxy component (ne2), which has an
    annular Gaussian form centered at radius A2 with scale length 1.8 kpc
    and vertical sech² profile with scale height h2.

    Formula:
        rr = sqrt(x² + y²)
        rrarg = ((rr - A2) / 1.8)²
        g2 = exp(-rrarg) if rrarg < 10, else 0
        ne2 = n2 * g2 * sech²(z/h2)

    Args:
        x: Galactocentric x coordinate [kpc], shape (..., )
        y: Galactocentric y coordinate [kpc], shape (..., )
        z: Galactocentric z coordinate [kpc], shape (..., )
        n2: Density coefficient [cm^-3]
        h2: Vertical scale height [kpc]
        A2: Peak radius of annular component [kpc]
        F2: Fluctuation parameter (optional), returned unchanged if provided

    Returns:
        ne2: Electron density [cm^-3], shape (..., )
        F2: Fluctuation parameter (if F2 input was not None)

    Notes:
        - All inputs are vectorized - x, y, z can have arbitrary leading dimensions
        - The radial scale length (1.8 kpc) is hardcoded as in original FORTRAN
        - The cutoff at rrarg=10 prevents numerical underflow (exp(-10) ≈ 4.5e-5)
        - Component is differentiable with respect to all inputs

    Reference:
        NE2001 model (Cordes & Lazio 2002, 2003)
        Original FORTRAN: density.NE2001.f, function NE_INNER
    """
    # Calculate cylindrical radius
    rr = torch.sqrt(x**2 + y**2)

    # Radial Gaussian function centered at A2 with scale 1.8 kpc
    # The 1.8 kpc scale length is hardcoded in original FORTRAN
    rrarg = ((rr - A2) / 1.8) ** 2

    # Apply cutoff to prevent numerical underflow
    # Original FORTRAN: if(rrarg.lt.10.0) g2=exp(-rrarg)
    g2 = torch.where(rrarg < 10.0,
                     torch.exp(-rrarg),
                     torch.zeros_like(rrarg))

    # Vertical sech² profile
    vertical_profile = sech2(z / h2)

    # Combined density
    ne2 = n2 * g2 * vertical_profile

    if F2 is not None:
        return ne2, F2
    return ne2
