"""
Low Density Region in Q1 (LDRQ1) component

Implements neLDRQ1 from NE2001: ellipsoidal trough with rotation in xy-plane.
"""

import torch


def neLDRQ1(x, y, z, data):
    """
    Calculate electron density in Low Density Region in Q1 (ellipsoidal model).

    The LDR is modeled as a rotated ellipsoid. The rotation is applied in the
    xy-plane (rotation about z-axis by angle theta).

    Rotation coefficients:
        s = sin(theta), c = cos(theta)
        ap = (c/aa)^2 + (s/bb)^2
        bp = (s/aa)^2 + (c/bb)^2
        cp = 1/cc^2
        dp = 2*c*s*(1/aa^2 - 1/bb^2)

    Ellipsoid test:
        q = (x-xldr)^2*ap + (y-yldr)^2*bp + (z-zldr)^2*cp + (x-xldr)*(y-yldr)*dp
        Inside if q <= 1.0

    Args:
        x, y, z: Galactocentric coordinates (kpc). Can be tensors of any shape.
        data: NE2001Data object containing LDR parameters

    Returns:
        ne: Electron density (cm^-3)
        F: Fluctuation parameter
        w: Weight (1 if inside, 0 if outside)

    References:
        NE2001 neLDRQ1 function (neLISM.NE2001.f, lines 123-212)
    """
    # Get parameters
    aa = data.aldr        # Major axis (kpc)
    bb = data.bldr        # Intermediate axis (kpc)
    cc = data.cldr        # Minor axis (kpc)
    xldr = data.xldr      # Center x (kpc)
    yldr = data.yldr      # Center y (kpc)
    zldr = data.zldr      # Center z (kpc)
    theta = data.thetaldr # Rotation angle (degrees)
    ne0 = data.neldr      # Density (cm^-3)
    F0 = data.Fldr        # Fluctuation parameter

    # Convert theta to radians
    theta_rad = theta * torch.pi / 180.0
    s = torch.sin(theta_rad)
    c = torch.cos(theta_rad)

    # Compute rotation coefficients
    ap = (c / aa) ** 2 + (s / bb) ** 2
    bp = (s / aa) ** 2 + (c / bb) ** 2
    cp = 1.0 / (cc ** 2)
    dp = 2.0 * c * s * (1.0 / (aa ** 2) - 1.0 / (bb ** 2))

    # Compute ellipsoid test
    dx = x - xldr
    dy = y - yldr
    dz = z - zldr

    q = dx ** 2 * ap + dy ** 2 * bp + dz ** 2 * cp + dx * dy * dp

    # Inside if q <= 1.0
    inside = q <= 1.0

    # Return density, fluctuation, and weight
    ne = torch.where(inside, ne0, torch.zeros_like(x))
    F = torch.where(inside, F0, torch.zeros_like(x))
    w = torch.where(inside, torch.ones_like(x, dtype=torch.long), torch.zeros_like(x, dtype=torch.long))

    return ne, F, w
