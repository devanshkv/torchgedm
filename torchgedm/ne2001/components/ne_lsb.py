"""
Local Super Bubble (LSB) component

Implements neLSB from NE2001: ellipsoidal trough with rotation in xy-plane.
"""

import torch


def neLSB(x, y, z, data):
    """
    Calculate electron density in Local Super Bubble (ellipsoidal model).

    The LSB is modeled as a rotated ellipsoid. The rotation is applied in the
    xy-plane (rotation about z-axis by angle theta).

    Rotation coefficients:
        s = sin(theta), c = cos(theta)
        ap = (c/aa)^2 + (s/bb)^2
        bp = (s/aa)^2 + (c/bb)^2
        cp = 1/cc^2
        dp = 2*c*s*(1/aa^2 - 1/bb^2)

    Ellipsoid test:
        q = (x-xlsb)^2*ap + (y-ylsb)^2*bp + (z-zlsb)^2*cp + (x-xlsb)*(y-ylsb)*dp
        Inside if q <= 1.0

    Args:
        x, y, z: Galactocentric coordinates (kpc). Can be tensors of any shape.
        data: NE2001Data object containing LSB parameters

    Returns:
        ne: Electron density (cm^-3)
        F: Fluctuation parameter
        w: Weight (1 if inside, 0 if outside)

    References:
        NE2001 neLSB function (neLISM.NE2001.f, lines 215-304)
    """
    # Get parameters
    aa = data.alsb        # Major axis (kpc)
    bb = data.blsb        # Intermediate axis (kpc)
    cc = data.clsb        # Minor axis (kpc)
    xlsb = data.xlsb      # Center x (kpc)
    ylsb = data.ylsb      # Center y (kpc)
    zlsb = data.zlsb      # Center z (kpc)
    theta = data.thetalsb # Rotation angle (degrees)
    ne0 = data.nelsb      # Density (cm^-3)
    F0 = data.Flsb        # Fluctuation parameter

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
    dx = x - xlsb
    dy = y - ylsb
    dz = z - zlsb

    q = dx ** 2 * ap + dy ** 2 * bp + dz ** 2 * cp + dx * dy * dp

    # Inside if q <= 1.0
    inside = q <= 1.0

    # Return density, fluctuation, and weight
    ne = torch.where(inside, ne0, torch.zeros_like(x))
    F = torch.where(inside, F0, torch.zeros_like(x))
    w = torch.where(inside, torch.ones_like(x, dtype=torch.long), torch.zeros_like(x, dtype=torch.long))

    return ne, F, w
