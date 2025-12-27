"""
Local Hot Bubble (LHB) component

Implements neLHB2 from NE2001: cylindrical model with slant in y-z plane
and varying cross-section with z.
"""

import torch


def neLHB(x, y, z, data):
    """
    Calculate electron density in Local Hot Bubble (cylindrical model).

    The LHB is modeled as a cylinder that:
    - Slants in the y direction vs z (yaxis = ylhb + yzslope*z)
    - Has cross-sectional area that varies with z
    - For z<=0 and z>=(zlhb-clhb), the major axis shrinks linearly toward 0
    - For other z, major axis = alhb (constant)

    Geometry test:
    - qxy = ((x-xlhb)/aa)^2 + ((y-yaxis)/bb)^2
    - qz = |z-zlhb|/cc
    - Inside if qxy <= 1.0 AND qz <= 1.0

    Args:
        x, y, z: Galactocentric coordinates (kpc). Can be tensors of any shape.
        data: NE2001Data object containing LHB parameters

    Returns:
        ne: Electron density (cm^-3)
        F: Fluctuation parameter
        w: Weight (1 if inside, 0 if outside)

    References:
        NE2001 neLHB2 function (neLISM.NE2001.f, lines 400-488)
    """
    # Get parameters
    aa_base = data.alhb  # Base major axis (kpc)
    bb = data.blhb       # Minor axis (kpc)
    cc = data.clhb       # Height scale (kpc)
    xlhb = data.xlhb     # Center x (kpc)
    ylhb = data.ylhb     # Center y (kpc)
    zlhb = data.zlhb     # Center z (kpc)
    theta = data.thetalhb  # Slant angle (degrees)
    ne0 = data.nelhb     # Density (cm^-3)
    F0 = data.Flhb       # Fluctuation parameter

    # Convert theta to radians and calculate slope
    theta_rad = theta * torch.pi / 180.0
    yzslope = torch.tan(theta_rad)

    # Calculate yaxis (shifts with z)
    yaxis = ylhb + yzslope * z

    # Calculate aa (varies with z for z < 0)
    # For z <= 0 and z >= zlhb-clhb: aa shrinks linearly
    # aa = 0.001 + (alhb - 0.001) * (1 - z/(zlhb - clhb))
    # For other z: aa = alhb
    aa = torch.where(
        (z <= 0.0) & (z >= zlhb - cc),
        0.001 + (aa_base - 0.001) * (1.0 - z / (zlhb - cc)),
        aa_base
    )

    # Cylindrical cross-section test
    qxy = ((x - xlhb) / aa) ** 2 + ((y - yaxis) / bb) ** 2

    # Height test
    qz = torch.abs(z - zlhb) / cc

    # Inside if both tests pass
    inside = (qxy <= 1.0) & (qz <= 1.0)

    # Return density, fluctuation, and weight
    ne = torch.where(inside, ne0, torch.zeros_like(x))
    F = torch.where(inside, F0, torch.zeros_like(x))
    w = torch.where(inside, torch.ones_like(x, dtype=torch.long), torch.zeros_like(x, dtype=torch.long))

    return ne, F, w
