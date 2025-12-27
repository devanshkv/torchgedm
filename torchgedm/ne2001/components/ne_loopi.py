"""
Loop I component

Implements neLOOPI from NE2001: spheroid with enhanced shell,
truncated for z < 0.
"""

import torch


def neLOOPI(x, y, z, data):
    """
    Calculate electron density in Loop I (spheroid with shell).

    Loop I is modeled as a spheroid with:
    - Inner region (r <= rlpI): lower density
    - Shell region (rlpI < r <= rlpI + drlpI): enhanced density
    - Outside (r > rlpI + drlpI): zero density
    - Truncated for z < 0 (only exists above Galactic plane)

    Geometry:
    - Spherical distance from center: r = sqrt((x-xlpI)^2 + (y-ylpI)^2 + (z-zlpI)^2)
    - Three regions based on r:
      * r <= a1 (inner): ne = nelpI, F = FlpI
      * a1 < r <= a2 (shell): ne = dnelpI, F = dFlpI
      * r > a2 (outside): ne = 0, F = 0
    - All regions have w = 1 if inside (r <= a2) and z >= 0

    Args:
        x, y, z: Galactocentric coordinates (kpc). Can be tensors of any shape.
        data: NE2001Data object containing Loop I parameters

    Returns:
        ne: Electron density (cm^-3)
        F: Fluctuation parameter
        w: Weight (1 if inside, 0 if outside)

    References:
        NE2001 neLOOPI function (neLISM.NE2001.c, lines 607-671)
    """
    # Get parameters
    xlpI = data.xlpI      # Center x (kpc)
    ylpI = data.ylpI      # Center y (kpc)
    zlpI = data.zlpI      # Center z (kpc)
    rlpI = data.rlpI      # Inner radius (kpc)
    drlpI = data.drlpI    # Shell width (kpc)
    nelpI = data.nelpI    # Inner density (cm^-3)
    dnelpI = data.dnelpI  # Shell density (cm^-3)
    FlpI = data.FlpI      # Inner fluctuation parameter
    dFlpI = data.dFlpI    # Shell fluctuation parameter

    # Calculate radii
    a1 = rlpI             # Inner radius
    a2 = rlpI + drlpI     # Outer radius

    # Initialize outputs with zeros
    ne = torch.zeros_like(x)
    F = torch.zeros_like(x)
    w = torch.zeros_like(x, dtype=torch.long)

    # Truncate for z < 0
    valid_z = z >= 0.0

    # Calculate spherical distance from center
    r = torch.sqrt((x - xlpI)**2 + (y - ylpI)**2 + (z - zlpI)**2)

    # Three regions:
    # 1. Inside volume (r <= a1)
    inside_volume = valid_z & (r <= a1)
    ne = torch.where(inside_volume, nelpI, ne)
    F = torch.where(inside_volume, FlpI, F)
    w = torch.where(inside_volume, torch.ones_like(w), w)

    # 2. Inside shell (a1 < r <= a2)
    inside_shell = valid_z & (r > a1) & (r <= a2)
    ne = torch.where(inside_shell, dnelpI, ne)
    F = torch.where(inside_shell, dFlpI, F)
    w = torch.where(inside_shell, torch.ones_like(w), w)

    # 3. Outside (r > a2 or z < 0): already initialized to zero

    return ne, F, w
