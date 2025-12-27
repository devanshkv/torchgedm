"""
Discrete clump component of NE2001 model

This component represents individual dense clumps in the ISM with
specific locations and properties read from neclumpN.NE2001.dat
"""

import torch


def galactic_to_xyz(l_deg, b_deg, d_kpc):
    """
    Convert galactic coordinates (l, b, d) to galactocentric (x, y, z).

    Args:
        l_deg: Galactic longitude [degrees]
        b_deg: Galactic latitude [degrees]
        d_kpc: Distance from Sun [kpc]

    Returns:
        x, y, z: Galactocentric coordinates [kpc]
    """
    # Convert degrees to radians
    l_rad = l_deg * (torch.pi / 180.0)
    b_rad = b_deg * (torch.pi / 180.0)

    # Galactocentric conversion (Sun at x=0, y=8.5 kpc)
    slc = torch.sin(l_rad)
    clc = torch.cos(l_rad)
    sbc = torch.sin(b_rad)
    cbc = torch.cos(b_rad)

    rgalc = d_kpc * cbc
    x = rgalc * slc
    y = 8.5 - rgalc * clc  # Sun at y=8.5
    z = d_kpc * sbc

    return x, y, z


def neclumpN(x, y, z, clump_l, clump_b, clump_dc, clump_nec,
             clump_Fc, clump_rc, clump_edge):
    """
    Calculate electron density from discrete clumps.

    This implements the NE2001 clump component, which represents individual
    dense clumps in the ISM. Each clump is modeled as a spherical Gaussian
    with either exponential rolloff (edge=0) or hard truncation (edge=1).

    Args:
        x: Galactocentric x coordinate [kpc], shape (..., )
        y: Galactocentric y coordinate [kpc], shape (..., )
        z: Galactocentric z coordinate [kpc], shape (..., )
        clump_l: Galactic longitude of clump centers [degrees], shape (n_clumps,)
        clump_b: Galactic latitude of clump centers [degrees], shape (n_clumps,)
        clump_dc: Distance of clumps from Sun [kpc], shape (n_clumps,)
        clump_nec: Peak electron density of clumps [cm^-3], shape (n_clumps,)
        clump_Fc: Fluctuation parameter, shape (n_clumps,)
        clump_rc: Clump radius at 1/e [kpc], shape (n_clumps,)
        clump_edge: Edge type (0=exponential, 1=hard edge), shape (n_clumps,)

    Returns:
        necN: Total electron density from all clumps [cm^-3], shape (..., )
        FcN: Fluctuation parameter of last hit clump, shape (..., )
        hitclump: Index of clump hit (0=no hit, j>0=j-th clump), shape (..., )

    Notes:
        - Vectorized over both input coordinates and clumps
        - For edge=0: density = nec * exp(-arg) for arg < 5, where arg = r²/rc²
        - For edge=1: density = nec for arg <= 1, zero otherwise
        - If multiple clumps overlap, densities sum and last hit clump is returned
        - All operations are differentiable

    Reference:
        NE2001 model (Cordes & Lazio 2002, 2003)
        Original FORTRAN/C: neclumpN.f, neclumpN.c
    """
    # Get device and original shape
    device = x.device
    orig_shape = x.shape

    # Get number of clumps
    n_clumps = clump_l.shape[0]

    # Handle empty clump case
    if n_clumps == 0:
        necN = torch.zeros_like(x)
        FcN = torch.zeros_like(x)
        hitclump = torch.zeros_like(x, dtype=torch.long)
        return necN, FcN, hitclump

    # Convert clump (l, b, d) to (x, y, z) coordinates
    xc, yc, zc = galactic_to_xyz(clump_l, clump_b, clump_dc)

    # Flatten input coordinates and add clump dimension
    # x, y, z: (..., ) -> (N, 1) where N = product of original shape
    x_flat = x.reshape(-1, 1)  # (N, 1)
    y_flat = y.reshape(-1, 1)  # (N, 1)
    z_flat = z.reshape(-1, 1)  # (N, 1)

    # Reshape clump parameters for broadcasting
    # (n_clumps,) -> (1, n_clumps)
    xc = xc.unsqueeze(0)  # (1, n_clumps)
    yc = yc.unsqueeze(0)  # (1, n_clumps)
    zc = zc.unsqueeze(0)  # (1, n_clumps)
    clump_nec_bc = clump_nec.unsqueeze(0)  # (1, n_clumps)
    clump_Fc_bc = clump_Fc.unsqueeze(0)  # (1, n_clumps)
    clump_rc_bc = clump_rc.unsqueeze(0)  # (1, n_clumps)
    clump_edge_bc = clump_edge.unsqueeze(0)  # (1, n_clumps)

    # Calculate distance from each point to each clump
    # Result shape: (N, n_clumps)
    dx = x_flat - xc
    dy = y_flat - yc
    dz = z_flat - zc

    # arg = (dx² + dy² + dz²) / rc²
    arg = (dx**2 + dy**2 + dz**2) / (clump_rc_bc**2)

    # Calculate density contribution from each clump
    # Shape: (N, n_clumps)
    density_contrib = torch.zeros_like(arg)

    # Edge type 0: exponential rolloff for arg < 5
    edge0_mask = (clump_edge_bc == 0) & (arg < 5.0)
    density_contrib = torch.where(
        edge0_mask,
        clump_nec_bc * torch.exp(-arg),
        density_contrib
    )

    # Edge type 1: hard edge at arg = 1
    edge1_mask = (clump_edge_bc == 1) & (arg <= 1.0)
    density_contrib = torch.where(
        edge1_mask,
        clump_nec_bc,
        density_contrib
    )

    # Sum density from all clumps
    # Shape: (N,)
    necN_flat = density_contrib.sum(dim=-1)

    # Find which clump was hit (last one if multiple)
    # hit_mask: (N, n_clumps) - True where clump contributes
    hit_mask = (edge0_mask | edge1_mask)

    # Find last hit clump index (1-indexed, 0=no hit)
    # Create indices: (1, 2, 3, ..., n_clumps) -> (1, n_clumps)
    clump_indices = torch.arange(1, n_clumps + 1, device=device).unsqueeze(0)

    # Get maximum index where hit (0 if no hit)
    hitclump_flat = torch.where(
        hit_mask,
        clump_indices,
        torch.zeros_like(clump_indices)
    ).max(dim=-1).values  # (N,)

    # Get fluctuation parameter from last hit clump
    # For each position, find Fc of the last hit clump
    FcN_flat = torch.zeros_like(necN_flat)
    for j in range(n_clumps):
        # Select Fc for positions where clump j+1 was the last hit
        mask = (hitclump_flat == j + 1)
        FcN_flat = torch.where(
            mask,
            clump_Fc[j] + 0.0 * necN_flat,  # Add 0*necN to preserve gradients
            FcN_flat
        )

    # Reshape back to original shape
    necN = necN_flat.reshape(orig_shape)
    FcN = FcN_flat.reshape(orig_shape)
    hitclump = hitclump_flat.reshape(orig_shape)

    return necN, FcN, hitclump
