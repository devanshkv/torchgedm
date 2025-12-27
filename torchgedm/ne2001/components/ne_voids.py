"""
Discrete void component of NE2001 model

This component represents individual low-density voids in the ISM with
specific locations and ellipsoidal geometries read from nevoidN.NE2001.dat
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


def nevoidN(x, y, z, void_l, void_b, void_dv, void_nev, void_Fv,
            void_aav, void_bbv, void_ccv, void_thvy, void_thvz, void_edge):
    """
    Calculate electron density from discrete voids.

    This implements the NE2001 void component, which represents individual
    low-density voids in the ISM. Each void is modeled as an ellipsoid with
    rotation angles and either exponential rolloff (edge=0) or hard truncation (edge=1).

    The rotation matrix is: Λ_z * Λ_y (rotation around y-axis first, then z-axis)

    Λ_y =  c1  0  s1        Λ_z =  c2  s2   0
            0  1   0               -s2  c2   0
          -s1  0  c1                 0   0   1

    Λ_z*Λ_y =  c1*c2   s2   s1*c2
              -s2*c1   c2  -s1*s2
                 -s1    0      c1

    Args:
        x: Galactocentric x coordinate [kpc], shape (..., )
        y: Galactocentric y coordinate [kpc], shape (..., )
        z: Galactocentric z coordinate [kpc], shape (..., )
        void_l: Galactic longitude of void centers [degrees], shape (n_voids,)
        void_b: Galactic latitude of void centers [degrees], shape (n_voids,)
        void_dv: Distance of voids from Sun [kpc], shape (n_voids,)
        void_nev: Electron density in void [cm^-3], shape (n_voids,)
        void_Fv: Fluctuation parameter, shape (n_voids,)
        void_aav: Void major axis at 1/e [kpc], shape (n_voids,)
        void_bbv: Void minor axis at 1/e [kpc], shape (n_voids,)
        void_ccv: Void minor axis at 1/e [kpc], shape (n_voids,)
        void_thvy: Rotation angle around y-axis [degrees], shape (n_voids,)
        void_thvz: Rotation angle around z-axis [degrees], shape (n_voids,)
        void_edge: Edge type (0=exponential, 1=hard edge), shape (n_voids,)

    Returns:
        nevN: Electron density from voids [cm^-3], shape (..., )
        FvN: Fluctuation parameter of hit void, shape (..., )
        hitvoid: Index of void hit (0=no hit, j>0=j-th void), shape (..., )
        wvoid: Void weight (0=no hit, 1=hit), shape (..., )

    Notes:
        - Vectorized over both input coordinates and voids
        - For edge=0: density = nev * exp(-q) for q < 3, where q = ellipsoidal distance
        - For edge=1: density = nev for q <= 1, zero otherwise
        - Only one void can be hit (unlike clumps which sum)
        - All operations are differentiable

    Reference:
        NE2001 model (Cordes & Lazio 2002, 2003)
        Original FORTRAN/C: nevoidN.f, nevoidN.c
    """
    # Get device and original shape
    device = x.device
    orig_shape = x.shape

    # Get number of voids
    n_voids = void_l.shape[0]

    # Handle empty void case
    if n_voids == 0:
        nevN = torch.zeros_like(x)
        FvN = torch.zeros_like(x)
        hitvoid = torch.zeros_like(x, dtype=torch.long)
        wvoid = torch.zeros_like(x, dtype=torch.long)
        return nevN, FvN, hitvoid, wvoid

    # Convert void (l, b, d) to (x, y, z) coordinates
    xv, yv, zv = galactic_to_xyz(void_l, void_b, void_dv)

    # Precompute rotation matrix components
    # Convert angles to radians
    th1 = void_thvy * (torch.pi / 180.0)  # Rotation around y
    th2 = void_thvz * (torch.pi / 180.0)  # Rotation around z

    s1 = torch.sin(th1)
    c1 = torch.cos(th1)
    s2 = torch.sin(th2)
    c2 = torch.cos(th2)

    # Rotation matrix elements
    cc12 = c1 * c2
    ss12 = s1 * s2
    cs21 = c2 * s1
    cs12 = c1 * s2

    # Flatten input coordinates and add void dimension
    # x, y, z: (..., ) -> (N, 1) where N = product of original shape
    x_flat = x.reshape(-1, 1)  # (N, 1)
    y_flat = y.reshape(-1, 1)  # (N, 1)
    z_flat = z.reshape(-1, 1)  # (N, 1)

    # Reshape void parameters for broadcasting
    # (n_voids,) -> (1, n_voids)
    xv = xv.unsqueeze(0)  # (1, n_voids)
    yv = yv.unsqueeze(0)  # (1, n_voids)
    zv = zv.unsqueeze(0)  # (1, n_voids)
    void_nev_bc = void_nev.unsqueeze(0)  # (1, n_voids)
    void_Fv_bc = void_Fv.unsqueeze(0)  # (1, n_voids)
    void_aav_bc = void_aav.unsqueeze(0)  # (1, n_voids)
    void_bbv_bc = void_bbv.unsqueeze(0)  # (1, n_voids)
    void_ccv_bc = void_ccv.unsqueeze(0)  # (1, n_voids)
    void_edge_bc = void_edge.unsqueeze(0)  # (1, n_voids)
    cc12 = cc12.unsqueeze(0)  # (1, n_voids)
    ss12 = ss12.unsqueeze(0)  # (1, n_voids)
    cs21 = cs21.unsqueeze(0)  # (1, n_voids)
    cs12 = cs12.unsqueeze(0)  # (1, n_voids)
    s1 = s1.unsqueeze(0)  # (1, n_voids)
    c1 = c1.unsqueeze(0)  # (1, n_voids)
    s2 = s2.unsqueeze(0)  # (1, n_voids)
    c2 = c2.unsqueeze(0)  # (1, n_voids)

    # Calculate offset from void center
    # Result shape: (N, n_voids)
    dx = x_flat - xv
    dy = y_flat - yv
    dz = z_flat - zv

    # Apply rotation matrix and calculate ellipsoidal distance
    # Rotated coordinates
    dx_rot = cc12 * dx + s2 * dy + cs21 * dz
    dy_rot = -cs12 * dx + c2 * dy - ss12 * dz
    dz_rot = -s1 * dx + c1 * dz

    # Ellipsoidal distance: q = (x'/a)² + (y'/b)² + (z'/c)²
    q = (dx_rot / void_aav_bc)**2 + (dy_rot / void_bbv_bc)**2 + (dz_rot / void_ccv_bc)**2

    # Calculate density contribution from each void
    # Note: Only one void is active (unlike clumps which sum)
    # Shape: (N, n_voids)
    density_contrib = torch.zeros_like(q)

    # Edge type 0: exponential rolloff for q < 3
    edge0_mask = (void_edge_bc == 0) & (q < 3.0)
    density_contrib = torch.where(
        edge0_mask,
        void_nev_bc * torch.exp(-q),
        density_contrib
    )

    # Edge type 1: hard edge at q = 1
    edge1_mask = (void_edge_bc == 1) & (q <= 1.0)
    density_contrib = torch.where(
        edge1_mask,
        void_nev_bc,
        density_contrib
    )

    # Find which void was hit (only one, take first/last)
    # hit_mask: (N, n_voids) - True where void contributes
    hit_mask = (edge0_mask | edge1_mask)

    # Find last hit void index (1-indexed, 0=no hit)
    void_indices = torch.arange(1, n_voids + 1, device=device).unsqueeze(0)  # (1, n_voids)

    # Get maximum index where hit (0 if no hit)
    hitvoid_flat = torch.where(
        hit_mask,
        void_indices,
        torch.zeros_like(void_indices)
    ).max(dim=-1).values  # (N,)

    # For voids, only one void contributes (not summed like clumps)
    # Get density from the hit void
    nevN_flat = torch.zeros_like(x_flat.squeeze(-1))
    FvN_flat = torch.zeros_like(x_flat.squeeze(-1))

    for j in range(n_voids):
        mask = (hitvoid_flat == j + 1)
        # Extract density for this void at matching positions
        nevN_flat = torch.where(
            mask,
            density_contrib[:, j],
            nevN_flat
        )
        FvN_flat = torch.where(
            mask,
            void_Fv[j] + 0.0 * nevN_flat,  # Add 0*nevN to preserve gradients
            FvN_flat
        )

    # Void weight: 1 if hit, 0 otherwise
    wvoid_flat = torch.where(hitvoid_flat != 0,
                              torch.ones_like(hitvoid_flat),
                              torch.zeros_like(hitvoid_flat))

    # Reshape back to original shape
    nevN = nevN_flat.reshape(orig_shape)
    FvN = FvN_flat.reshape(orig_shape)
    hitvoid = hitvoid_flat.reshape(orig_shape)
    wvoid = wvoid_flat.reshape(orig_shape)

    return nevN, FvN, hitvoid, wvoid
