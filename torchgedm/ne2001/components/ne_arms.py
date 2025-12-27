"""
Spiral arms component for NE2001 electron density model.

Implements 5 logarithmic spiral arms with cubic spline interpolation,
matching the FORTRAN ne_arms_log_mod function.

Author: torchgedm
"""

import torch
import numpy as np
from typing import Tuple
from ..utils import sech2
from ..spline import CubicSpline


# Constants
RAD = 57.2957795130823  # radians to degrees
NARMS = 5
NARMPOINTS = 1000  # Increased from 500 to ensure full spiral coverage
NN = 20  # Number of control points for each arm
KS = 3  # Step size for coarse search

# Arm remapping from Wainscoat order to TC93 order (GC outwards toward Sun)
ARMMAP = torch.tensor([1, 3, 4, 2, 5]) - 1  # Convert to 0-indexed


def generate_spiral_arm_paths(
    arm_a: torch.Tensor,
    arm_rmin: torch.Tensor,
    arm_thmin: torch.Tensor,
    arm_extent: torch.Tensor,
    n_control: int = NN,
    n_points: int = NARMPOINTS,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate densely sampled spiral arm paths using cubic spline interpolation.

    This matches the FORTRAN initialization in lines 274-350 of density.NE2001.f.

    Args:
        arm_a: Logarithmic spiral parameter for each arm, shape (5,)
        arm_rmin: Minimum radius for each arm, shape (5,)
        arm_thmin: Minimum angle for each arm (radians), shape (5,)
        arm_extent: Angular extent for each arm (radians), shape (5,)
        n_control: Number of control points to generate (default 20)
        n_points: Number of densely sampled points (default 500)
        device: Device to create tensors on

    Returns:
        arm_x: X coordinates of arm paths, shape (5, n_points)
        arm_y: Y coordinates of arm paths, shape (5, n_points)
        arm_kmax: Number of valid points for each arm, shape (5,)
    """
    narms = arm_a.shape[0]
    device = arm_a.device

    # Initialize output arrays
    arm_x = torch.zeros(narms, n_points, device=device)
    arm_y = torch.zeros(narms, n_points, device=device)
    arm_kmax = torch.zeros(narms, dtype=torch.long, device=device)

    # Create spline interpolator (stateful, reuses coefficients)
    spline = CubicSpline()

    for j in range(narms):
        # Generate control points for this arm
        # th1(n,j) = thmin(j) + (n-1) * extent(j) / (NN-1)
        theta_control = arm_thmin[j] + torch.arange(n_control, device=device) * arm_extent[j] / (n_control - 1)

        # Logarithmic spiral: r1(n,j) = rmin(j) * exp((th1(n,j) - thmin(j)) / a(j))
        r_control = arm_rmin[j] * torch.exp((theta_control - arm_thmin[j]) / arm_a[j])

        # Apply sculpting (modifications to specific arms)
        r_control = apply_arm_sculpting(j, theta_control * RAD, r_control, device)

        # Convert theta to degrees for spline interpolation (FORTRAN line 294)
        theta_control_deg = theta_control * RAD

        # Dense sampling using cubic spline interpolation
        # FORTRAN lines 330-346
        dth = 5.0 / r_control[0]  # Angular step size
        theta_dense = theta_control_deg[0] - 0.999 * dth

        # Compute spline coefficients (FORTRAN line 332: call with negative NN)
        _ = spline(theta_control_deg, r_control, torch.tensor([theta_dense], device=device), compute_coeffs=True)

        # Sample densely along the arm
        k = 0
        while k < n_points - 1:
            theta_dense = theta_dense + dth

            # Check if we've exceeded the arm extent
            if theta_dense > theta_control_deg[-1]:
                break

            # Evaluate spline at this theta
            r_dense = spline(theta_control_deg, r_control, torch.tensor([theta_dense], device=device), compute_coeffs=False)

            # Convert to Cartesian coordinates (FORTRAN lines 340-341)
            # arm(j,k,1) = -r*sin(th/rad)
            # arm(j,k,2) =  r*cos(th/rad)
            theta_rad = theta_dense / RAD
            arm_x[j, k] = -r_dense[0] * torch.sin(theta_rad)
            arm_y[j, k] = r_dense[0] * torch.cos(theta_rad)

            k += 1

        arm_kmax[j] = k

    return arm_x, arm_y, arm_kmax


def apply_arm_sculpting(
    arm_idx: int,
    theta_deg: torch.Tensor,
    r: torch.Tensor,
    device: str
) -> torch.Tensor:
    """
    Apply sculpting (distortions) to specific spiral arms at specific angles.

    This matches FORTRAN lines 296-322.

    Args:
        arm_idx: Arm index (0-4)
        theta_deg: Angle in degrees, shape (n,)
        r: Radius values, shape (n,)
        device: Device for tensors

    Returns:
        r_sculpted: Modified radius values, shape (n,)
    """
    r_sculpted = r.clone()

    # Get mapped arm number (TC93 numbering)
    jj = ARMMAP[arm_idx].item()

    # Arm 3 (TC93) sculpting (FORTRAN lines 297-313)
    if jj == 2:  # TC93 arm 3 (0-indexed: 2)
        # Region 1: 370-410 degrees
        mask1 = (theta_deg > 370) & (theta_deg <= 410)
        if mask1.any():
            arg = (theta_deg[mask1] - 390) * 180.0 / (40.0 * RAD)
            r_sculpted[mask1] = r[mask1] * (1.0 + 0.04 * torch.cos(arg))

        # Region 2: 315-370 degrees
        mask2 = (theta_deg > 315) & (theta_deg <= 370)
        if mask2.any():
            arg = (theta_deg[mask2] - 345) * 180.0 / (55.0 * RAD)
            r_sculpted[mask2] = r[mask2] * (1.0 - 0.07 * torch.cos(arg))

        # Region 3: 180-315 degrees
        mask3 = (theta_deg > 180) & (theta_deg <= 315)
        if mask3.any():
            arg = (theta_deg[mask3] - 260) * 180.0 / (135.0 * RAD)
            r_sculpted[mask3] = r[mask3] * (1.0 + 0.16 * torch.cos(arg))

    # Arm 2 (TC93) sculpting (FORTRAN lines 315-321)
    elif jj == 1:  # TC93 arm 2 (0-indexed: 1)
        # Region: 290-395 degrees
        mask = (theta_deg > 290) & (theta_deg <= 395)
        if mask.any():
            arg = (theta_deg[mask] - 350) * 180.0 / (105.0 * RAD)
            r_sculpted[mask] = r[mask] * (1.0 - 0.11 * torch.cos(arg))

    return r_sculpted


def compute_arm_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    arm_x: torch.Tensor,
    arm_y: torch.Tensor,
    arm_kmax: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute minimum distance from (x,y) point to each spiral arm.

    Implements the coarse + fine search with linear interpolation from
    FORTRAN lines 370-408.

    Args:
        x: X coordinates, shape (...)
        y: Y coordinates, shape (...)
        arm_x: Arm X coordinates, shape (5, n_points)
        arm_y: Arm Y coordinates, shape (5, n_points)
        arm_kmax: Number of valid points per arm, shape (5,)

    Returns:
        smin: Minimum distance to each arm, shape (..., 5)
        arm_idx: Index of closest point on each arm, shape (..., 5)
    """
    # Flatten input for easier processing
    orig_shape = x.shape
    x_flat = x.flatten()
    y_flat = y.flatten()
    n_pts = x_flat.shape[0]
    narms = arm_x.shape[0]

    smin = torch.full((n_pts, narms), 1e10, device=x.device)
    arm_closest_idx = torch.zeros((n_pts, narms), dtype=torch.long, device=x.device)

    for j in range(narms):
        kmax = arm_kmax[j].item()

        # Coarse search (FORTRAN lines 373-379)
        # Step by 2*KS+1 = 7 points
        k_coarse = torch.arange(1 + KS, kmax - KS, 2 * KS + 1, device=x.device)
        dx_coarse = x_flat.unsqueeze(1) - arm_x[j, k_coarse].unsqueeze(0)
        dy_coarse = y_flat.unsqueeze(1) - arm_y[j, k_coarse].unsqueeze(0)
        sq_coarse = dx_coarse ** 2 + dy_coarse ** 2

        # Find coarse minimum for each point
        sqmin_coarse, kk_coarse_idx = torch.min(sq_coarse, dim=1)
        kk_coarse = k_coarse[kk_coarse_idx]

        # Fine search (FORTRAN lines 380-388)
        # Search within ±2*KS = ±6 points of coarse minimum
        for i in range(n_pts):
            kmi = max(kk_coarse[i].item() - 2 * KS, 0)
            kma = min(kk_coarse[i].item() + 2 * KS, kmax - 1)

            k_fine = torch.arange(kmi, kma + 1, device=x.device)
            dx_fine = x_flat[i] - arm_x[j, k_fine]
            dy_fine = y_flat[i] - arm_y[j, k_fine]
            sq_fine = dx_fine ** 2 + dy_fine ** 2

            sqmin_fine, kk_fine_idx = torch.min(sq_fine, dim=0)
            kk = k_fine[kk_fine_idx].item()

            # Linear interpolation refinement (FORTRAN lines 389-407)
            if kk > 0 and kk < kmax - 1:
                sq1 = (x_flat[i] - arm_x[j, kk - 1]) ** 2 + (y_flat[i] - arm_y[j, kk - 1]) ** 2
                sq2 = (x_flat[i] - arm_x[j, kk + 1]) ** 2 + (y_flat[i] - arm_y[j, kk + 1]) ** 2

                if sq1 < sq2:
                    kl = kk - 1
                else:
                    kl = kk + 1

                # Compute line parameters
                dx_arm = arm_x[j, kk] - arm_x[j, kl]
                dy_arm = arm_y[j, kk] - arm_y[j, kl]

                if abs(dx_arm) > 1e-10:
                    emm = dy_arm / dx_arm
                    ebb = arm_y[j, kk] - emm * arm_x[j, kk]
                    exx = (x_flat[i] + emm * y_flat[i] - emm * ebb) / (1.0 + emm ** 2)

                    # Check if interpolation point is within segment
                    test = (exx - arm_x[j, kk]) / (arm_x[j, kl] - arm_x[j, kk])
                    if test < 0.0 or test > 1.0:
                        exx = arm_x[j, kk]

                    eyy = emm * exx + ebb
                else:
                    exx = arm_x[j, kk]
                    eyy = arm_y[j, kk]

                sqmin = (x_flat[i] - exx) ** 2 + (y_flat[i] - eyy) ** 2
                smin[i, j] = torch.sqrt(sqmin)
            else:
                smin[i, j] = torch.sqrt(sqmin_fine)

            arm_closest_idx[i, j] = kk

    # Reshape to original shape
    smin = smin.reshape(*orig_shape, narms)
    arm_closest_idx = arm_closest_idx.reshape(*orig_shape, narms)

    return smin, arm_closest_idx


def ne_arms(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    arm_x: torch.Tensor,
    arm_y: torch.Tensor,
    arm_kmax: torch.Tensor,
    na: torch.Tensor,
    ha: torch.Tensor,
    wa: torch.Tensor,
    Aa: torch.Tensor,
    Fa: torch.Tensor,
    narm: torch.Tensor,
    warm: torch.Tensor,
    harm: torch.Tensor,
    farm: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate spiral arm electron density contribution.

    Implements the NE_ARMS_LOG_MOD function from FORTRAN (lines 181-509).

    Args:
        x, y, z: Galactocentric coordinates in kpc, shape (...)
        arm_x, arm_y: Precomputed arm paths, shape (5, n_points)
        arm_kmax: Number of valid points per arm, shape (5,)
        na: Base spiral arm density scale
        ha: Vertical scale height for spiral arms
        wa: Radial scale width for spiral arms
        Aa: Radial cutoff for spiral arms
        Fa: Base fluctuation parameter
        narm: Density scale factor for each arm, shape (5,)
        warm: Width scale factor for each arm, shape (5,)
        harm: Height scale factor for each arm, shape (5,)
        farm: Fluctuation scale factor for each arm, shape (5,)

    Returns:
        ne_a: Spiral arm electron density in cm^-3, shape (...)
        F_a: Fluctuation parameter, shape (...)
        whicharm: Which arm dominates (1-5, 0 if none), shape (...)
        whicharm_weight: Distance-based weight for nearest arm, shape (...)
    """
    device = x.device
    rr = torch.sqrt(x ** 2 + y ** 2)

    # Compute angle in xy-plane (FORTRAN line 365)
    # thxy = atan2(-x, y) * rad
    thxy = torch.atan2(-x, y) * RAD
    # Ensure angle is in [0, 360)
    thxy = torch.where(thxy < 0, thxy + 360.0, thxy)

    # Initialize output
    nea = torch.zeros_like(x)
    whicharm = torch.zeros_like(x, dtype=torch.long)
    Farms = torch.zeros_like(x)
    sminmin = torch.full_like(x, 1e10)

    # Only compute if within vertical extent (FORTRAN line 369)
    z_mask = torch.abs(z / ha) < 10.0

    if z_mask.any():
        # Compute distances to all arms
        smin, _ = compute_arm_distance(x[z_mask], y[z_mask], arm_x, arm_y, arm_kmax)

        # Process each arm (FORTRAN lines 370-497)
        for j in range(NARMS):
            jj = ARMMAP[j].item()  # TC93 arm number

            # Check if close to this arm (within 3*wa)
            close_mask = smin[..., j] < 3.0 * wa

            if close_mask.any():
                # Gaussian weighting by distance (FORTRAN line 415)
                ga = torch.exp(-(smin[..., j][close_mask] / (warm[jj] * wa)) ** 2)

                # Radial dependence (FORTRAN lines 421-426)
                rr_close = rr[z_mask][close_mask]
                radial_mask = rr_close > Aa
                if radial_mask.any():
                    ga[radial_mask] = ga[radial_mask] * sech2((rr_close[radial_mask] - Aa) / 2.0)

                # Arm-specific reweighting (FORTRAN lines 429-492)
                thxy_close = thxy[z_mask][close_mask]
                ga = apply_arm_reweighting(jj, thxy_close, ga)

                # Add this arm's contribution (FORTRAN line 495)
                z_close = z[z_mask][close_mask]
                contrib = narm[jj] * na * ga * sech2(z_close / (harm[jj] * ha))

                # Accumulate into nea (need to map back to full array)
                full_idx = torch.where(z_mask)[0][torch.where(close_mask)[0]]
                nea.view(-1)[full_idx] += contrib

                # Track which arm is closest (FORTRAN lines 417-420)
                update_mask = smin[..., j][close_mask] < sminmin[z_mask][close_mask]
                if update_mask.any():
                    update_full_idx = full_idx[update_mask]
                    whicharm.view(-1)[update_full_idx] = j + 1  # 1-indexed
                    sminmin.view(-1)[update_full_idx] = smin[..., j][close_mask][update_mask]

    # Remap arm numbers and set fluctuation parameters (FORTRAN lines 502-507)
    mask_has_arm = whicharm > 0
    if mask_has_arm.any():
        # Remap from Wainscoat to TC93 numbering
        whicharm_remapped = torch.zeros_like(whicharm)
        for j in range(NARMS):
            whicharm_remapped[whicharm == (j + 1)] = ARMMAP[j] + 1

        whicharm = whicharm_remapped
        Farms[mask_has_arm] = Fa * farm[whicharm[mask_has_arm] - 1]

    return nea, Farms, whicharm, sminmin


def apply_arm_reweighting(
    arm_tc93: int,
    thxy: torch.Tensor,
    ga: torch.Tensor
) -> torch.Tensor:
    """
    Apply arm-specific reweighting factors at certain angles.

    This implements the complex arm reweighting from FORTRAN lines 429-492.

    Args:
        arm_tc93: Arm number in TC93 numbering (0-indexed)
        thxy: Azimuthal angle in degrees, shape (...)
        ga: Current weighting factor, shape (...)

    Returns:
        ga_reweighted: Modified weighting factor, shape (...)
    """
    ga_out = ga.clone()

    # Arm 3 (TC93) reweighting (FORTRAN lines 430-449)
    if arm_tc93 == 2:  # TC93 arm 3 (0-indexed: 2)
        th3a = 290.0
        th3b = 363.0
        fac3min = 0.0

        test3 = thxy - th3a
        test3 = torch.where(test3 < 0, test3 + 360.0, test3)

        mask = (test3 >= 0) & (test3 < (th3b - th3a))
        if mask.any():
            arg = 6.2831853 * (thxy[mask] - th3a) / (th3b - th3a)
            fac = (1.0 + fac3min + (1.0 - fac3min) * torch.cos(arg)) / 2.0
            fac = fac ** 4.0
            ga_out[mask] = ga[mask] * fac

    # Arm 2 (TC93) reweighting (FORTRAN lines 452-492)
    elif arm_tc93 == 1:  # TC93 arm 2 (0-indexed: 1)
        # First reweighting region: 340-370 degrees (weakening)
        th2a = 340.0
        th2b = 370.0
        fac2min = 0.1

        test2 = thxy - th2a
        test2 = torch.where(test2 < 0, test2 + 360.0, test2)

        mask = (test2 >= 0) & (test2 < (th2b - th2a))
        if mask.any():
            arg = 6.2831853 * (thxy[mask] - th2a) / (th2b - th2a)
            fac = (1.0 + fac2min + (1.0 - fac2min) * torch.cos(arg)) / 2.0
            ga_out[mask] = ga[mask] * fac

    return ga_out
