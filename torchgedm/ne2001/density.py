"""
DENSITY_2001: Combined electron density calculation for NE2001 model

This module implements the main density calculation that combines all 7 components
of the NE2001 electron density model.

Reference:
    density.NE2001.f (FORTRAN source, lines 12-121)
"""

import torch
from typing import Tuple, Dict, Any
from .components import (
    ne_outer, ne_inner, ne_gc, ne_arms,
    neLISM, neclumpN, nevoidN
)
from .data_loader import NE2001Data


class DensityResult:
    """
    Container for DENSITY_2001 results.

    Provides convenient access to all density components, fluctuation
    parameters, and flags returned by the model.
    """

    def __init__(
        self,
        # Densities (cm^-3)
        ne1: torch.Tensor,
        ne2: torch.Tensor,
        nea: torch.Tensor,
        negc: torch.Tensor,
        nelism: torch.Tensor,
        necN: torch.Tensor,
        nevN: torch.Tensor,
        # Fluctuation parameters
        F1: torch.Tensor,
        F2: torch.Tensor,
        Fa: torch.Tensor,
        Fgc: torch.Tensor,
        Flism: torch.Tensor,
        FcN: torch.Tensor,
        FvN: torch.Tensor,
        # Flags and weights
        whicharm: torch.Tensor,
        wlism: torch.Tensor,
        wLDR: torch.Tensor,
        wLHB: torch.Tensor,
        wLSB: torch.Tensor,
        wLOOPI: torch.Tensor,
        hitclump: torch.Tensor,
        hitvoid: torch.Tensor,
        wvoid: torch.Tensor
    ):
        # Electron densities (cm^-3)
        self.ne1 = ne1          # Outer thick disk
        self.ne2 = ne2          # Inner thin disk
        self.nea = nea          # Spiral arms
        self.negc = negc        # Galactic center
        self.nelism = nelism    # Local ISM (combined)
        self.necN = necN        # Discrete clumps
        self.nevN = nevN        # Voids

        # Fluctuation parameters (for scattering)
        self.F1 = F1
        self.F2 = F2
        self.Fa = Fa
        self.Fgc = Fgc
        self.Flism = Flism
        self.FcN = FcN
        self.FvN = FvN

        # Flags and weights
        self.whicharm = whicharm      # Which arm (0-5, 0=none)
        self.wlism = wlism            # 1 if in any LISM component
        self.wLDR = wLDR              # 1 if in LDR
        self.wLHB = wLHB              # 1 if in LHB
        self.wLSB = wLSB              # 1 if in LSB
        self.wLOOPI = wLOOPI          # 1 if in Loop I
        self.hitclump = hitclump      # Clump index (0=none)
        self.hitvoid = hitvoid        # Void index (0=none)
        self.wvoid = wvoid            # Void weight

    def total_density(self, data: 'NE2001Data') -> torch.Tensor:
        """
        Calculate total electron density with proper override logic.

        Implements FORTRAN logic from dmdsm.NE2001.f lines 249-256:
        - wlism=1: LISM overrides smooth Galactic components (ne1, ne2, nea, negc)
        - wvoid=1: Voids override everything except clumps

        Args:
            data: NE2001Data object containing component weights (wg1, wg2, etc.)

        Returns:
            Total ne (cm^-3), shape same as input coordinates

        Reference:
            dmdsm.NE2001.f lines 249-256
        """
        # Get component weights as scalars
        wg1 = data.wg1.item()
        wg2 = data.wg2.item()
        wga = data.wga.item()
        wggc = data.wggc.item()
        wglism = data.wglism.item()
        wgcN = data.wgcN.item()
        wgvN = data.wgvN.item()

        # FORTRAN line 249-255:
        # ne = (1.-wglism*wlism)*(wg1*ne1 + wg2*ne2 + wga*nea + wggc*negc) + wglism*wlism*nelism
        smooth_components = (
            wg1 * self.ne1 +
            wg2 * self.ne2 +
            wga * self.nea +
            wggc * self.negc
        )

        # LISM override: if wlism=1, LISM replaces smooth components
        ne = (1.0 - wglism * self.wlism) * smooth_components + wglism * self.wlism * self.nelism

        # FORTRAN line 256:
        # ne = (1-wgvN*wvoid)*ne + wgvN*wvoid*nevN + wgcN*necN
        ne = (1 - wgvN * self.wvoid) * ne + wgvN * self.wvoid * self.nevN + wgcN * self.necN

        return ne

    def total_fluctuation(self, data: 'NE2001Data') -> torch.Tensor:
        """
        Calculate total fluctuation parameter with proper override logic.

        Implements similar logic to total_density but for F-parameters.
        Used for scattering measure calculations: SM = integral of (F * ne^2) ds

        Args:
            data: NE2001Data object containing component weights (wg1, wg2, etc.)

        Returns:
            Total F parameter, shape same as input coordinates
        """
        # Get component weights as scalars
        wg1 = data.wg1.item()
        wg2 = data.wg2.item()
        wga = data.wga.item()
        wggc = data.wggc.item()
        wglism = data.wglism.item()
        wgcN = data.wgcN.item()
        wgvN = data.wgvN.item()

        # Smooth Galactic components
        smooth_F = (
            wg1 * self.F1 +
            wg2 * self.F2 +
            wga * self.Fa +
            wggc * self.Fgc
        )

        # LISM override: if wlism=1, LISM F replaces smooth components
        F = (1.0 - wglism * self.wlism) * smooth_F + wglism * self.wlism * self.Flism

        # Void override: if wvoid=1, void F replaces everything except clumps
        F = (1 - wgvN * self.wvoid) * F + wgvN * self.wvoid * self.FvN + wgcN * self.FcN

        return F

    def as_dict(self) -> Dict[str, torch.Tensor]:
        """Return all results as a dictionary."""
        return {
            'ne1': self.ne1, 'ne2': self.ne2, 'nea': self.nea,
            'negc': self.negc, 'nelism': self.nelism, 'necN': self.necN, 'nevN': self.nevN,
            'F1': self.F1, 'F2': self.F2, 'Fa': self.Fa,
            'Fgc': self.Fgc, 'Flism': self.Flism, 'FcN': self.FcN, 'FvN': self.FvN,
            'whicharm': self.whicharm, 'wlism': self.wlism,
            'wLDR': self.wLDR, 'wLHB': self.wLHB, 'wLSB': self.wLSB, 'wLOOPI': self.wLOOPI,
            'hitclump': self.hitclump, 'hitvoid': self.hitvoid, 'wvoid': self.wvoid
        }


def density_2001(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    data: NE2001Data
) -> DensityResult:
    """
    Calculate electron density at galactocentric position (x, y, z).

    This is the main function that combines all 7 components of the NE2001
    electron density model:
        1. ne_outer: Thick disk
        2. ne_inner: Thin disk (annular)
        3. ne_arms: Logarithmic spiral arms
        4. ne_gc: Galactic center
        5. ne_LISM: Local ISM (LHB, LSB, LDR, Loop I)
        6. neclumpN: Discrete clumps
        7. nevoidN: Voids

    Args:
        x: Galactocentric x coordinate (kpc), shape (...)
        y: Galactocentric y coordinate (kpc), shape (...)
        z: Galactocentric z coordinate (kpc), shape (...)
        data: NE2001Data object containing all model parameters

    Returns:
        DensityResult object containing all densities, fluctuation parameters,
        and flags. All tensors have shape (...) matching input coordinates.

    Note:
        - All inputs must have the same shape or be broadcastable
        - Supports batching over arbitrary dimensions
        - Fully differentiable with respect to all inputs
        - Component flags (wg1, wg2, etc.) control which components are active

    Reference:
        density.NE2001.f, SUBROUTINE DENSITY_2001 (lines 12-121)
    """
    device = x.device
    shape = x.shape

    # Initialize all outputs to zero (FORTRAN lines 88-105)
    ne1 = torch.zeros(shape, device=device)
    ne2 = torch.zeros(shape, device=device)
    nea = torch.zeros(shape, device=device)
    negc = torch.zeros(shape, device=device)
    nelism = torch.zeros(shape, device=device)
    necN = torch.zeros(shape, device=device)
    nevN = torch.zeros(shape, device=device)

    F1 = torch.zeros(shape, device=device)
    F2 = torch.zeros(shape, device=device)
    Fa = torch.zeros(shape, device=device)
    Fgc = torch.zeros(shape, device=device)
    Flism = torch.zeros(shape, device=device)
    FcN = torch.zeros(shape, device=device)
    FvN = torch.zeros(shape, device=device)

    whicharm = torch.zeros(shape, dtype=torch.long, device=device)
    wlism = torch.zeros(shape, dtype=torch.long, device=device)
    wLDR = torch.zeros(shape, dtype=torch.long, device=device)
    wLHB = torch.zeros(shape, dtype=torch.long, device=device)
    wLSB = torch.zeros(shape, dtype=torch.long, device=device)
    wLOOPI = torch.zeros(shape, dtype=torch.long, device=device)
    hitclump = torch.zeros(shape, dtype=torch.long, device=device)
    hitvoid = torch.zeros(shape, dtype=torch.long, device=device)
    wvoid = torch.zeros(shape, dtype=torch.long, device=device)

    # Call each component conditionally based on weights (FORTRAN lines 107-114)

    # Component 1: Outer thick disk
    if data.wg1.item() == 1:
        ne1_temp, F1_temp = ne_outer(x, y, z, data.n1h1, data.h1, data.A1, data.F1)
        ne1 = ne1_temp
        F1 = F1_temp

    # Component 2: Inner thin disk
    if data.wg2.item() == 1:
        ne2_temp, F2_temp = ne_inner(x, y, z, data.n2, data.h2, data.A2, data.F2)
        ne2 = ne2_temp
        F2 = F2_temp

    # Component 3: Spiral arms
    if data.wga.item() == 1:
        # Generate spiral arm paths if not already cached
        if not hasattr(data, '_arm_paths_generated'):
            from .components import generate_spiral_arm_paths
            data.arm_x, data.arm_y, data.arm_kmax = generate_spiral_arm_paths(
                data.arm_a, data.arm_rmin, data.arm_thmin, data.arm_extent,
                device=str(device)
            )
            data._arm_paths_generated = True

        nea_temp, Fa_temp, whicharm_temp, _ = ne_arms(
            x, y, z,
            data.arm_x, data.arm_y, data.arm_kmax,
            data.na, data.ha, data.wa, data.Aa, data.Fa,
            data.narm, data.warm, data.harm, data.farm
        )
        nea = nea_temp
        Fa = Fa_temp
        whicharm = whicharm_temp

    # Component 4: Galactic center
    if data.wggc.item() == 1:
        negc_temp, Fgc_temp = ne_gc(x, y, z, data.xgc, data.ygc, data.zgc,
                                     data.rgc, data.hgc, data.negc0, data.Fgc0)
        negc = negc_temp
        Fgc = Fgc_temp

    # Component 5: Local ISM (combined: LHB, LSB, LDR, Loop I)
    if data.wglism.item() == 1:
        nelism, Flism, wlism, wLDR, wLHB, wLSB, wLOOPI = neLISM(x, y, z, data)

    # Component 6: Discrete clumps
    if data.wgcN.item() == 1:
        necN_temp, FcN_temp, hitclump_temp = neclumpN(
            x, y, z,
            data.clump_l, data.clump_b, data.clump_dc,
            data.clump_nec, data.clump_Fc, data.clump_rc, data.clump_edge
        )
        necN = necN_temp
        FcN = FcN_temp
        hitclump = hitclump_temp

    # Component 7: Voids
    if data.wgvN.item() == 1:
        nevN_temp, FvN_temp, hitvoid_temp, wvoid_temp = nevoidN(
            x, y, z,
            data.void_l, data.void_b, data.void_dv,
            data.void_nev, data.void_Fv,
            data.void_aav, data.void_bbv, data.void_ccv,
            data.void_thvy, data.void_thvz, data.void_edge
        )
        nevN = nevN_temp
        FvN = FvN_temp
        hitvoid = hitvoid_temp
        wvoid = wvoid_temp

    return DensityResult(
        ne1=ne1, ne2=ne2, nea=nea, negc=negc, nelism=nelism, necN=necN, nevN=nevN,
        F1=F1, F2=F2, Fa=Fa, Fgc=Fgc, Flism=Flism, FcN=FcN, FvN=FvN,
        whicharm=whicharm, wlism=wlism, wLDR=wLDR, wLHB=wLHB, wLSB=wLSB, wLOOPI=wLOOPI,
        hitclump=hitclump, hitvoid=hitvoid, wvoid=wvoid
    )
