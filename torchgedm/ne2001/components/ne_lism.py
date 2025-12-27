"""
Combined Local ISM (LISM) component

Combines all 4 local ISM components (LHB, LSB, LDRQ1, Loop I) with proper
weighting hierarchy:
    LHB > Loop I > LSB > LDR

References:
    NE2001 ne_LISM function (neLISM.NE2001.f, lines 21-121)
"""

import torch
from .ne_lhb import neLHB
from .ne_lsb import neLSB
from .ne_ldr import neLDRQ1
from .ne_loopi import neLOOPI


def neLISM(x, y, z, data):
    """
    Calculate electron density in Local ISM (combined from 4 components).

    Weighting hierarchy (highest to lowest priority):
        1. LHB (Local Hot Bubble) - overrides everything
        2. Loop I - overrides LSB and LDR
        3. LSB (Local Super Bubble) - overrides LDR
        4. LDR (Low Density Region Q1) - lowest priority

    The combination formula ensures that higher priority components
    completely override lower priority ones when their weights are active.

    Formula (from FORTRAN neLISM.NE2001.f lines 91-96):
        ne_LISM = (1-wLHB) *
                  (
                    (1-wLOOPI) * (wLSB*neLSB + (1-wLSB)*neLDR)
                +     wLOOPI * neLOOPI
                  )
                +     wLHB  * neLHB

    This means:
        - If wLHB=1: use only LHB density
        - If wLOOPI=1 (and wLHB=0): use only Loop I density
        - If wLSB=1 (and wLHB=0, wLOOPI=0): use only LSB density
        - Otherwise: use LDR density

    Args:
        x, y, z: Galactocentric coordinates (kpc). Can be tensors of any shape.
        data: NE2001Data object containing LISM parameters

    Returns:
        ne: Combined electron density (cm^-3)
        F: Combined fluctuation parameter
        wLISM: Maximum weight across all components (1 if inside any, 0 if outside all)
        wLDR: Weight for LDR component (0 or 1)
        wLHB: Weight for LHB component (0 or 1)
        wLSB: Weight for LSB component (0 or 1)
        wLOOPI: Weight for Loop I component (0 or 1)

    References:
        NE2001 ne_LISM function (neLISM.NE2001.f, lines 21-121)
        Weighting scheme described in comments (lines 7-13)
    """
    # Call all 4 component functions
    ne_ldr, F_ldr, w_ldr = neLDRQ1(x, y, z, data)
    ne_lsb, F_lsb, w_lsb = neLSB(x, y, z, data)
    ne_lhb, F_lhb, w_lhb = neLHB(x, y, z, data)
    ne_loopi, F_loopi, w_loopi = neLOOPI(x, y, z, data)

    # Convert weights to float for arithmetic operations
    w_ldr_f = w_ldr.float()
    w_lsb_f = w_lsb.float()
    w_lhb_f = w_lhb.float()
    w_loopi_f = w_loopi.float()

    # Apply hierarchical weighting formula
    # Lines 91-96 of neLISM.NE2001.f
    ne_lism = (1.0 - w_lhb_f) * (
        (1.0 - w_loopi_f) * (w_lsb_f * ne_lsb + (1.0 - w_lsb_f) * ne_ldr)
        + w_loopi_f * ne_loopi
    ) + w_lhb_f * ne_lhb

    # Apply same weighting to fluctuation parameters
    # Lines 98-103 of neLISM.NE2001.f
    F_lism = (1.0 - w_lhb_f) * (
        (1.0 - w_loopi_f) * (w_lsb_f * F_lsb + (1.0 - w_lsb_f) * F_ldr)
        + w_loopi_f * F_loopi
    ) + w_lhb_f * F_lhb

    # Return maximum weight of any component
    # Line 108 of neLISM.NE2001.f: wLISM = max(wLOOPI, max(wLDR, max(wLSB, wLHB)))
    w_lism = torch.max(
        torch.max(
            torch.max(w_ldr, w_lsb),
            w_lhb
        ),
        w_loopi
    )

    # Return all weights as in FORTRAN signature:
    # ne_LISM(x,y,z,FLISM,wLISM,wldr,wlhb,wlsb,wloopI)
    return ne_lism, F_lism, w_lism, w_ldr, w_lhb, w_lsb, w_loopi
