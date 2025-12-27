"""
Outer thick disk component (component 1) for NE2001 model

Implements the large-scale thick disk electron density distribution.
"""

import torch
import math
from ..utils import sech2


# Sun's galactocentric radius (kpc)
RSUN = 8.5


def ne_outer(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    n1h1: torch.Tensor,
    h1: torch.Tensor,
    A1: torch.Tensor,
    F1: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute outer thick disk electron density component.

    This implements the thick disk component of the NE2001 model, which
    represents the large-scale Galactic disk structure.

    Args:
        x: Galactocentric x coordinate (kpc)
        y: Galactocentric y coordinate (kpc)
        z: Galactocentric z coordinate (kpc)
        n1h1: Electron density scale parameter (cm^-3)
        h1: Scale height (kpc)
        A1: Radial scale length (kpc)
        F1: Fluctuation parameter

    Returns:
        Tuple of (ne1, F_outer) where:
        - ne1: Electron density (cm^-3)
        - F_outer: Fluctuation parameter (same as F1)

    Note:
        The density model is:
        ne1 = (n1h1/h1) * g1 * sech²(z/h1)

        where g1 is a radial function:
        - g1 = cos(π/2 * rr/A1) / cos(π/2 * rsun/A1)  if rr ≤ A1
        - g1 = 0  if rr > A1

        and rr = sqrt(x² + y²) is the cylindrical radius.
    """
    # Calculate cylindrical radius
    rr = torch.sqrt(x**2 + y**2)

    # Calculate normalization factor (constant for all points)
    # suncos = cos(π/2 * rsun / A1)
    pihalf = math.pi / 2
    suncos = torch.cos(torch.tensor(pihalf * RSUN, dtype=x.dtype, device=x.device) / A1)

    # Calculate radial function g1
    # If rr > A1: g1 = 0
    # Otherwise: g1 = cos(π/2 * rr/A1) / suncos
    g1 = torch.where(
        rr > A1,
        torch.zeros_like(rr),
        torch.cos(pihalf * rr / A1) / suncos
    )

    # Calculate electron density
    # ne1 = (n1h1/h1) * g1 * sech²(z/h1)
    ne1 = (n1h1 / h1) * g1 * sech2(z / h1)

    # F_outer is just the parameter F1
    F_outer = F1.expand_as(ne1) if F1.dim() == 0 else F1

    return ne1, F_outer
