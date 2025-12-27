"""
Cubic spline interpolation in PyTorch.

Implements the classic cubic spline algorithm from Numerical Recipes,
matching the FORTRAN cspline function in NE2001.

Author: torchgedm
"""

import torch
from typing import Optional, Tuple


def compute_spline_coefficients(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Compute second derivatives for natural cubic spline.

    Uses the tridiagonal algorithm with natural boundary conditions
    (second derivative = 0 at endpoints).

    Args:
        x: Knot x-coordinates, shape (..., n). Must be sorted in ascending order.
        y: Knot y-coordinates, shape (..., n).

    Returns:
        y2: Second derivatives at knots, shape (..., n).

    Note:
        This implements the algorithm from lines 690-704 of density.NE2001.f.
        Natural boundary conditions: y2[0] = y2[n-1] = 0.
    """
    n = x.shape[-1]

    # Initialize arrays - use clone to avoid inplace ops
    y2 = torch.zeros_like(y).clone()
    u = torch.zeros_like(y).clone()

    # Forward pass: compute u and y2 (lines 693-698)
    # Build lists to avoid inplace modification during gradient computation
    y2_list = [y2[..., 0]]  # Natural BC: y2[0] = 0
    u_list = [u[..., 0]]    # Natural BC: u[0] = 0

    for i in range(1, n - 1):
        # sig = (x[i] - x[i-1]) / (x[i+1] - x[i-1])
        sig = (x[..., i] - x[..., i-1]) / (x[..., i+1] - x[..., i-1])
        p = sig * y2_list[i-1] + 2.0
        y2_i = (sig - 1.0) / p

        # Compute u[i]
        dy_right = (y[..., i+1] - y[..., i]) / (x[..., i+1] - x[..., i])
        dy_left = (y[..., i] - y[..., i-1]) / (x[..., i] - x[..., i-1])
        u_i = (6.0 * (dy_right - dy_left) / (x[..., i+1] - x[..., i-1]) - sig * u_list[i-1]) / p

        y2_list.append(y2_i)
        u_list.append(u_i)

    # Natural boundary at end
    qn = 0.0
    un = 0.0
    y2_n = (un - qn * u_list[-1]) / (qn * y2_list[-1] + 1.0)
    y2_list.append(y2_n)

    # Backward pass: finalize y2 (lines 703-704)
    # Work backwards to compute final y2 values
    y2_final = [y2_list[-1]]  # Start with last element
    for k in range(n - 2, -1, -1):
        y2_k = y2_list[k] * y2_final[0] + u_list[k]
        y2_final.insert(0, y2_k)

    # Stack into tensor
    y2 = torch.stack(y2_final, dim=-1)

    return y2


def evaluate_spline(
    x: torch.Tensor,
    y: torch.Tensor,
    y2: torch.Tensor,
    xout: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate cubic spline at given points.

    Args:
        x: Knot x-coordinates, shape (..., n). Must be sorted.
        y: Knot y-coordinates, shape (..., n).
        y2: Second derivatives at knots, shape (..., n).
        xout: Points to evaluate at, shape (..., m) or (...).

    Returns:
        yout: Interpolated values, shape (..., m) or (...).

    Note:
        This implements the evaluation from lines 707-723 of density.NE2001.f.
        Uses binary search to find bracketing interval, then cubic interpolation.
    """
    # Handle scalar xout
    scalar_input = xout.dim() == x.dim() - 1
    if scalar_input:
        xout = xout.unsqueeze(-1)

    n = x.shape[-1]
    m = xout.shape[-1]

    # Find bracketing intervals using searchsorted
    # This is the vectorized equivalent of the binary search (lines 707-717)
    # searchsorted returns the index where xout would be inserted
    khi = torch.searchsorted(x, xout, right=False)

    # Clamp to valid range [1, n-1]
    khi = torch.clamp(khi, 1, n - 1)
    klo = khi - 1

    # Gather the bracketing values
    # Need to handle broadcasting for batched operations
    x_klo = torch.gather(x.unsqueeze(-1).expand(*x.shape, m), -2, klo.unsqueeze(-2)).squeeze(-2)
    x_khi = torch.gather(x.unsqueeze(-1).expand(*x.shape, m), -2, khi.unsqueeze(-2)).squeeze(-2)
    y_klo = torch.gather(y.unsqueeze(-1).expand(*y.shape, m), -2, klo.unsqueeze(-2)).squeeze(-2)
    y_khi = torch.gather(y.unsqueeze(-1).expand(*y.shape, m), -2, khi.unsqueeze(-2)).squeeze(-2)
    y2_klo = torch.gather(y2.unsqueeze(-1).expand(*y2.shape, m), -2, klo.unsqueeze(-2)).squeeze(-2)
    y2_khi = torch.gather(y2.unsqueeze(-1).expand(*y2.shape, m), -2, khi.unsqueeze(-2)).squeeze(-2)

    # Compute interpolation (lines 718-723)
    h = x_khi - x_klo
    a = (x_khi - xout) / h
    b = (xout - x_klo) / h

    # Cubic spline formula
    yout = (a * y_klo + b * y_khi +
            ((a**3 - a) * y2_klo + (b**3 - b) * y2_khi) * (h**2) / 6.0)

    if scalar_input:
        yout = yout.squeeze(-1)

    return yout


class CubicSpline:
    """
    Cubic spline interpolation with caching.

    Mimics the stateful behavior of the FORTRAN cspline subroutine,
    which uses 'save' to cache second derivatives between calls.

    Example:
        >>> spline = CubicSpline()
        >>> # First call with compute_coeffs=True
        >>> y1 = spline(x_knots, y_knots, x_eval, compute_coeffs=True)
        >>> # Subsequent calls reuse cached coefficients
        >>> y2 = spline(x_knots, y_knots, x_eval2, compute_coeffs=False)
    """

    def __init__(self):
        self.y2: Optional[torch.Tensor] = None

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        xout: torch.Tensor,
        compute_coeffs: bool = True,
    ) -> torch.Tensor:
        """
        Evaluate cubic spline.

        Args:
            x: Knot x-coordinates, shape (..., n). Must be sorted.
            y: Knot y-coordinates, shape (..., n).
            xout: Points to evaluate at, shape (..., m) or (...).
            compute_coeffs: If True, recompute second derivatives.
                           If False, use cached coefficients.

        Returns:
            yout: Interpolated values, shape (..., m) or (...).
        """
        if compute_coeffs or self.y2 is None:
            self.y2 = compute_spline_coefficients(x, y)

        return evaluate_spline(x, y, self.y2, xout)

    def reset(self):
        """Clear cached coefficients."""
        self.y2 = None


def cspline(
    x: torch.Tensor,
    y: torch.Tensor,
    xout: torch.Tensor,
    y2: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Cubic spline interpolation (functional interface).

    This is a functional interface that matches the FORTRAN cspline behavior:
    - If y2 is None, computes second derivatives
    - If y2 is provided, uses them directly
    - Returns both interpolated values and coefficients

    Args:
        x: Knot x-coordinates, shape (..., n). Must be sorted.
        y: Knot y-coordinates, shape (..., n).
        xout: Points to evaluate at, shape (..., m) or (...).
        y2: Optional pre-computed second derivatives, shape (..., n).

    Returns:
        yout: Interpolated values, shape (..., m) or (...).
        y2: Second derivatives, shape (..., n).

    Example:
        >>> # First call computes coefficients
        >>> yout1, y2 = cspline(x_knots, y_knots, x_eval1)
        >>> # Reuse coefficients for second call
        >>> yout2, _ = cspline(x_knots, y_knots, x_eval2, y2=y2)
    """
    if y2 is None:
        y2 = compute_spline_coefficients(x, y)

    yout = evaluate_spline(x, y, y2, xout)

    return yout, y2
