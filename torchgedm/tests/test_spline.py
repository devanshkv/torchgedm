"""
Tests for cubic spline interpolation.

Tests the PyTorch implementation against known values and properties.
"""

import torch
import pytest
import numpy as np
from torchgedm.ne2001.spline import (
    compute_spline_coefficients,
    evaluate_spline,
    cspline,
    CubicSpline,
)


class TestSplineCoefficients:
    """Test computation of spline second derivatives."""

    def test_linear_function(self):
        """For linear data, second derivatives should be near zero."""
        x = torch.linspace(0, 10, 11)
        y = 2 * x + 3
        y2 = compute_spline_coefficients(x, y)

        # Second derivatives should be very small for linear data
        assert torch.allclose(y2, torch.zeros_like(y2), atol=1e-10)

    def test_quadratic_function(self):
        """For quadratic data, second derivatives should be approximately constant."""
        x = torch.linspace(-5, 5, 11)
        y = x**2
        y2 = compute_spline_coefficients(x, y)

        # For y = x^2, y'' = 2 everywhere
        # Natural spline has y2[0] = y2[-1] = 0, but interior should be close to 2
        # With natural boundary conditions, we can't expect exact match
        assert torch.allclose(y2[1:-1], torch.full_like(y2[1:-1], 2.0), atol=1.0)
        # Check that the mean is close to 2.0
        assert torch.allclose(y2[1:-1].mean(), torch.tensor(2.0), atol=0.3)

    def test_natural_boundary_conditions(self):
        """Natural spline has zero second derivative at endpoints."""
        x = torch.linspace(0, 1, 5)
        y = torch.sin(x)
        y2 = compute_spline_coefficients(x, y)

        assert torch.allclose(y2[0], torch.tensor(0.0), atol=1e-10)
        assert torch.allclose(y2[-1], torch.tensor(0.0), atol=1e-10)

    def test_batched_computation(self):
        """Test batched spline coefficient computation."""
        x = torch.linspace(0, 1, 5).unsqueeze(0).expand(3, 5)
        y = torch.stack([
            torch.sin(x[0]),
            torch.cos(x[0]),
            x[0]**2,
        ])
        y2 = compute_spline_coefficients(x, y)

        assert y2.shape == (3, 5)
        # Each batch should have natural boundary conditions
        assert torch.allclose(y2[:, 0], torch.zeros(3), atol=1e-10)
        assert torch.allclose(y2[:, -1], torch.zeros(3), atol=1e-10)

    def test_differentiable(self):
        """Spline coefficients should be differentiable w.r.t. y."""
        x = torch.linspace(0, 1, 5)
        y = torch.sin(x)
        y.requires_grad = True

        y2 = compute_spline_coefficients(x, y)
        loss = y2.sum()
        loss.backward()

        assert y.grad is not None
        assert not torch.isnan(y.grad).any()


class TestSplineEvaluation:
    """Test spline evaluation at query points."""

    def test_interpolates_knots(self):
        """Spline should exactly interpolate knot points."""
        x = torch.tensor([0.0, 1.0, 2.0, 3.0])
        y = torch.tensor([1.0, 2.0, 1.5, 3.0])
        y2 = compute_spline_coefficients(x, y)

        # Evaluate at knot points
        yout = evaluate_spline(x, y, y2, x)

        assert torch.allclose(yout, y, atol=1e-6)

    def test_linear_interpolation_between_knots(self):
        """For linear data, spline reduces to linear interpolation."""
        x = torch.tensor([0.0, 1.0, 2.0])
        y = torch.tensor([0.0, 1.0, 2.0])
        y2 = compute_spline_coefficients(x, y)

        xout = torch.tensor([0.5, 1.5])
        yout = evaluate_spline(x, y, y2, xout)

        expected = torch.tensor([0.5, 1.5])
        assert torch.allclose(yout, expected, atol=1e-6)

    def test_smooth_curve(self):
        """Spline should produce smooth interpolation."""
        x = torch.tensor([0.0, 1.0, 2.0, 3.0])
        y = torch.tensor([0.0, 1.0, 0.0, 1.0])
        y2 = compute_spline_coefficients(x, y)

        xout = torch.linspace(0, 3, 31)
        yout = evaluate_spline(x, y, y2, xout)

        # Check smoothness: no sudden jumps
        dy = yout[1:] - yout[:-1]
        assert (dy.abs() < 0.5).all()  # No jumps larger than 0.5

    def test_extrapolation_clamping(self):
        """Points outside knot range use boundary intervals."""
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([1.0, 4.0, 9.0])
        y2 = compute_spline_coefficients(x, y)

        # Test extrapolation
        xout = torch.tensor([0.5, 3.5])
        yout = evaluate_spline(x, y, y2, xout)

        # Should use first/last interval
        assert yout.shape == xout.shape
        assert not torch.isnan(yout).any()
        assert not torch.isinf(yout).any()

    def test_batched_evaluation(self):
        """Test batched spline evaluation."""
        x = torch.linspace(0, 1, 5).unsqueeze(0).expand(2, 5)
        y = torch.stack([
            torch.sin(x[0]),
            torch.cos(x[0]),
        ])
        y2 = compute_spline_coefficients(x, y)

        xout = torch.linspace(0, 1, 11).unsqueeze(0).expand(2, 11)
        yout = evaluate_spline(x, y, y2, xout)

        assert yout.shape == (2, 11)

    def test_scalar_xout(self):
        """Test evaluation at single scalar point."""
        x = torch.tensor([0.0, 1.0, 2.0])
        y = torch.tensor([0.0, 1.0, 0.0])
        y2 = compute_spline_coefficients(x, y)

        xout = torch.tensor(0.5)
        yout = evaluate_spline(x, y, y2, xout)

        assert yout.dim() == 0  # Scalar output
        assert not torch.isnan(yout)

    def test_differentiable(self):
        """Spline evaluation should be differentiable."""
        x = torch.tensor([0.0, 1.0, 2.0])
        y = torch.tensor([0.0, 1.0, 0.0])
        y.requires_grad = True
        y2 = compute_spline_coefficients(x, y)

        xout = torch.tensor([0.5, 1.5])
        yout = evaluate_spline(x, y, y2, xout)

        loss = yout.sum()
        loss.backward()

        assert y.grad is not None
        assert not torch.isnan(y.grad).any()


class TestCsplineFunction:
    """Test functional interface."""

    def test_basic_interpolation(self):
        """Test basic interpolation with functional interface."""
        x = torch.tensor([0.0, 1.0, 2.0, 3.0])
        y = torch.tensor([1.0, 2.0, 1.5, 3.0])
        xout = torch.tensor([0.5, 1.5, 2.5])

        yout, y2 = cspline(x, y, xout)

        assert yout.shape == xout.shape
        assert y2.shape == x.shape
        assert not torch.isnan(yout).any()

    def test_coefficient_reuse(self):
        """Test reusing precomputed coefficients."""
        x = torch.tensor([0.0, 1.0, 2.0, 3.0])
        y = torch.tensor([1.0, 2.0, 1.5, 3.0])

        # First call computes coefficients
        yout1, y2 = cspline(x, y, torch.tensor([0.5]))

        # Second call reuses coefficients
        yout2, y2_reused = cspline(x, y, torch.tensor([1.5]), y2=y2)

        assert torch.allclose(y2, y2_reused)
        assert yout1.shape == (1,)
        assert yout2.shape == (1,)


class TestCubicSplineClass:
    """Test stateful CubicSpline class."""

    def test_stateful_caching(self):
        """Test that coefficients are cached between calls."""
        spline = CubicSpline()
        x = torch.tensor([0.0, 1.0, 2.0, 3.0])
        y = torch.tensor([1.0, 2.0, 1.5, 3.0])

        # First call with compute_coeffs=True
        yout1 = spline(x, y, torch.tensor([0.5]), compute_coeffs=True)
        y2_cached = spline.y2.clone()

        # Second call with compute_coeffs=False should reuse
        yout2 = spline(x, y, torch.tensor([1.5]), compute_coeffs=False)

        assert torch.allclose(spline.y2, y2_cached)
        assert yout1.shape == (1,)
        assert yout2.shape == (1,)

    def test_reset(self):
        """Test clearing cached coefficients."""
        spline = CubicSpline()
        x = torch.tensor([0.0, 1.0, 2.0])
        y = torch.tensor([0.0, 1.0, 0.0])

        spline(x, y, torch.tensor([0.5]), compute_coeffs=True)
        assert spline.y2 is not None

        spline.reset()
        assert spline.y2 is None

    def test_auto_compute_when_none(self):
        """Test that coefficients are computed if not cached."""
        spline = CubicSpline()
        x = torch.tensor([0.0, 1.0, 2.0])
        y = torch.tensor([0.0, 1.0, 0.0])

        # First call without explicit compute_coeffs should still work
        yout = spline(x, y, torch.tensor([0.5]))

        assert yout.shape == (1,)
        assert spline.y2 is not None


class TestSpiralArmUseCase:
    """Test spline on spiral arm-like data."""

    def test_spiral_arm_interpolation(self):
        """Test interpolation of spiral arm path (theta vs r)."""
        # Simulate spiral arm control points
        theta = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0]) * torch.pi
        r = torch.tensor([8.0, 9.0, 10.5, 12.0, 13.0])

        # Interpolate at finer resolution
        theta_fine = torch.linspace(0, 2*torch.pi, 101)
        r_fine, y2 = cspline(theta, r, theta_fine)

        assert r_fine.shape == theta_fine.shape
        assert not torch.isnan(r_fine).any()

        # Check monotonicity (spiral arms generally increase in radius)
        assert (r_fine[1:] >= r_fine[:-1] - 0.5).all()  # Allow small decreases

    def test_multiple_arms(self):
        """Test batched interpolation for multiple spiral arms."""
        n_arms = 5
        n_knots = 10

        # Create multiple spiral arms
        theta = torch.linspace(0, 2*torch.pi, n_knots).unsqueeze(0).expand(n_arms, n_knots)
        r = torch.linspace(5, 15, n_knots).unsqueeze(0).expand(n_arms, n_knots)

        # Add variation per arm
        r = r + torch.randn(n_arms, 1) * 0.5

        # Interpolate all arms at once
        theta_fine = torch.linspace(0, 2*torch.pi, 101).unsqueeze(0).expand(n_arms, 101)
        r_fine, y2 = cspline(theta, r, theta_fine)

        assert r_fine.shape == (n_arms, 101)
        assert not torch.isnan(r_fine).any()


class TestNumericalProperties:
    """Test numerical properties and edge cases."""

    def test_monotonic_input(self):
        """Test that x must be monotonically increasing."""
        x = torch.tensor([0.0, 2.0, 1.0, 3.0])  # Not sorted
        y = torch.tensor([1.0, 2.0, 1.5, 3.0])

        # Should still run but results may be incorrect
        # (Real implementation should validate input)
        y2 = compute_spline_coefficients(x, y)
        assert y2.shape == x.shape

    def test_minimum_knots(self):
        """Test with minimum number of knots."""
        x = torch.tensor([0.0, 1.0])
        y = torch.tensor([0.0, 1.0])
        y2 = compute_spline_coefficients(x, y)

        # Should handle 2 knots (degenerates to linear)
        xout = torch.tensor([0.5])
        yout = evaluate_spline(x, y, y2, xout)

        assert torch.allclose(yout, torch.tensor([0.5]), atol=1e-6)

    def test_many_knots(self):
        """Test with many knots."""
        n = 100
        x = torch.linspace(0, 10, n)
        y = torch.sin(x)
        y2 = compute_spline_coefficients(x, y)

        xout = torch.linspace(0, 10, 1000)
        yout = evaluate_spline(x, y, y2, xout)

        assert yout.shape == xout.shape
        assert not torch.isnan(yout).any()

    def test_precision(self):
        """Test double precision."""
        x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        y = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        y2 = compute_spline_coefficients(x, y)

        assert y2.dtype == torch.float64

        xout = torch.tensor([0.5], dtype=torch.float64)
        yout = evaluate_spline(x, y, y2, xout)

        assert yout.dtype == torch.float64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
