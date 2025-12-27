"""
Tests for coordinate transformations between galactic and galactocentric systems.

These tests validate the bidirectional coordinate transformations:
- galactic_to_galactocentric: (l, b, d) -> (x, y, z)
- galactocentric_to_galactic: (x, y, z) -> (l, b, d)
"""

import pytest
import torch
import numpy as np
from torchgedm.ne2001.utils import (
    galactic_to_galactocentric,
    galactocentric_to_galactic,
)


class TestGalacticToGalactocentric:
    """Tests for galactic -> galactocentric transformation"""

    def test_sun_position(self):
        """Sun at origin in galactic coords should be at (0, 8.5, 0) in galactocentric"""
        l = torch.tensor(0.0)
        b = torch.tensor(0.0)
        d = torch.tensor(0.0)

        x, y, z = galactic_to_galactocentric(l, b, d)

        assert torch.isclose(x, torch.tensor(0.0), atol=1e-6)
        assert torch.isclose(y, torch.tensor(8.5), atol=1e-6)
        assert torch.isclose(z, torch.tensor(0.0), atol=1e-6)

    def test_galactic_center_direction(self):
        """Point at l=0, b=0, d=8.5 should be at Galactic center (0, 0, 0)"""
        l = torch.tensor(0.0)
        b = torch.tensor(0.0)
        d = torch.tensor(8.5)

        x, y, z = galactic_to_galactocentric(l, b, d)

        assert torch.isclose(x, torch.tensor(0.0), atol=1e-6)
        assert torch.isclose(y, torch.tensor(0.0), atol=1e-6)
        assert torch.isclose(z, torch.tensor(0.0), atol=1e-6)

    def test_north_galactic_pole(self):
        """Point at b=90 should be above Sun along z-axis"""
        l = torch.tensor(0.0)  # Longitude doesn't matter at pole
        b = torch.tensor(90.0)
        d = torch.tensor(1.0)

        x, y, z = galactic_to_galactocentric(l, b, d)

        assert torch.isclose(x, torch.tensor(0.0), atol=1e-6)
        assert torch.isclose(y, torch.tensor(8.5), atol=1e-6)
        assert torch.isclose(z, torch.tensor(1.0), atol=1e-6)

    def test_south_galactic_pole(self):
        """Point at b=-90 should be below Sun along z-axis"""
        l = torch.tensor(0.0)
        b = torch.tensor(-90.0)
        d = torch.tensor(1.0)

        x, y, z = galactic_to_galactocentric(l, b, d)

        assert torch.isclose(x, torch.tensor(0.0), atol=1e-6)
        assert torch.isclose(y, torch.tensor(8.5), atol=1e-6)
        assert torch.isclose(z, torch.tensor(-1.0), atol=1e-6)

    def test_positive_longitude(self):
        """Point at l=90, b=0 should be in +x direction"""
        l = torch.tensor(90.0)
        b = torch.tensor(0.0)
        d = torch.tensor(1.0)

        x, y, z = galactic_to_galactocentric(l, b, d)

        assert torch.isclose(x, torch.tensor(1.0), atol=1e-6)
        assert torch.isclose(y, torch.tensor(8.5), atol=1e-6)
        assert torch.isclose(z, torch.tensor(0.0), atol=1e-6)

    def test_anticenter_direction(self):
        """Point at l=180, b=0 should be away from Galactic center"""
        l = torch.tensor(180.0)
        b = torch.tensor(0.0)
        d = torch.tensor(1.0)

        x, y, z = galactic_to_galactocentric(l, b, d)

        assert torch.isclose(x, torch.tensor(0.0), atol=1e-6)
        assert torch.isclose(y, torch.tensor(9.5), atol=1e-6)
        assert torch.isclose(z, torch.tensor(0.0), atol=1e-6)

    def test_batching(self):
        """Test that transformation works with batched inputs"""
        l = torch.tensor([0.0, 90.0, 180.0, 270.0])
        b = torch.tensor([0.0, 0.0, 0.0, 0.0])
        d = torch.tensor([1.0, 1.0, 1.0, 1.0])

        x, y, z = galactic_to_galactocentric(l, b, d)

        assert x.shape == (4,)
        assert y.shape == (4,)
        assert z.shape == (4,)

        # Check individual points
        assert torch.isclose(x[0], torch.tensor(0.0), atol=1e-6)
        assert torch.isclose(x[1], torch.tensor(1.0), atol=1e-6)
        assert torch.isclose(y[2], torch.tensor(9.5), atol=1e-6)

    def test_multidimensional_batching(self):
        """Test with 2D batched inputs"""
        l = torch.tensor([[0.0, 90.0], [180.0, 270.0]])
        b = torch.zeros(2, 2)
        d = torch.ones(2, 2)

        x, y, z = galactic_to_galactocentric(l, b, d)

        assert x.shape == (2, 2)
        assert y.shape == (2, 2)
        assert z.shape == (2, 2)

    def test_differentiability(self):
        """Test that transformation is differentiable"""
        l = torch.tensor(45.0, requires_grad=True)
        b = torch.tensor(30.0, requires_grad=True)
        d = torch.tensor(5.0, requires_grad=True)

        x, y, z = galactic_to_galactocentric(l, b, d)

        # Compute a scalar output for backprop
        loss = x**2 + y**2 + z**2

        loss.backward()

        # Check that gradients exist and are non-zero
        assert l.grad is not None
        assert b.grad is not None
        assert d.grad is not None
        assert not torch.isclose(l.grad, torch.tensor(0.0))


class TestGalactocentricToGalactic:
    """Tests for galactocentric -> galactic transformation"""

    def test_sun_position(self):
        """Point at (0, 8.5, 0) should be at Sun (l, b, d) = (*, *, 0)"""
        x = torch.tensor(0.0)
        y = torch.tensor(8.5)
        z = torch.tensor(0.0)

        l, b, d = galactocentric_to_galactic(x, y, z)

        # Distance should be zero (Sun's position)
        assert torch.isclose(d, torch.tensor(0.0), atol=1e-6)
        # At origin, l and b are undefined but should not be NaN
        assert not torch.isnan(l)
        assert not torch.isnan(b)

    def test_galactic_center(self):
        """Galactic center at (0, 0, 0) should be at l=0, b=0, d=8.5"""
        x = torch.tensor(0.0)
        y = torch.tensor(0.0)
        z = torch.tensor(0.0)

        l, b, d = galactocentric_to_galactic(x, y, z)

        assert torch.isclose(l, torch.tensor(0.0), atol=1e-6)
        assert torch.isclose(b, torch.tensor(0.0), atol=1e-6)
        assert torch.isclose(d, torch.tensor(8.5), atol=1e-6)

    def test_point_above_sun(self):
        """Point above Sun should have b=90"""
        x = torch.tensor(0.0)
        y = torch.tensor(8.5)
        z = torch.tensor(1.0)

        l, b, d = galactocentric_to_galactic(x, y, z)

        assert torch.isclose(b, torch.tensor(90.0), atol=1e-6)
        assert torch.isclose(d, torch.tensor(1.0), atol=1e-6)

    def test_point_in_positive_x(self):
        """Point in +x direction should have l=90"""
        x = torch.tensor(1.0)
        y = torch.tensor(8.5)
        z = torch.tensor(0.0)

        l, b, d = galactocentric_to_galactic(x, y, z)

        assert torch.isclose(l, torch.tensor(90.0), atol=1e-6)
        assert torch.isclose(b, torch.tensor(0.0), atol=1e-6)
        assert torch.isclose(d, torch.tensor(1.0), atol=1e-6)

    def test_longitude_range(self):
        """Test that longitude is in [0, 360) range"""
        # Test point in negative x (should give l in [180, 360))
        x = torch.tensor(-1.0)
        y = torch.tensor(8.5)
        z = torch.tensor(0.0)

        l, b, d = galactocentric_to_galactic(x, y, z)

        assert l >= 0.0
        assert l < 360.0
        assert torch.isclose(l, torch.tensor(270.0), atol=1e-6)

    def test_batching(self):
        """Test with batched inputs"""
        x = torch.tensor([0.0, 1.0, 0.0, -1.0])
        y = torch.tensor([8.5, 8.5, 9.5, 8.5])
        z = torch.tensor([0.0, 0.0, 0.0, 0.0])

        l, b, d = galactocentric_to_galactic(x, y, z)

        assert l.shape == (4,)
        assert b.shape == (4,)
        assert d.shape == (4,)

    def test_differentiability(self):
        """Test that transformation is differentiable"""
        x = torch.tensor(1.0, requires_grad=True)
        y = torch.tensor(9.0, requires_grad=True)
        z = torch.tensor(0.5, requires_grad=True)

        l, b, d = galactocentric_to_galactic(x, y, z)

        # Compute a scalar output for backprop
        loss = l**2 + b**2 + d**2

        loss.backward()

        # Check that gradients exist and are non-zero
        assert x.grad is not None
        assert y.grad is not None
        assert z.grad is not None


class TestRoundTrip:
    """Test round-trip transformations: galactic -> galactocentric -> galactic"""

    def test_round_trip_single_point(self):
        """Test round-trip for a single point"""
        l_orig = torch.tensor(45.0)
        b_orig = torch.tensor(30.0)
        d_orig = torch.tensor(5.0)

        # Forward transformation
        x, y, z = galactic_to_galactocentric(l_orig, b_orig, d_orig)

        # Inverse transformation
        l_recovered, b_recovered, d_recovered = galactocentric_to_galactic(x, y, z)

        # Check that we recover original values
        assert torch.isclose(l_recovered, l_orig, atol=1e-6)
        assert torch.isclose(b_recovered, b_orig, atol=1e-6)
        assert torch.isclose(d_recovered, d_orig, atol=1e-6)

    def test_round_trip_batch(self):
        """Test round-trip for batched points"""
        torch.manual_seed(42)
        n = 100

        # Generate random galactic coordinates
        l_orig = torch.rand(n) * 360.0  # [0, 360)
        b_orig = (torch.rand(n) - 0.5) * 180.0  # [-90, 90]
        d_orig = torch.rand(n) * 20.0 + 0.1  # [0.1, 20.1] kpc

        # Forward transformation
        x, y, z = galactic_to_galactocentric(l_orig, b_orig, d_orig)

        # Inverse transformation
        l_recovered, b_recovered, d_recovered = galactocentric_to_galactic(x, y, z)

        # Check that we recover original values
        # Handle longitude wraparound (0 and 360 are the same)
        l_diff = torch.abs(l_recovered - l_orig)
        l_diff = torch.minimum(l_diff, 360.0 - l_diff)

        # Tolerances account for floating-point precision and polar singularities
        assert torch.all(l_diff < 0.001)  # 0.001 degrees ≈ 3.6 arcsec
        assert torch.allclose(b_recovered, b_orig, atol=0.005)  # 0.005 degrees ≈ 18 arcsec
        assert torch.allclose(d_recovered, d_orig, rtol=1e-5, atol=1e-7)

    def test_round_trip_near_poles(self):
        """Test round-trip for points near galactic poles"""
        # Near north pole
        l_orig = torch.tensor([0.0, 90.0, 180.0, 270.0])
        b_orig = torch.tensor([89.9, 89.9, 89.9, 89.9])
        d_orig = torch.tensor([1.0, 2.0, 3.0, 4.0])

        x, y, z = galactic_to_galactocentric(l_orig, b_orig, d_orig)
        l_recovered, b_recovered, d_recovered = galactocentric_to_galactic(x, y, z)

        # At high latitudes, longitude becomes less well-defined but should still round-trip
        # More relaxed tolerances for near-pole cases
        assert torch.allclose(b_recovered, b_orig, atol=0.01)  # 0.01 degrees ≈ 36 arcsec
        assert torch.allclose(d_recovered, d_orig, rtol=1e-4, atol=1e-7)

    def test_round_trip_at_pole(self):
        """Test round-trip at exact galactic pole"""
        l_orig = torch.tensor(0.0)
        b_orig = torch.tensor(90.0)
        d_orig = torch.tensor(1.0)

        x, y, z = galactic_to_galactocentric(l_orig, b_orig, d_orig)
        l_recovered, b_recovered, d_recovered = galactocentric_to_galactic(x, y, z)

        # At exact pole, longitude is undefined but lat and distance should match
        assert torch.isclose(b_recovered, b_orig, atol=1e-6)
        assert torch.isclose(d_recovered, d_orig, rtol=1e-6)


class TestValidationAgainstOriginal:
    """Validation tests comparing against original implementation"""

    def test_comparison_with_ne_clumps_function(self):
        """Compare with the galactic_to_xyz function in ne_clumps.py"""
        from torchgedm.ne2001.components.ne_clumps import galactic_to_xyz

        torch.manual_seed(42)
        n = 100

        # Generate random test points
        l_deg = torch.rand(n) * 360.0
        b_deg = (torch.rand(n) - 0.5) * 180.0
        d_kpc = torch.rand(n) * 20.0 + 0.1

        # Use new implementation
        x_new, y_new, z_new = galactic_to_galactocentric(l_deg, b_deg, d_kpc)

        # Use original implementation
        x_orig, y_orig, z_orig = galactic_to_xyz(l_deg, b_deg, d_kpc)

        # Compare results
        assert torch.allclose(x_new, x_orig, rtol=1e-6, atol=1e-8)
        assert torch.allclose(y_new, y_orig, rtol=1e-6, atol=1e-8)
        assert torch.allclose(z_new, z_orig, rtol=1e-6, atol=1e-8)

    def test_large_scale_validation(self):
        """Test with 1000 random coordinates as required by acceptance criteria"""
        torch.manual_seed(123)
        n = 1000

        # Generate random galactic coordinates
        # Avoid extreme poles where longitude is undefined
        l = torch.rand(n) * 360.0  # [0, 360)
        b = (torch.rand(n) - 0.5) * 179.0  # [-89.5, 89.5] avoid exact poles
        d = torch.rand(n) * 50.0 + 0.1  # [0.1, 50.1] kpc

        # Forward transformation
        x, y, z = galactic_to_galactocentric(l, b, d)

        # Inverse transformation
        l_rec, b_rec, d_rec = galactocentric_to_galactic(x, y, z)

        # Check absolute differences (primary metric)
        # Handle longitude wraparound
        l_diff = torch.abs(l_rec - l)
        l_diff = torch.minimum(l_diff, 360.0 - l_diff)

        # Acceptance criteria: all differences below threshold
        # Tolerances reflect floating-point precision limits
        assert torch.max(l_diff) < 0.001, f"Max l difference: {torch.max(l_diff)}"  # 0.001° ≈ 3.6"
        assert torch.allclose(b_rec, b, atol=0.005), f"Max b difference: {torch.max(torch.abs(b_rec - b))}"  # 0.005° ≈ 18"
        assert torch.allclose(d_rec, d, rtol=3e-6, atol=1e-7), f"Max d rel difference: {torch.max(torch.abs(d_rec - d) / d)}"


class TestEdgeCases:
    """Test edge cases and numerical stability"""

    def test_zero_distance(self):
        """Test with zero distance (observer at Sun)"""
        l = torch.tensor(45.0)
        b = torch.tensor(30.0)
        d = torch.tensor(0.0)

        x, y, z = galactic_to_galactocentric(l, b, d)

        # Should be at Sun's position
        assert torch.isclose(x, torch.tensor(0.0), atol=1e-8)
        assert torch.isclose(y, torch.tensor(8.5), atol=1e-8)
        assert torch.isclose(z, torch.tensor(0.0), atol=1e-8)

    def test_very_large_distance(self):
        """Test with very large distances"""
        l = torch.tensor(45.0)
        b = torch.tensor(30.0)
        d = torch.tensor(1000.0)

        x, y, z = galactic_to_galactocentric(l, b, d)
        l_rec, b_rec, d_rec = galactocentric_to_galactic(x, y, z)

        assert torch.isclose(l_rec, l, atol=1e-4)
        assert torch.isclose(b_rec, b, atol=1e-4)
        assert torch.isclose(d_rec, d, rtol=1e-6)

    def test_custom_r_sun(self):
        """Test with custom solar radius"""
        l = torch.tensor(0.0)
        b = torch.tensor(0.0)
        d = torch.tensor(10.0)
        r_sun = 10.0

        x, y, z = galactic_to_galactocentric(l, b, d, r_sun=r_sun)

        # Galactic center should be at (0, 0, 0)
        assert torch.isclose(x, torch.tensor(0.0), atol=1e-6)
        assert torch.isclose(y, torch.tensor(0.0), atol=1e-6)
        assert torch.isclose(z, torch.tensor(0.0), atol=1e-6)

        # Inverse with same r_sun
        l_rec, b_rec, d_rec = galactocentric_to_galactic(x, y, z, r_sun=r_sun)

        assert torch.isclose(l_rec, l, atol=1e-6)
        assert torch.isclose(b_rec, b, atol=1e-6)
        assert torch.isclose(d_rec, d, atol=1e-6)
