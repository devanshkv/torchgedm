"""
Unit tests for DM-distance integration functions.

Tests both dist_to_dm and dm_to_dist conversions against known values
and validates internal consistency (round-trip conversions).
"""

import pytest
import torch
import numpy as np
from torchgedm.ne2001 import dist_to_dm, dm_to_dist, NE2001Data


@pytest.fixture
def data():
    """Load NE2001 data once for all tests"""
    return NE2001Data()


class TestDistToDM:
    """Tests for distance -> DM conversion"""

    def test_zero_distance(self, data):
        """DM should be zero at zero distance"""
        l = torch.tensor(0.0)
        b = torch.tensor(0.0)
        dist = torch.tensor(0.0)

        dm = dist_to_dm(l, b, dist, data)

        assert torch.isclose(dm, torch.tensor(0.0), atol=1e-6)

    def test_sun_position_small_distance(self, data):
        """DM at small distance from Sun should be small and positive"""
        l = torch.tensor(0.0)
        b = torch.tensor(0.0)
        dist = torch.tensor(0.01)  # 10 pc

        dm = dist_to_dm(l, b, dist, data)

        # Should be small positive value (roughly ne * distance)
        # Expected: ~0.03 cm^-3 * 10 pc = 0.3 pc/cm3
        assert dm > 0.0
        assert dm < 10.0  # Sanity check

    def test_galactic_center_direction(self, data):
        """
        DM towards Galactic center should be higher due to higher density.
        l=0, b=0 points towards GC.
        """
        l = torch.tensor(0.0)
        b = torch.tensor(0.0)
        dist = torch.tensor(1.0)  # 1 kpc

        dm = dist_to_dm(l, b, dist, data)

        # Should be substantial (several tens of pc/cm3)
        assert dm > 10.0
        assert dm < 200.0

    def test_high_latitude_lower_dm(self, data):
        """
        DM at high latitude should be lower (out of plane, lower density)
        """
        l_low = torch.tensor(0.0)
        b_low = torch.tensor(0.0)
        l_high = torch.tensor(0.0)
        b_high = torch.tensor(60.0)  # 60 degrees above plane
        dist = torch.tensor(1.0)

        dm_low = dist_to_dm(l_low, b_low, dist, data)
        dm_high = dist_to_dm(l_high, b_high, dist, data)

        # High latitude should have lower DM
        assert dm_high < dm_low

    def test_linearity_at_small_distances(self, data):
        """
        At small distances, DM should be approximately linear with distance
        (density doesn't change much over short distances)
        """
        l = torch.tensor(90.0)
        b = torch.tensor(0.0)
        dist1 = torch.tensor(0.1)
        dist2 = torch.tensor(0.2)

        dm1 = dist_to_dm(l, b, dist1, data)
        dm2 = dist_to_dm(l, b, dist2, data)

        # DM2 should be approximately 2x DM1 (within 20% tolerance)
        ratio = dm2 / dm1
        assert torch.isclose(ratio, torch.tensor(2.0), rtol=0.2)

    def test_large_distance_saturation(self, data):
        """DM should saturate at very large distances (low density at edges)"""
        l = torch.tensor(0.0)
        b = torch.tensor(0.0)
        dist1 = torch.tensor(10.0)
        dist2 = torch.tensor(20.0)

        dm1 = dist_to_dm(l, b, dist1, data)
        dm2 = dist_to_dm(l, b, dist2, data)

        # DM should increase, but not linearly (saturation)
        assert dm2 > dm1
        # At large distances, doubling distance should not double DM
        ratio = dm2 / dm1
        assert ratio < 2.0

    def test_differentiability(self, data):
        """dist_to_dm should be differentiable with respect to distance"""
        l = torch.tensor(0.0)
        b = torch.tensor(0.0)
        dist = torch.tensor(1.0, requires_grad=True)

        dm = dist_to_dm(l, b, dist, data)
        dm.backward()

        # Gradient should exist and be positive
        assert dist.grad is not None
        assert dist.grad > 0.0

    def test_multiple_directions(self, data):
        """Test various sky directions"""
        test_cases = [
            (0.0, 0.0, 1.0),      # Towards GC
            (180.0, 0.0, 1.0),    # Anti-GC
            (90.0, 0.0, 1.0),     # Tangent point
            (270.0, 0.0, 1.0),    # Opposite tangent
            (45.0, 45.0, 1.0),    # Above plane
            (135.0, -30.0, 1.0),  # Below plane
        ]

        for l_val, b_val, d_val in test_cases:
            l = torch.tensor(l_val)
            b = torch.tensor(b_val)
            dist = torch.tensor(d_val)

            dm = dist_to_dm(l, b, dist, data)

            # All should give positive DM
            assert dm >= 0.0, f"Failed for l={l_val}, b={b_val}"
            assert dm < 500.0, f"Unexpectedly high DM for l={l_val}, b={b_val}"


class TestDMToDist:
    """Tests for DM -> distance conversion"""

    def test_zero_dm(self, data):
        """Distance should be zero for zero DM"""
        l = torch.tensor(0.0)
        b = torch.tensor(0.0)
        dm = torch.tensor(0.0)

        dist, is_limit = dm_to_dist(l, b, dm, data)

        assert torch.isclose(dist, torch.tensor(0.0), atol=1e-6)
        assert not is_limit

    def test_small_dm(self, data):
        """Small DM should give small distance"""
        l = torch.tensor(0.0)
        b = torch.tensor(0.0)
        dm = torch.tensor(1.0)  # 1 pc/cm3

        dist, is_limit = dm_to_dist(l, b, dm, data)

        # Should be less than 1 kpc (rough check)
        assert dist > 0.0
        assert dist < 1.0
        assert not is_limit

    def test_moderate_dm(self, data):
        """Moderate DM should give reasonable distance"""
        l = torch.tensor(0.0)
        b = torch.tensor(0.0)
        dm = torch.tensor(100.0)

        dist, is_limit = dm_to_dist(l, b, dm, data)

        # Should be several kpc
        assert dist > 1.0
        assert dist < 20.0
        assert not is_limit

    def test_high_latitude_longer_path(self, data):
        """
        At high latitude, same DM requires longer path (lower density)
        """
        l = torch.tensor(0.0)
        b_low = torch.tensor(0.0)
        b_high = torch.tensor(60.0)
        dm = torch.tensor(50.0)

        dist_low, _ = dm_to_dist(l, b_low, dm, data)
        dist_high, _ = dm_to_dist(l, b_high, dm, data)

        # High latitude needs longer path for same DM
        assert dist_high > dist_low

    def test_very_large_dm_boundary(self, data):
        """Very large DM may hit boundary and return lower limit"""
        l = torch.tensor(0.0)
        b = torch.tensor(0.0)
        dm = torch.tensor(10000.0)  # Unrealistically large

        dist, is_limit = dm_to_dist(l, b, dm, data)

        # Should either reach boundary or give very large distance
        # is_limit flag may be True if boundary hit
        assert dist > 0.0
        # Either hit boundary or reached very far
        assert dist <= 50.0  # D_MAX boundary

    def test_differentiability(self, data):
        """dm_to_dist should be differentiable with respect to DM"""
        l = torch.tensor(0.0)
        b = torch.tensor(0.0)
        dm = torch.tensor(50.0, requires_grad=True)

        dist, _ = dm_to_dist(l, b, dm, data)
        dist.backward()

        # Gradient should exist and be positive
        assert dm.grad is not None
        assert dm.grad > 0.0


class TestRoundTrip:
    """Tests for consistency between dist_to_dm and dm_to_dist"""

    def test_roundtrip_small_distance(self, data):
        """dist -> DM -> dist should recover original distance"""
        l = torch.tensor(0.0)
        b = torch.tensor(0.0)
        dist_original = torch.tensor(0.5)

        # dist -> DM
        dm = dist_to_dm(l, b, dist_original, data)

        # DM -> dist
        dist_recovered, is_limit = dm_to_dist(l, b, dm, data)

        # Should recover original distance (within tolerance)
        assert not is_limit
        rel_error = torch.abs(dist_recovered - dist_original) / dist_original
        assert rel_error < 0.01, f"Round-trip error: {rel_error*100:.2f}%"

    def test_roundtrip_moderate_distance(self, data):
        """Test round-trip at moderate distance"""
        l = torch.tensor(90.0)
        b = torch.tensor(0.0)
        dist_original = torch.tensor(5.0)

        dm = dist_to_dm(l, b, dist_original, data)
        dist_recovered, is_limit = dm_to_dist(l, b, dm, data)

        assert not is_limit
        rel_error = torch.abs(dist_recovered - dist_original) / dist_original
        assert rel_error < 0.01

    def test_roundtrip_high_latitude(self, data):
        """Test round-trip at high latitude"""
        l = torch.tensor(45.0)
        b = torch.tensor(45.0)
        dist_original = torch.tensor(2.0)

        dm = dist_to_dm(l, b, dist_original, data)
        dist_recovered, is_limit = dm_to_dist(l, b, dm, data)

        assert not is_limit
        rel_error = torch.abs(dist_recovered - dist_original) / dist_original
        assert rel_error < 0.01

    def test_roundtrip_various_directions(self, data):
        """Test round-trip for various sky directions"""
        test_cases = [
            (0.0, 0.0, 1.0),
            (180.0, 0.0, 3.0),
            (90.0, 0.0, 2.0),
            (270.0, 0.0, 4.0),
            (45.0, 30.0, 1.5),
        ]

        for l_val, b_val, d_val in test_cases:
            l = torch.tensor(l_val)
            b = torch.tensor(b_val)
            dist_original = torch.tensor(d_val)

            dm = dist_to_dm(l, b, dist_original, data)
            dist_recovered, is_limit = dm_to_dist(l, b, dm, data)

            assert not is_limit, f"Hit limit for l={l_val}, b={b_val}"
            rel_error = torch.abs(dist_recovered - dist_original) / dist_original
            assert rel_error < 0.01, f"Round-trip failed for l={l_val}, b={b_val}: error={rel_error*100:.2f}%"


class TestComparisons:
    """Tests comparing directions and validating physical expectations"""

    def test_gc_vs_antigc(self, data):
        """
        Towards GC should have higher DM than away from GC (at same distance)
        """
        b = torch.tensor(0.0)
        dist = torch.tensor(2.0)

        dm_gc = dist_to_dm(torch.tensor(0.0), b, dist, data)       # Towards GC
        dm_antigc = dist_to_dm(torch.tensor(180.0), b, dist, data)  # Away from GC

        assert dm_gc > dm_antigc

    def test_positive_dm_everywhere(self, data):
        """DM should be non-negative everywhere"""
        # Test grid of directions
        l_vals = torch.linspace(0, 360, 8)
        b_vals = torch.linspace(-60, 60, 5)
        dist = torch.tensor(1.0)

        for l_val in l_vals:
            for b_val in b_vals:
                dm = dist_to_dm(l_val, b_val, dist, data)
                assert dm >= 0.0, f"Negative DM at l={l_val:.0f}, b={b_val:.0f}"

    def test_monotonic_with_distance(self, data):
        """DM should increase monotonically with distance (along same LOS)"""
        l = torch.tensor(45.0)
        b = torch.tensor(15.0)
        distances = torch.tensor([0.5, 1.0, 2.0, 4.0, 8.0])

        dms = [dist_to_dm(l, b, d, data) for d in distances]

        # Check monotonicity
        for i in range(len(dms) - 1):
            assert dms[i+1] > dms[i], f"DM not monotonic at distance {distances[i+1]}"


class TestNumericalStability:
    """Tests for numerical stability and edge cases"""

    def test_very_small_distance(self, data):
        """Test with very small distance (1 pc)"""
        l = torch.tensor(0.0)
        b = torch.tensor(0.0)
        dist = torch.tensor(0.001)  # 1 pc

        dm = dist_to_dm(l, b, dist, data)

        assert torch.isfinite(dm)
        assert dm >= 0.0

    def test_integration_step_independence(self, data):
        """Result should be similar with different step sizes"""
        l = torch.tensor(0.0)
        b = torch.tensor(0.0)
        dist = torch.tensor(2.0)

        # Default step
        dm_default = dist_to_dm(l, b, dist, data, dstep=0.01)

        # Finer step
        dm_fine = dist_to_dm(l, b, dist, data, dstep=0.005)

        # Should agree to within 0.5%
        rel_diff = torch.abs(dm_fine - dm_default) / dm_default
        assert rel_diff < 0.005

    def test_adaptive_stepping_small_distance(self, data):
        """Adaptive stepping should work for very small distances"""
        l = torch.tensor(0.0)
        b = torch.tensor(0.0)
        dist = torch.tensor(0.05)  # 50 pc (requires small steps)

        dm = dist_to_dm(l, b, dist, data)

        assert torch.isfinite(dm)
        assert dm > 0.0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
