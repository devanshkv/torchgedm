"""
Tests for spiral arms component (ne_arms).

Tests the PyTorch implementation of logarithmic spiral arms with
cubic spline interpolation.
"""

import torch
import pytest
import numpy as np
from torchgedm.ne2001.data_loader import NE2001Data
from torchgedm.ne2001.components.ne_arms import (
    generate_spiral_arm_paths,
    compute_arm_distance,
    ne_arms,
    apply_arm_sculpting,
    apply_arm_reweighting,
)


@pytest.fixture
def ne2001_data():
    """Load NE2001 parameters"""
    return NE2001Data(device='cpu')


@pytest.fixture
def spiral_arm_paths(ne2001_data):
    """Generate spiral arm paths for testing"""
    arm_x, arm_y, arm_kmax = generate_spiral_arm_paths(
        ne2001_data.arm_a,
        ne2001_data.arm_rmin,
        ne2001_data.arm_thmin,
        ne2001_data.arm_extent,
    )
    return arm_x, arm_y, arm_kmax


class TestSpiralArmGeneration:
    """Test spiral arm path generation"""

    def test_generates_correct_shapes(self, ne2001_data):
        """Test that arm generation produces correct output shapes"""
        arm_x, arm_y, arm_kmax = generate_spiral_arm_paths(
            ne2001_data.arm_a,
            ne2001_data.arm_rmin,
            ne2001_data.arm_thmin,
            ne2001_data.arm_extent,
        )

        assert arm_x.shape == (5, 1000)  # 5 arms, 1000 points each
        assert arm_y.shape == (5, 1000)
        assert arm_kmax.shape == (5,)

    def test_arm_kmax_values(self, ne2001_data):
        """Test that kmax values are reasonable"""
        arm_x, arm_y, arm_kmax = generate_spiral_arm_paths(
            ne2001_data.arm_a,
            ne2001_data.arm_rmin,
            ne2001_data.arm_thmin,
            ne2001_data.arm_extent,
        )

        # Each arm should have used some points
        assert (arm_kmax > 0).all()
        assert (arm_kmax < 1000).all()

    def test_arms_are_continuous(self, spiral_arm_paths):
        """Test that arm paths are continuous (no large jumps)"""
        arm_x, arm_y, arm_kmax = spiral_arm_paths

        for j in range(5):
            kmax = arm_kmax[j].item()
            if kmax > 1:
                # Check consecutive points are close
                dx = arm_x[j, 1:kmax] - arm_x[j, :kmax-1]
                dy = arm_y[j, 1:kmax] - arm_y[j, :kmax-1]
                dist = torch.sqrt(dx**2 + dy**2)

                # Maximum step should be reasonable (< 1 kpc)
                assert (dist < 1.0).all(), f"Arm {j} has discontinuity"

    def test_arms_form_spirals(self, spiral_arm_paths):
        """Test that arms actually spiral outward"""
        arm_x, arm_y, arm_kmax = spiral_arm_paths

        for j in range(5):
            kmax = arm_kmax[j].item()
            if kmax > 10:
                # Compute radius at several points
                r_start = torch.sqrt(arm_x[j, 0]**2 + arm_y[j, 0]**2)
                r_mid = torch.sqrt(arm_x[j, kmax//2]**2 + arm_y[j, kmax//2]**2)
                r_end = torch.sqrt(arm_x[j, kmax-1]**2 + arm_y[j, kmax-1]**2)

                # Spiral should generally increase in radius
                # (allowing for some sculpting effects)
                assert r_end > r_start * 0.5, f"Arm {j} doesn't spiral outward"

    def test_reproduces_spiral_formula(self, ne2001_data):
        """Test that unsculpted arms match logarithmic spiral formula"""
        # Test a simple arm without sculpting
        a = torch.tensor([4.0])
        rmin = torch.tensor([5.0])
        thmin = torch.tensor([0.0])
        extent = torch.tensor([3.14])

        arm_x, arm_y, arm_kmax = generate_spiral_arm_paths(
            a, rmin, thmin, extent, n_control=10, n_points=100
        )

        kmax = arm_kmax[0].item()
        r = torch.sqrt(arm_x[0, :kmax]**2 + arm_y[0, :kmax]**2)
        theta = torch.atan2(-arm_x[0, :kmax], arm_y[0, :kmax])

        # Check a few points match logarithmic spiral: r = rmin * exp(theta / a)
        # (approximately, since we used spline interpolation)
        r_expected = rmin * torch.exp((theta - thmin) / a)
        rel_error = torch.abs(r - r_expected) / r_expected
        assert rel_error.mean() < 0.1, "Doesn't match logarithmic spiral"


class TestArmSculpting:
    """Test arm sculpting (distortion) functions"""

    def test_sculpting_doesnt_crash(self):
        """Test that sculpting runs without errors"""
        theta_deg = torch.linspace(0, 360, 100)
        r = torch.ones(100) * 10.0

        for arm_idx in range(5):
            r_sculpted = apply_arm_sculpting(arm_idx, theta_deg, r, 'cpu')
            assert r_sculpted.shape == r.shape
            assert not torch.isnan(r_sculpted).any()

    def test_sculpting_modifies_specific_arms(self):
        """Test that sculpting only affects arms 2 and 3 (TC93)"""
        theta_deg = torch.linspace(300, 400, 50)
        r = torch.ones(50) * 10.0

        # Arms 0, 1, 4 should not be sculpted in this region
        for arm_idx in [0, 4]:
            r_sculpted = apply_arm_sculpting(arm_idx, theta_deg, r, 'cpu')
            assert torch.allclose(r_sculpted, r), f"Arm {arm_idx} shouldn't be sculpted"

    def test_sculpting_is_local(self):
        """Test that sculpting is localized to specific angular ranges"""
        # Test arm 2 (TC93 arm 3) which has sculpting at 180-410 degrees
        arm_idx = 1  # Maps to TC93 arm 3

        # Outside sculpted region (far from 180-410 range)
        theta_outside = torch.tensor([100.0, 150.0, 450.0])
        r_outside = torch.ones(3) * 10.0
        r_sculpted_outside = apply_arm_sculpting(arm_idx, theta_outside, r_outside, 'cpu')
        assert torch.allclose(r_sculpted_outside, r_outside), "Outside region shouldn't be sculpted"

        # Inside sculpted region (315-370 range has sculpting)
        theta_inside = torch.tensor([340.0, 350.0, 360.0])
        r_inside = torch.ones(3) * 10.0
        r_sculpted_inside = apply_arm_sculpting(arm_idx, theta_inside, r_inside, 'cpu')
        # Should be modified (all three are in the 315-370 range)
        assert not torch.allclose(r_sculpted_inside, r_inside), "Inside region should be sculpted"


class TestArmDistance:
    """Test minimum distance computation to arms"""

    def test_distance_shape(self, spiral_arm_paths):
        """Test output shapes for distance computation"""
        arm_x, arm_y, arm_kmax = spiral_arm_paths

        # Single point
        x = torch.tensor([0.0])
        y = torch.tensor([8.5])
        smin, _ = compute_arm_distance(x, y, arm_x, arm_y, arm_kmax)
        assert smin.shape == (1, 5)

        # Multiple points
        x = torch.randn(10)
        y = torch.randn(10)
        smin, _ = compute_arm_distance(x, y, arm_x, arm_y, arm_kmax)
        assert smin.shape == (10, 5)

    def test_distance_at_sun_position(self, spiral_arm_paths):
        """Test distance at Sun's position"""
        arm_x, arm_y, arm_kmax = spiral_arm_paths

        # Sun is at (0, 8.5) kpc
        x = torch.tensor([0.0])
        y = torch.tensor([8.5])
        smin, _ = compute_arm_distance(x, y, arm_x, arm_y, arm_kmax)

        # Sun should be close to one of the arms
        assert smin.min() < 2.0, "Sun should be near at least one arm"

    def test_distance_is_positive(self, spiral_arm_paths):
        """Test that distances are always non-negative"""
        arm_x, arm_y, arm_kmax = spiral_arm_paths

        x = torch.randn(20) * 10
        y = torch.randn(20) * 10
        smin, _ = compute_arm_distance(x, y, arm_x, arm_y, arm_kmax)

        assert (smin >= 0).all()

    def test_distance_symmetry(self, spiral_arm_paths):
        """Test that distance computation handles different regions"""
        arm_x, arm_y, arm_kmax = spiral_arm_paths

        # Test points in different quadrants
        x = torch.tensor([5.0, -5.0, 5.0, -5.0])
        y = torch.tensor([5.0, 5.0, -5.0, -5.0])
        smin, _ = compute_arm_distance(x, y, arm_x, arm_y, arm_kmax)

        assert smin.shape == (4, 5)
        assert not torch.isnan(smin).any()


class TestArmReweighting:
    """Test arm-specific reweighting factors"""

    def test_reweighting_doesnt_crash(self):
        """Test that reweighting runs without errors"""
        thxy = torch.linspace(0, 360, 100)
        ga = torch.ones(100)

        for arm_tc93 in range(5):
            ga_out = apply_arm_reweighting(arm_tc93, thxy, ga)
            assert ga_out.shape == ga.shape
            assert not torch.isnan(ga_out).any()

    def test_reweighting_affects_specific_arms(self):
        """Test that reweighting is arm-specific"""
        thxy = torch.tensor([350.0])
        ga = torch.ones(1)

        # Arm 2 (TC93) should be reweighted at 340-370 degrees
        ga_arm2 = apply_arm_reweighting(1, thxy, ga)  # TC93 arm 2 is index 1
        assert not torch.allclose(ga_arm2, ga), "Arm 2 should be reweighted"

        # Other arms should not
        ga_arm0 = apply_arm_reweighting(0, thxy, ga)
        assert torch.allclose(ga_arm0, ga), "Arm 0 shouldn't be reweighted"

    def test_reweighting_bounds(self):
        """Test that reweighting keeps values reasonable"""
        thxy = torch.linspace(0, 360, 200)
        ga = torch.ones(200)

        for arm_tc93 in range(5):
            ga_out = apply_arm_reweighting(arm_tc93, thxy, ga)
            # Reweighting should reduce or keep the same, not increase wildly
            assert (ga_out >= 0).all()
            assert (ga_out <= ga * 2).all()  # Allow up to 2x increase


class TestNeArms:
    """Test complete ne_arms function"""

    def test_ne_arms_runs(self, ne2001_data, spiral_arm_paths):
        """Test that ne_arms executes without errors"""
        arm_x, arm_y, arm_kmax = spiral_arm_paths

        x = torch.tensor([0.0])
        y = torch.tensor([8.5])
        z = torch.tensor([0.0])

        nea, Fa, whicharm, sminmin = ne_arms(
            x, y, z, arm_x, arm_y, arm_kmax,
            ne2001_data.na, ne2001_data.ha, ne2001_data.wa,
            ne2001_data.Aa, ne2001_data.Fa,
            ne2001_data.narm, ne2001_data.warm,
            ne2001_data.harm, ne2001_data.farm,
        )

        assert nea.shape == x.shape
        assert Fa.shape == x.shape
        assert whicharm.shape == x.shape
        assert sminmin.shape == x.shape

    def test_sun_position_has_arms(self, ne2001_data, spiral_arm_paths):
        """Test that Sun's position shows arm density"""
        arm_x, arm_y, arm_kmax = spiral_arm_paths

        # Sun is at (0, 8.5, 0) kpc
        x = torch.tensor([0.0])
        y = torch.tensor([8.5])
        z = torch.tensor([0.0])

        nea, Fa, whicharm, sminmin = ne_arms(
            x, y, z, arm_x, arm_y, arm_kmax,
            ne2001_data.na, ne2001_data.ha, ne2001_data.wa,
            ne2001_data.Aa, ne2001_data.Fa,
            ne2001_data.narm, ne2001_data.warm,
            ne2001_data.harm, ne2001_data.farm,
        )

        # Should have some arm density
        assert nea > 0, "Sun should have arm density"
        assert whicharm > 0, "Sun should be in an arm"

    def test_far_from_galaxy_has_no_arms(self, ne2001_data, spiral_arm_paths):
        """Test that points far from galaxy have no arm density"""
        arm_x, arm_y, arm_kmax = spiral_arm_paths

        # Very far from galaxy
        x = torch.tensor([50.0])
        y = torch.tensor([50.0])
        z = torch.tensor([0.0])

        nea, Fa, whicharm, sminmin = ne_arms(
            x, y, z, arm_x, arm_y, arm_kmax,
            ne2001_data.na, ne2001_data.ha, ne2001_data.wa,
            ne2001_data.Aa, ne2001_data.Fa,
            ne2001_data.narm, ne2001_data.warm,
            ne2001_data.harm, ne2001_data.farm,
        )

        # Should have no arm density
        assert nea == 0, "Far point should have no arm density"
        assert whicharm == 0, "Far point should not be in any arm"

    def test_vertical_falloff(self, ne2001_data, spiral_arm_paths):
        """Test that density falls off vertically"""
        arm_x, arm_y, arm_kmax = spiral_arm_paths

        x = torch.tensor([0.0, 0.0, 0.0])
        y = torch.tensor([8.5, 8.5, 8.5])
        z = torch.tensor([0.0, 0.5, 1.0])

        nea, _, _, _ = ne_arms(
            x, y, z, arm_x, arm_y, arm_kmax,
            ne2001_data.na, ne2001_data.ha, ne2001_data.wa,
            ne2001_data.Aa, ne2001_data.Fa,
            ne2001_data.narm, ne2001_data.warm,
            ne2001_data.harm, ne2001_data.farm,
        )

        # Density should decrease with height
        assert nea[0] > nea[1] > nea[2], "Density should fall off with z"

    def test_batched_computation(self, ne2001_data, spiral_arm_paths):
        """Test batched density computation"""
        arm_x, arm_y, arm_kmax = spiral_arm_paths

        # Random points
        n = 50
        x = torch.randn(n) * 10
        y = torch.randn(n) * 10
        z = torch.randn(n) * 0.5

        nea, Fa, whicharm, sminmin = ne_arms(
            x, y, z, arm_x, arm_y, arm_kmax,
            ne2001_data.na, ne2001_data.ha, ne2001_data.wa,
            ne2001_data.Aa, ne2001_data.Fa,
            ne2001_data.narm, ne2001_data.warm,
            ne2001_data.harm, ne2001_data.farm,
        )

        assert nea.shape == (n,)
        assert Fa.shape == (n,)
        assert whicharm.shape == (n,)
        assert not torch.isnan(nea).any()

    def test_arm_identification(self, ne2001_data, spiral_arm_paths):
        """Test that arm identification returns valid arm numbers"""
        arm_x, arm_y, arm_kmax = spiral_arm_paths

        # Grid of points
        x = torch.linspace(-10, 10, 20)
        y = torch.linspace(-10, 10, 20)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = torch.zeros_like(X_flat)

        _, _, whicharm, _ = ne_arms(
            X_flat, Y_flat, Z_flat, arm_x, arm_y, arm_kmax,
            ne2001_data.na, ne2001_data.ha, ne2001_data.wa,
            ne2001_data.Aa, ne2001_data.Fa,
            ne2001_data.narm, ne2001_data.warm,
            ne2001_data.harm, ne2001_data.farm,
        )

        # Arm numbers should be 0-5
        assert (whicharm >= 0).all()
        assert (whicharm <= 5).all()

        # At least some points should be in arms
        assert (whicharm > 0).any(), "Some points should be in arms"


class TestNumericalProperties:
    """Test numerical properties and stability"""

    def test_differentiability(self, ne2001_data, spiral_arm_paths):
        """Test that density computation is differentiable"""
        arm_x, arm_y, arm_kmax = spiral_arm_paths

        x = torch.tensor([0.0], requires_grad=True)
        y = torch.tensor([8.5], requires_grad=True)
        z = torch.tensor([0.0], requires_grad=True)

        nea, _, _, _ = ne_arms(
            x, y, z, arm_x, arm_y, arm_kmax,
            ne2001_data.na, ne2001_data.ha, ne2001_data.wa,
            ne2001_data.Aa, ne2001_data.Fa,
            ne2001_data.narm, ne2001_data.warm,
            ne2001_data.harm, ne2001_data.farm,
        )

        # Should be able to compute gradients
        if nea > 0:  # Only if there's density
            nea.backward()
            assert x.grad is not None
            assert not torch.isnan(x.grad)

    def test_no_nans_or_infs(self, ne2001_data, spiral_arm_paths):
        """Test that computation doesn't produce NaNs or Infs"""
        arm_x, arm_y, arm_kmax = spiral_arm_paths

        # Test many random points
        x = torch.randn(100) * 20
        y = torch.randn(100) * 20
        z = torch.randn(100) * 2

        nea, Fa, whicharm, sminmin = ne_arms(
            x, y, z, arm_x, arm_y, arm_kmax,
            ne2001_data.na, ne2001_data.ha, ne2001_data.wa,
            ne2001_data.Aa, ne2001_data.Fa,
            ne2001_data.narm, ne2001_data.warm,
            ne2001_data.harm, ne2001_data.farm,
        )

        assert not torch.isnan(nea).any()
        assert not torch.isinf(nea).any()
        assert not torch.isnan(Fa).any()

    def test_density_non_negative(self, ne2001_data, spiral_arm_paths):
        """Test that density is always non-negative"""
        arm_x, arm_y, arm_kmax = spiral_arm_paths

        x = torch.randn(100) * 15
        y = torch.randn(100) * 15
        z = torch.randn(100)

        nea, _, _, _ = ne_arms(
            x, y, z, arm_x, arm_y, arm_kmax,
            ne2001_data.na, ne2001_data.ha, ne2001_data.wa,
            ne2001_data.Aa, ne2001_data.Fa,
            ne2001_data.narm, ne2001_data.warm,
            ne2001_data.harm, ne2001_data.farm,
        )

        assert (nea >= 0).all(), "Density should be non-negative"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
