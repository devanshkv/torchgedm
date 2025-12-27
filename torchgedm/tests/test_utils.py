"""
Unit tests for mathematical utility functions

Tests verify:
1. Numerical correctness against known values
2. Vectorization/batching support
3. Differentiability
4. Numerical stability
"""

import pytest
import torch
import numpy as np
from torchgedm.ne2001.utils import (
    sech2,
    rotation_matrix_x,
    rotation_matrix_y,
    rotation_matrix_z,
    apply_rotation,
    ellipsoid_distance,
    ellipsoid_mask,
    galactocentric_to_cylindrical,
    cylindrical_to_galactocentric,
)


class TestSech2:
    """Tests for sech²(x) function"""

    def test_known_values(self):
        """Test against known values of sech²(x)"""
        # sech²(0) = 1
        x = torch.tensor(0.0)
        result = sech2(x)
        assert torch.allclose(result, torch.tensor(1.0), atol=1e-6, rtol=1e-6)

        # sech²(∞) → 0
        x = torch.tensor(10.0)  # Large value approximates infinity
        result = sech2(x)
        assert result < 1e-4

        # Known value: sech²(1) = 1/cosh²(1) ≈ 0.4199...
        x = torch.tensor(1.0)
        result = sech2(x)
        expected = 1.0 / (torch.cosh(x) ** 2)
        assert torch.allclose(result, expected, atol=1e-6, rtol=1e-6)

    def test_batched(self):
        """Test vectorized computation on batched inputs"""
        x = torch.tensor([0.0, 1.0, 2.0, -1.0, -2.0])
        result = sech2(x)

        # Verify each element
        for i, xi in enumerate(x):
            expected = 1.0 / (torch.cosh(xi) ** 2)
            assert torch.allclose(result[i], expected, atol=1e-6, rtol=1e-6)

    def test_symmetry(self):
        """Test that sech²(-x) = sech²(x)"""
        x = torch.tensor([0.5, 1.0, 1.5, 2.0])
        pos_result = sech2(x)
        neg_result = sech2(-x)
        assert torch.allclose(pos_result, neg_result, atol=1e-6, rtol=1e-6)

    def test_differentiable(self):
        """Test that function is differentiable"""
        x = torch.tensor(1.0, requires_grad=True)
        y = sech2(x)
        y.backward()

        # Check gradient exists and is non-zero
        assert x.grad is not None
        assert x.grad.abs() > 1e-6


class TestRotationMatrices:
    """Tests for 3D rotation matrix functions"""

    def test_rotation_z_90deg(self):
        """Test 90-degree rotation around Z-axis"""
        theta = torch.tensor(np.pi / 2)  # 90 degrees
        R = rotation_matrix_z(theta)

        # Rotate point (1, 0, 0) should give (0, 1, 0)
        point = torch.tensor([1.0, 0.0, 0.0])
        rotated = apply_rotation(point, R)
        expected = torch.tensor([0.0, 1.0, 0.0])

        assert torch.allclose(rotated, expected, atol=1e-6, rtol=1e-6)

    def test_rotation_y_90deg(self):
        """Test 90-degree rotation around Y-axis"""
        theta = torch.tensor(np.pi / 2)  # 90 degrees
        R = rotation_matrix_y(theta)

        # Rotate point (1, 0, 0) should give (0, 0, -1)
        point = torch.tensor([1.0, 0.0, 0.0])
        rotated = apply_rotation(point, R)
        expected = torch.tensor([0.0, 0.0, -1.0])

        assert torch.allclose(rotated, expected, atol=1e-6, rtol=1e-6)

    def test_rotation_x_90deg(self):
        """Test 90-degree rotation around X-axis"""
        theta = torch.tensor(np.pi / 2)  # 90 degrees
        R = rotation_matrix_x(theta)

        # Rotate point (0, 1, 0) should give (0, 0, 1)
        point = torch.tensor([0.0, 1.0, 0.0])
        rotated = apply_rotation(point, R)
        expected = torch.tensor([0.0, 0.0, 1.0])

        assert torch.allclose(rotated, expected, atol=1e-6, rtol=1e-6)

    def test_rotation_identity(self):
        """Test that zero rotation gives identity matrix"""
        theta = torch.tensor(0.0)

        R_x = rotation_matrix_x(theta)
        R_y = rotation_matrix_y(theta)
        R_z = rotation_matrix_z(theta)

        identity = torch.eye(3)

        assert torch.allclose(R_x, identity, atol=1e-6, rtol=1e-6)
        assert torch.allclose(R_y, identity, atol=1e-6, rtol=1e-6)
        assert torch.allclose(R_z, identity, atol=1e-6, rtol=1e-6)

    def test_rotation_batched(self):
        """Test batched rotation matrices"""
        thetas = torch.tensor([0.0, np.pi/2, np.pi])
        R = rotation_matrix_z(thetas)

        # Should have shape (3, 3, 3)
        assert R.shape == (3, 3, 3)

        # Check identity for first rotation
        assert torch.allclose(R[0], torch.eye(3), atol=1e-6, rtol=1e-6)

    def test_rotation_orthogonal(self):
        """Test that rotation matrices are orthogonal (R^T R = I)"""
        theta = torch.tensor(0.7)

        for rotation_fn in [rotation_matrix_x, rotation_matrix_y, rotation_matrix_z]:
            R = rotation_fn(theta)
            product = torch.matmul(R.T, R)
            identity = torch.eye(3)

            assert torch.allclose(product, identity, atol=1e-6, rtol=1e-6)

    def test_rotation_determinant(self):
        """Test that rotation matrices have determinant = 1"""
        theta = torch.tensor(1.2)

        for rotation_fn in [rotation_matrix_x, rotation_matrix_y, rotation_matrix_z]:
            R = rotation_fn(theta)
            det = torch.linalg.det(R)

            assert torch.allclose(det, torch.tensor(1.0), atol=1e-6, rtol=1e-6)


class TestEllipsoid:
    """Tests for ellipsoid distance and masking functions"""

    def test_sphere_distance(self):
        """Test distance for simple sphere (a=b=c)"""
        # Sphere centered at origin with radius 1
        center = (0.0, 0.0, 0.0)
        axes = (1.0, 1.0, 1.0)

        # Point at (1, 0, 0) should be on surface (distance = 1)
        point = torch.tensor([1.0, 0.0, 0.0])
        distance = ellipsoid_distance(point, center, axes)
        assert torch.allclose(distance, torch.tensor(1.0), atol=1e-6, rtol=1e-6)

        # Point at (0.5, 0, 0) should be inside (distance = 0.5)
        point = torch.tensor([0.5, 0.0, 0.0])
        distance = ellipsoid_distance(point, center, axes)
        assert torch.allclose(distance, torch.tensor(0.5), atol=1e-6, rtol=1e-6)

        # Point at (2, 0, 0) should be outside (distance = 2)
        point = torch.tensor([2.0, 0.0, 0.0])
        distance = ellipsoid_distance(point, center, axes)
        assert torch.allclose(distance, torch.tensor(2.0), atol=1e-6, rtol=1e-6)

    def test_ellipsoid_distance(self):
        """Test distance for true ellipsoid"""
        center = (0.0, 0.0, 0.0)
        axes = (2.0, 1.0, 0.5)  # Different semi-axes

        # Point at (2, 0, 0) should be on surface
        point = torch.tensor([2.0, 0.0, 0.0])
        distance = ellipsoid_distance(point, center, axes)
        assert torch.allclose(distance, torch.tensor(1.0), atol=1e-6, rtol=1e-6)

        # Point at (0, 1, 0) should be on surface
        point = torch.tensor([0.0, 1.0, 0.0])
        distance = ellipsoid_distance(point, center, axes)
        assert torch.allclose(distance, torch.tensor(1.0), atol=1e-6, rtol=1e-6)

        # Point at (0, 0, 0.5) should be on surface
        point = torch.tensor([0.0, 0.0, 0.5])
        distance = ellipsoid_distance(point, center, axes)
        assert torch.allclose(distance, torch.tensor(1.0), atol=1e-6, rtol=1e-6)

    def test_ellipsoid_translated(self):
        """Test ellipsoid with non-zero center"""
        center = (1.0, 2.0, 3.0)
        axes = (1.0, 1.0, 1.0)

        # Point at center should have distance = 0
        point = torch.tensor([1.0, 2.0, 3.0])
        distance = ellipsoid_distance(point, center, axes)
        assert torch.allclose(distance, torch.tensor(0.0), atol=1e-6, rtol=1e-6)

        # Point at (2, 2, 3) should have distance = 1 (on surface)
        point = torch.tensor([2.0, 2.0, 3.0])
        distance = ellipsoid_distance(point, center, axes)
        assert torch.allclose(distance, torch.tensor(1.0), atol=1e-6, rtol=1e-6)

    def test_ellipsoid_rotated(self):
        """Test rotated ellipsoid"""
        center = (0.0, 0.0, 0.0)
        axes = (2.0, 1.0, 1.0)

        # Rotate 90 degrees around Z-axis
        theta = torch.tensor(np.pi / 2)

        # Before rotation, (2, 0, 0) is on surface
        # After rotation, (0, 2, 0) should be on surface
        point = torch.tensor([0.0, 2.0, 0.0])
        distance = ellipsoid_distance(point, center, axes, theta=theta)
        assert torch.allclose(distance, torch.tensor(1.0), atol=1e-5, rtol=1e-5)

    def test_ellipsoid_mask(self):
        """Test ellipsoid boolean mask"""
        center = (0.0, 0.0, 0.0)
        axes = (1.0, 1.0, 1.0)

        # Points inside, on, and outside the sphere
        points = torch.tensor([
            [0.0, 0.0, 0.0],    # Inside (center)
            [0.5, 0.0, 0.0],    # Inside
            [1.0, 0.0, 0.0],    # On surface
            [2.0, 0.0, 0.0],    # Outside
        ])

        mask = ellipsoid_mask(points, center, axes)
        expected = torch.tensor([True, True, True, False])

        assert torch.all(mask == expected)

    def test_ellipsoid_batched(self):
        """Test batched ellipsoid distance computation"""
        center = (0.0, 0.0, 0.0)
        axes = (1.0, 1.0, 1.0)

        # Multiple points
        points = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])

        distances = ellipsoid_distance(points, center, axes)
        expected = torch.tensor([0.0, 0.5, 1.0, 2.0])

        assert torch.allclose(distances, expected, atol=1e-6, rtol=1e-6)


class TestCylindricalCoordinates:
    """Tests for cylindrical coordinate transformations"""

    def test_cartesian_to_cylindrical(self):
        """Test conversion from Cartesian to cylindrical"""
        # Point on positive x-axis
        x = torch.tensor(1.0)
        y = torch.tensor(0.0)
        z = torch.tensor(0.0)

        R, phi, z_out = galactocentric_to_cylindrical(x, y, z)

        assert torch.allclose(R, torch.tensor(1.0), atol=1e-6, rtol=1e-6)
        assert torch.allclose(phi, torch.tensor(0.0), atol=1e-6, rtol=1e-6)
        assert torch.allclose(z_out, torch.tensor(0.0), atol=1e-6, rtol=1e-6)

        # Point on positive y-axis
        x = torch.tensor(0.0)
        y = torch.tensor(1.0)
        z = torch.tensor(0.0)

        R, phi, z_out = galactocentric_to_cylindrical(x, y, z)

        assert torch.allclose(R, torch.tensor(1.0), atol=1e-6, rtol=1e-6)
        assert torch.allclose(phi, torch.tensor(np.pi/2), atol=1e-6, rtol=1e-6)
        assert torch.allclose(z_out, torch.tensor(0.0), atol=1e-6, rtol=1e-6)

    def test_cylindrical_to_cartesian(self):
        """Test conversion from cylindrical to Cartesian"""
        # R=1, phi=0
        R = torch.tensor(1.0)
        phi = torch.tensor(0.0)
        z = torch.tensor(0.0)

        x, y, z_out = cylindrical_to_galactocentric(R, phi, z)

        assert torch.allclose(x, torch.tensor(1.0), atol=1e-6, rtol=1e-6)
        assert torch.allclose(y, torch.tensor(0.0), atol=1e-6, rtol=1e-6)
        assert torch.allclose(z_out, torch.tensor(0.0), atol=1e-6, rtol=1e-6)

        # R=1, phi=π/2
        R = torch.tensor(1.0)
        phi = torch.tensor(np.pi / 2)
        z = torch.tensor(0.0)

        x, y, z_out = cylindrical_to_galactocentric(R, phi, z)

        assert torch.allclose(x, torch.tensor(0.0), atol=1e-6, rtol=1e-6)
        assert torch.allclose(y, torch.tensor(1.0), atol=1e-6, rtol=1e-6)
        assert torch.allclose(z_out, torch.tensor(0.0), atol=1e-6, rtol=1e-6)

    def test_round_trip(self):
        """Test round-trip conversion"""
        # Random Cartesian coordinates
        x_orig = torch.tensor([1.0, 2.0, -1.5])
        y_orig = torch.tensor([0.5, -1.0, 2.0])
        z_orig = torch.tensor([0.3, 1.2, -0.7])

        # Convert to cylindrical and back
        R, phi, z = galactocentric_to_cylindrical(x_orig, y_orig, z_orig)
        x_new, y_new, z_new = cylindrical_to_galactocentric(R, phi, z)

        assert torch.allclose(x_new, x_orig, atol=1e-6, rtol=1e-6)
        assert torch.allclose(y_new, y_orig, atol=1e-6, rtol=1e-6)
        assert torch.allclose(z_new, z_orig, atol=1e-6, rtol=1e-6)


class TestDifferentiability:
    """Tests to ensure all functions are differentiable"""

    def test_sech2_grad(self):
        """Test sech2 gradient computation"""
        x = torch.tensor(1.0, requires_grad=True)
        y = sech2(x)
        y.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad)
        assert not torch.isinf(x.grad)

    def test_ellipsoid_grad(self):
        """Test ellipsoid distance gradient"""
        point = torch.tensor([1.0, 0.5, 0.3], requires_grad=True)
        center = (0.0, 0.0, 0.0)
        axes = (1.0, 1.0, 1.0)

        distance = ellipsoid_distance(point, center, axes)
        distance.backward()

        assert point.grad is not None
        assert not torch.any(torch.isnan(point.grad))
        assert not torch.any(torch.isinf(point.grad))

    def test_rotation_grad(self):
        """Test rotation gradient computation"""
        theta = torch.tensor(0.5, requires_grad=True)
        R = rotation_matrix_z(theta)
        point = torch.tensor([1.0, 0.0, 0.0])
        rotated = apply_rotation(point, R)

        # Compute loss as sum of rotated coordinates
        loss = rotated.sum()
        loss.backward()

        assert theta.grad is not None
        assert not torch.isnan(theta.grad)
        assert not torch.isinf(theta.grad)


class TestNumericalStability:
    """Tests for numerical stability with edge cases"""

    def test_sech2_large_values(self):
        """Test sech2 with large values (should not overflow)"""
        x = torch.tensor([100.0, -100.0, 1000.0])
        result = sech2(x)

        assert not torch.any(torch.isnan(result))
        assert not torch.any(torch.isinf(result))
        assert torch.all(result >= 0)
        assert torch.all(result <= 1)  # sech²(x) ∈ [0, 1]

    def test_ellipsoid_zero_axes(self):
        """Test ellipsoid with very small axes (near-degenerate)"""
        center = (0.0, 0.0, 0.0)
        axes = (1e-10, 1e-10, 1e-10)

        point = torch.tensor([0.0, 0.0, 0.0])
        distance = ellipsoid_distance(point, center, axes)

        assert not torch.isnan(distance)
        assert not torch.isinf(distance)
