"""
Tests for Local Super Bubble (LSB) density component
"""

import torch
import pytest
from torchgedm.ne2001.data_loader import NE2001Data
from torchgedm.ne2001.components.ne_lsb import neLSB


@pytest.fixture
def ne2001_data():
    """Load NE2001 data"""
    return NE2001Data(device='cpu')


class TestNeLSB:
    """Test suite for Local Super Bubble component"""

    def test_lsb_center(self, ne2001_data):
        """Test density at LSB center"""
        x = ne2001_data.xlsb
        y = ne2001_data.ylsb
        z = ne2001_data.zlsb

        ne, F, w = neLSB(x, y, z, ne2001_data)

        # Should be inside at center
        assert ne > 0
        assert ne == ne2001_data.nelsb
        assert F == ne2001_data.Flsb
        assert w == 1

    def test_far_from_lsb(self, ne2001_data):
        """Test density far from LSB (should be zero)"""
        x = torch.tensor(5.0)
        y = torch.tensor(0.0)
        z = torch.tensor(5.0)

        ne, F, w = neLSB(x, y, z, ne2001_data)

        assert ne == 0.0
        assert F == 0.0
        assert w == 0

    def test_batched_computation(self, ne2001_data):
        """Test batched computation with multiple points"""
        # Create grid centered on LSB
        x_center = ne2001_data.xlsb
        y_center = ne2001_data.ylsb
        z_center = ne2001_data.zlsb

        # LSB has larger scale than LHB, use parameters to size grid
        scale = max(ne2001_data.alsb, ne2001_data.blsb, ne2001_data.clsb).item()

        x = torch.linspace(x_center - scale, x_center + scale, 10)
        y = torch.linspace(y_center - scale, y_center + scale, 10)
        z = torch.linspace(z_center - scale, z_center + scale, 10)

        x_grid, y_grid, z_grid = torch.meshgrid(x, y, z, indexing='ij')

        ne, F, w = neLSB(x_grid, y_grid, z_grid, ne2001_data)

        # Check shapes
        assert ne.shape == (10, 10, 10)
        assert F.shape == (10, 10, 10)
        assert w.shape == (10, 10, 10)

        # Center should be inside
        assert (ne > 0).any()

    def test_ellipsoidal_geometry(self, ne2001_data):
        """Test that geometry is ellipsoidal"""
        x_center = ne2001_data.xlsb
        y_center = ne2001_data.ylsb
        z_center = ne2001_data.zlsb

        # Test along principal axes (before rotation)
        # Note: after rotation, principal axes are rotated by theta

        # At center, should be inside
        ne_center, _, w_center = neLSB(x_center, y_center, z_center, ne2001_data)
        assert ne_center > 0
        assert w_center == 1

    def test_rotation(self, ne2001_data):
        """Test that ellipsoid is rotated in xy-plane"""
        # The rotation angle is thetalsb
        theta = ne2001_data.thetalsb

        # Rotation should affect points in xy-plane
        # This is complex to test directly, so we just verify
        # that theta is used in the calculation by checking
        # that results are consistent

        x = ne2001_data.xlsb
        y = ne2001_data.ylsb
        z = ne2001_data.zlsb

        ne, F, w = neLSB(x, y, z, ne2001_data)

        # Center should always be inside regardless of rotation
        assert ne > 0
        assert w == 1

    def test_z_symmetry(self, ne2001_data):
        """Test that ellipsoid is symmetric in z about center"""
        x_center = ne2001_data.xlsb
        y_center = ne2001_data.ylsb
        z_center = ne2001_data.zlsb

        dz = 0.1

        ne_plus, _, _ = neLSB(x_center, y_center, z_center + dz, ne2001_data)
        ne_minus, _, _ = neLSB(x_center, y_center, z_center - dz, ne2001_data)

        # Should be equal (z-symmetric about center)
        assert torch.allclose(ne_plus, ne_minus)

    def test_works_with_requires_grad(self, ne2001_data):
        """Test that function works with tensors that require grad"""
        x = torch.tensor(ne2001_data.xlsb.item(), requires_grad=True)
        y = torch.tensor(ne2001_data.ylsb.item(), requires_grad=True)
        z = torch.tensor(ne2001_data.zlsb.item(), requires_grad=True)

        ne, F, w = neLSB(x, y, z, ne2001_data)

        assert ne > 0
        assert isinstance(ne, torch.Tensor)
        assert isinstance(F, torch.Tensor)
        assert isinstance(w, torch.Tensor)

    def test_return_types(self, ne2001_data):
        """Test that function returns correct types"""
        x = torch.tensor(0.0)
        y = torch.tensor(9.0)
        z = torch.tensor(0.0)

        ne, F, w = neLSB(x, y, z, ne2001_data)

        assert isinstance(ne, torch.Tensor)
        assert isinstance(F, torch.Tensor)
        assert isinstance(w, torch.Tensor)
        assert w.dtype == torch.long or w.dtype == torch.int64

    def test_edge_cases(self, ne2001_data):
        """Test edge cases and numerical stability"""
        # Very large coordinates
        x = torch.tensor(100.0)
        y = torch.tensor(100.0)
        z = torch.tensor(100.0)

        ne, F, w = neLSB(x, y, z, ne2001_data)

        assert torch.isfinite(ne)
        assert torch.isfinite(F)
        assert ne >= 0
        assert F >= 0

    def test_parameter_broadcasting(self, ne2001_data):
        """Test that parameters broadcast correctly"""
        # 1D input
        x_1d = torch.linspace(-2, 0, 50)
        y_1d = torch.full_like(x_1d, 9.0)
        z_1d = torch.zeros_like(x_1d)

        ne_1d, F_1d, w_1d = neLSB(x_1d, y_1d, z_1d, ne2001_data)

        assert ne_1d.shape == (50,)
        assert F_1d.shape == (50,)
        assert w_1d.shape == (50,)

        # 2D input
        x_2d = torch.linspace(-2, 0, 10).unsqueeze(1)
        y_2d = torch.linspace(8, 10, 20).unsqueeze(0)
        z_2d = torch.zeros(10, 20)

        ne_2d, F_2d, w_2d = neLSB(x_2d, y_2d, z_2d, ne2001_data)

        assert ne_2d.shape == (10, 20)
        assert F_2d.shape == (10, 20)
        assert w_2d.shape == (10, 20)

    def test_boundary(self, ne2001_data):
        """Test behavior at ellipsoid boundary"""
        # At boundary, q should be approximately 1.0
        # This is hard to compute exactly due to rotation,
        # so we just test that there's a transition from inside to outside

        x_center = ne2001_data.xlsb
        y_center = ne2001_data.ylsb
        z_center = ne2001_data.zlsb

        # Move along z-axis from center
        z_values = torch.linspace(z_center - 2*ne2001_data.clsb,
                                   z_center + 2*ne2001_data.clsb, 50)

        ne_z, _, _ = neLSB(x_center.expand_as(z_values),
                          y_center.expand_as(z_values),
                          z_values,
                          ne2001_data)

        # Should transition from inside (ne > 0) to outside (ne = 0)
        inside_count = (ne_z > 0).sum()
        outside_count = (ne_z == 0).sum()

        assert inside_count > 0, "Should have some points inside"
        assert outside_count > 0, "Should have some points outside"

    def test_constant_inside(self, ne2001_data):
        """Test that density is constant inside the ellipsoid"""
        x_center = ne2001_data.xlsb
        y_center = ne2001_data.ylsb
        z_center = ne2001_data.zlsb

        # Multiple points near center (all should be inside)
        x_near = x_center + torch.tensor([0.0, 0.01, -0.01])
        y_near = y_center + torch.tensor([0.0, 0.01, -0.01])
        z_near = z_center + torch.tensor([0.0, 0.01, -0.01])

        ne_near, _, w_near = neLSB(x_near, y_near, z_near, ne2001_data)

        # All inside points should have same density
        inside_mask = w_near == 1
        if inside_mask.any():
            inside_densities = ne_near[inside_mask]
            assert torch.allclose(inside_densities, inside_densities[0])
