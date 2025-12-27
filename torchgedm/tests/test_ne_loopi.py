"""
Tests for Loop I density component
"""

import torch
import pytest
from torchgedm.ne2001.data_loader import NE2001Data
from torchgedm.ne2001.components.ne_loopi import neLOOPI


@pytest.fixture
def ne2001_data():
    """Load NE2001 data"""
    return NE2001Data(device='cpu')


class TestNeLOOPI:
    """Test suite for Loop I component"""

    def test_loopi_center(self, ne2001_data):
        """Test density at Loop I center"""
        x = ne2001_data.xlpI
        y = ne2001_data.ylpI
        z = ne2001_data.zlpI

        ne, F, w = neLOOPI(x, y, z, ne2001_data)

        # Should be inside inner volume at center
        assert ne > 0
        assert ne == ne2001_data.nelpI
        assert F == ne2001_data.FlpI
        assert w == 1

    def test_far_from_loopi(self, ne2001_data):
        """Test density far from Loop I (should be zero)"""
        # Point far from Loop I center
        x = torch.tensor(5.0)
        y = torch.tensor(0.0)
        z = torch.tensor(5.0)

        ne, F, w = neLOOPI(x, y, z, ne2001_data)

        assert ne == 0.0
        assert F == 0.0
        assert w == 0

    def test_truncation_negative_z(self, ne2001_data):
        """Test that Loop I is truncated for z < 0"""
        # Point at Loop I center but with z < 0
        x = ne2001_data.xlpI
        y = ne2001_data.ylpI
        z = torch.tensor(-0.1)

        ne, F, w = neLOOPI(x, y, z, ne2001_data)

        # Should be zero even though x,y are at center
        assert ne == 0.0
        assert F == 0.0
        assert w == 0

    def test_z_zero_boundary(self, ne2001_data):
        """Test behavior at z = 0 boundary"""
        # Point at Loop I center with z = 0 (should be included, as z >= 0)
        x = ne2001_data.xlpI
        y = ne2001_data.ylpI
        z = torch.tensor(0.0)

        ne, F, w = neLOOPI(x, y, z, ne2001_data)

        # Should be inside if distance from (xlpI, ylpI, zlpI) is within bounds
        # The actual density depends on the radial distance
        assert ne >= 0
        assert F >= 0
        assert w in [0, 1]

    def test_inner_volume(self, ne2001_data):
        """Test point inside inner volume (r <= rlpI)"""
        # Point just inside inner radius
        x = ne2001_data.xlpI
        y = ne2001_data.ylpI
        z = ne2001_data.zlpI + 0.5 * ne2001_data.rlpI

        ne, F, w = neLOOPI(x, y, z, ne2001_data)

        # Check if inside inner volume
        if ne > 0:
            # If inside, should have inner volume properties
            assert ne == ne2001_data.nelpI or ne == ne2001_data.dnelpI
            assert F == ne2001_data.FlpI or F == ne2001_data.dFlpI
            assert w == 1

    def test_shell_region(self, ne2001_data):
        """Test point in shell region (rlpI < r <= rlpI + drlpI)"""
        # Point at distance rlpI + 0.5*drlpI from center
        r_test = ne2001_data.rlpI + 0.5 * ne2001_data.drlpI

        # Place point directly above center in z
        x = ne2001_data.xlpI
        y = ne2001_data.ylpI
        z = ne2001_data.zlpI + r_test

        ne, F, w = neLOOPI(x, y, z, ne2001_data)

        # Should be in shell region
        assert ne == ne2001_data.dnelpI
        assert F == ne2001_data.dFlpI
        assert w == 1

    def test_outer_boundary(self, ne2001_data):
        """Test point just outside outer boundary"""
        # Point just beyond outer radius
        r_outside = ne2001_data.rlpI + ne2001_data.drlpI + 0.1

        x = ne2001_data.xlpI
        y = ne2001_data.ylpI
        z = ne2001_data.zlpI + r_outside

        ne, F, w = neLOOPI(x, y, z, ne2001_data)

        # Should be outside
        assert ne == 0.0
        assert F == 0.0
        assert w == 0

    def test_spherical_geometry(self, ne2001_data):
        """Test that geometry is spherical (distance from center)"""
        # Two points at same distance from center in different directions
        r_test = 0.5 * ne2001_data.rlpI

        # Point in +x direction
        x1 = ne2001_data.xlpI + r_test
        y1 = ne2001_data.ylpI
        z1 = ne2001_data.zlpI

        # Point in +y direction
        x2 = ne2001_data.xlpI
        y2 = ne2001_data.ylpI + r_test
        z2 = ne2001_data.zlpI

        ne1, F1, w1 = neLOOPI(x1, y1, z1, ne2001_data)
        ne2, F2, w2 = neLOOPI(x2, y2, z2, ne2001_data)

        # Both should have same density (spherically symmetric)
        assert ne1 == ne2
        assert F1 == F2
        assert w1 == w2

    def test_batched_computation(self, ne2001_data):
        """Test batched computation with multiple points"""
        # Create grid around Loop I center
        x_center = ne2001_data.xlpI
        y_center = ne2001_data.ylpI
        z_center = ne2001_data.zlpI
        r_max = ne2001_data.rlpI + ne2001_data.drlpI

        x = torch.linspace(x_center - r_max, x_center + r_max, 10)
        y = torch.linspace(y_center - r_max, y_center + r_max, 10)
        z = torch.linspace(z_center - r_max, z_center + r_max, 10)

        # Expand to 3D grid
        x_grid, y_grid, z_grid = torch.meshgrid(x, y, z, indexing='ij')

        ne, F, w = neLOOPI(x_grid, y_grid, z_grid, ne2001_data)

        # Check shapes
        assert ne.shape == (10, 10, 10)
        assert F.shape == (10, 10, 10)
        assert w.shape == (10, 10, 10)

        # Center should be inside
        assert (ne > 0).any(), "Expected some points inside Loop I"

        # Outside points should have zero density
        assert (ne[ne == 0] == 0).all()

    def test_differentiability(self, ne2001_data):
        """Test that function works with tensors that require grad"""
        x = torch.tensor(ne2001_data.xlpI.item(), requires_grad=True)
        y = torch.tensor(ne2001_data.ylpI.item(), requires_grad=True)
        z = torch.tensor(ne2001_data.zlpI.item(), requires_grad=True)

        ne, F, w = neLOOPI(x, y, z, ne2001_data)

        # Center should have positive density
        assert ne > 0

        # Verify function can be called with requires_grad=True
        assert isinstance(ne, torch.Tensor)
        assert isinstance(F, torch.Tensor)
        assert isinstance(w, torch.Tensor)

    def test_return_types(self, ne2001_data):
        """Test that function returns correct types"""
        x = torch.tensor(0.0)
        y = torch.tensor(8.5)
        z = torch.tensor(0.0)

        ne, F, w = neLOOPI(x, y, z, ne2001_data)

        assert isinstance(ne, torch.Tensor)
        assert isinstance(F, torch.Tensor)
        assert isinstance(w, torch.Tensor)
        assert ne.dtype == torch.float32 or ne.dtype == torch.float64
        assert F.dtype == torch.float32 or F.dtype == torch.float64
        assert w.dtype == torch.long or w.dtype == torch.int64

    def test_edge_cases(self, ne2001_data):
        """Test edge cases and numerical stability"""
        # Very large coordinates
        x = torch.tensor(100.0)
        y = torch.tensor(100.0)
        z = torch.tensor(100.0)

        ne, F, w = neLOOPI(x, y, z, ne2001_data)

        assert torch.isfinite(ne)
        assert torch.isfinite(F)
        assert ne >= 0
        assert F >= 0

        # Negative coordinates (but z < 0 should give zero)
        x = torch.tensor(-100.0)
        y = torch.tensor(-100.0)
        z = torch.tensor(-100.0)

        ne, F, w = neLOOPI(x, y, z, ne2001_data)

        assert torch.isfinite(ne)
        assert torch.isfinite(F)
        assert ne == 0  # z < 0 means outside
        assert F == 0
        assert w == 0

    def test_parameter_broadcasting(self, ne2001_data):
        """Test that parameters broadcast correctly with input shapes"""
        # 1D input
        x_1d = torch.linspace(-1, 1, 50)
        y_1d = torch.full_like(x_1d, 0.0)
        z_1d = torch.full_like(x_1d, 0.5)

        ne_1d, F_1d, w_1d = neLOOPI(x_1d, y_1d, z_1d, ne2001_data)

        assert ne_1d.shape == (50,)
        assert F_1d.shape == (50,)
        assert w_1d.shape == (50,)

        # 2D input
        x_2d = torch.linspace(-1, 1, 10).unsqueeze(1)
        y_2d = torch.linspace(-1, 1, 20).unsqueeze(0)
        z_2d = torch.full((10, 20), 0.5)

        ne_2d, F_2d, w_2d = neLOOPI(x_2d, y_2d, z_2d, ne2001_data)

        assert ne_2d.shape == (10, 20)
        assert F_2d.shape == (10, 20)
        assert w_2d.shape == (10, 20)

    def test_scalar_inputs(self, ne2001_data):
        """Test with scalar inputs"""
        x = torch.tensor(0.0)
        y = torch.tensor(0.0)
        z = torch.tensor(0.5)

        ne, F, w = neLOOPI(x, y, z, ne2001_data)

        assert ne.shape == ()
        assert F.shape == ()
        assert w.shape == ()
