"""
Unit tests for DENSITY_2001 (combined density function)

Tests the main density calculation that combines all 7 components of NE2001.
"""

import torch
import pytest
from torchgedm.ne2001 import density_2001, DensityResult, NE2001Data


class TestDensity2001:
    """Test suite for density_2001 function"""

    @pytest.fixture
    def ne2001_data(self):
        """Load NE2001 data for testing"""
        return NE2001Data(device='cpu')

    def test_density_at_sun_position(self, ne2001_data):
        """Test density at Sun's position (x=0, y=8.5, z=0)"""
        x = torch.tensor(0.0)
        y = torch.tensor(8.5)
        z = torch.tensor(0.0)

        result = density_2001(x, y, z, ne2001_data)

        # Verify result is DensityResult
        assert isinstance(result, DensityResult)

        # All densities should be non-negative
        assert result.ne1 >= 0
        assert result.ne2 >= 0
        assert result.nea >= 0
        assert result.negc >= 0
        assert result.nelism >= 0
        assert result.necN >= 0
        assert result.nevN >= 0

        # Total density should be finite
        total_ne = result.total_density(ne2001_data)
        assert torch.isfinite(total_ne)
        assert total_ne >= 0

    def test_density_at_galactic_center(self, ne2001_data):
        """Test density at Galactic center (x=0, y=0, z=0)"""
        x = torch.tensor(0.0)
        y = torch.tensor(0.0)
        z = torch.tensor(0.0)

        result = density_2001(x, y, z, ne2001_data)

        # At GC, we expect some density (thick disk + possibly GC component)
        total_ne = result.total_density(ne2001_data)
        assert total_ne > 0
        assert torch.isfinite(total_ne)

    def test_density_batched_single_point(self, ne2001_data):
        """Test batched computation with single point matches scalar"""
        x_scalar = torch.tensor(1.0)
        y_scalar = torch.tensor(8.5)
        z_scalar = torch.tensor(0.1)

        x_batch = torch.tensor([1.0])
        y_batch = torch.tensor([8.5])
        z_batch = torch.tensor([0.1])

        result_scalar = density_2001(x_scalar, y_scalar, z_scalar, ne2001_data)
        result_batch = density_2001(x_batch, y_batch, z_batch, ne2001_data)

        # Results should match (within numerical precision)
        assert torch.allclose(result_scalar.ne1, result_batch.ne1[0], atol=1e-10)
        assert torch.allclose(result_scalar.ne2, result_batch.ne2[0], atol=1e-10)
        assert torch.allclose(result_scalar.total_density(ne2001_data), result_batch.total_density(ne2001_data)[0], atol=1e-10)

    def test_density_batched_10_points(self, ne2001_data):
        """Test batched computation with 10 random points"""
        torch.manual_seed(42)
        n_points = 10

        # Random galactocentric coordinates
        x = torch.randn(n_points) * 5.0  # ±5 kpc
        y = torch.randn(n_points) * 5.0 + 8.5  # around Sun
        z = torch.randn(n_points) * 1.0  # ±1 kpc

        result = density_2001(x, y, z, ne2001_data)

        # All results should have shape (10,)
        assert result.ne1.shape == (n_points,)
        assert result.ne2.shape == (n_points,)
        assert result.nea.shape == (n_points,)
        assert result.negc.shape == (n_points,)
        assert result.nelism.shape == (n_points,)
        assert result.necN.shape == (n_points,)
        assert result.nevN.shape == (n_points,)

        # All densities should be non-negative and finite
        total_ne = result.total_density(ne2001_data)
        assert torch.all(total_ne >= 0)
        assert torch.all(torch.isfinite(total_ne))

    def test_density_2d_grid(self, ne2001_data):
        """Test 2D grid of points"""
        x_grid = torch.linspace(-5, 5, 5)
        y_grid = torch.linspace(5, 12, 5)

        # Create 2D grid
        x = x_grid.unsqueeze(1).expand(5, 5)
        y = y_grid.unsqueeze(0).expand(5, 5)
        z = torch.zeros(5, 5)

        result = density_2001(x, y, z, ne2001_data)

        # All results should have shape (5, 5)
        assert result.ne1.shape == (5, 5)
        assert result.total_density(ne2001_data).shape == (5, 5)

        # All densities should be non-negative
        assert torch.all(result.total_density(ne2001_data) >= 0)

    def test_density_far_from_galaxy(self, ne2001_data):
        """Test density far from Galaxy"""
        x = torch.tensor(50.0)  # 50 kpc away
        y = torch.tensor(50.0)
        z = torch.tensor(10.0)

        result = density_2001(x, y, z, ne2001_data)

        # Far from galaxy, expect very low or zero density
        total_ne = result.total_density(ne2001_data)
        assert total_ne < 0.01  # Should be nearly zero
        assert total_ne >= 0

    def test_result_as_dict(self, ne2001_data):
        """Test DensityResult.as_dict() method"""
        x = torch.tensor(0.0)
        y = torch.tensor(8.5)
        z = torch.tensor(0.0)

        result = density_2001(x, y, z, ne2001_data)
        result_dict = result.as_dict()

        # Should have all expected keys
        expected_keys = [
            'ne1', 'ne2', 'nea', 'negc', 'nelism', 'necN', 'nevN',
            'F1', 'F2', 'Fa', 'Fgc', 'Flism', 'FcN', 'FvN',
            'whicharm', 'wlism', 'wLDR', 'wLHB', 'wLSB', 'wLOOPI',
            'hitclump', 'hitvoid', 'wvoid'
        ]
        for key in expected_keys:
            assert key in result_dict
            assert torch.is_tensor(result_dict[key])

    def test_density_differentiable(self, ne2001_data):
        """Test that density is differentiable with respect to coordinates"""
        x = torch.tensor(1.0, requires_grad=True)
        y = torch.tensor(8.5, requires_grad=True)
        z = torch.tensor(0.1, requires_grad=True)

        result = density_2001(x, y, z, ne2001_data)
        total_ne = result.total_density(ne2001_data)

        # Compute gradient
        total_ne.backward()

        # Gradients should exist and be finite
        assert x.grad is not None
        assert y.grad is not None
        assert z.grad is not None
        assert torch.isfinite(x.grad)
        assert torch.isfinite(y.grad)
        assert torch.isfinite(z.grad)

    def test_component_flags(self, ne2001_data):
        """Test that component flags are correctly set"""
        x = torch.tensor(0.0)
        y = torch.tensor(8.5)
        z = torch.tensor(0.0)

        result = density_2001(x, y, z, ne2001_data)

        # Flag types should be long (integer)
        assert result.whicharm.dtype == torch.long
        assert result.wlism.dtype == torch.long
        assert result.hitclump.dtype == torch.long
        assert result.hitvoid.dtype == torch.long

        # Flags should be in valid ranges
        assert 0 <= result.whicharm <= 5
        assert result.wlism in [0, 1]

    def test_fluctuation_parameters(self, ne2001_data):
        """Test that fluctuation parameters are returned"""
        x = torch.tensor(1.0)
        y = torch.tensor(8.5)
        z = torch.tensor(0.1)

        result = density_2001(x, y, z, ne2001_data)

        # All F parameters should be finite
        assert torch.isfinite(result.F1)
        assert torch.isfinite(result.F2)
        assert torch.isfinite(result.Fa)
        assert torch.isfinite(result.Fgc)
        assert torch.isfinite(result.Flism)
        assert torch.isfinite(result.FcN)
        assert torch.isfinite(result.FvN)

        # F parameters should be non-negative (used for scattering)
        assert result.F1 >= 0
        assert result.F2 >= 0

    def test_density_symmetry_z(self, ne2001_data):
        """Test z-axis symmetry for components that should be symmetric"""
        x = torch.tensor(1.0)
        y = torch.tensor(8.5)
        z_pos = torch.tensor(0.1)
        z_neg = torch.tensor(-0.1)

        result_pos = density_2001(x, y, z_pos, ne2001_data)
        result_neg = density_2001(x, y, z_neg, ne2001_data)

        # Thick disk (ne1) should be symmetric in z
        assert torch.allclose(result_pos.ne1, result_neg.ne1, atol=1e-10)

        # Thin disk (ne2) should be symmetric in z
        assert torch.allclose(result_pos.ne2, result_neg.ne2, atol=1e-10)

    def test_broadcast_compatibility(self, ne2001_data):
        """Test that inputs can be broadcast together"""
        # Scalar x, batched y, z
        x = torch.tensor(1.0)
        y = torch.tensor([8.0, 8.5, 9.0])
        z = torch.tensor([0.0, 0.1, 0.2])

        result = density_2001(x, y, z, ne2001_data)

        # Result should have shape (3,)
        assert result.ne1.shape == (3,)
        assert result.total_density(ne2001_data).shape == (3,)

    def test_zero_weights_all_components(self):
        """Test that setting all weights to zero gives zero density"""
        # Create data with all weights set to 0
        data = NE2001Data(device='cpu')
        data.wg1 = torch.tensor(0.0)
        data.wg2 = torch.tensor(0.0)
        data.wga = torch.tensor(0.0)
        data.wggc = torch.tensor(0.0)
        data.wglism = torch.tensor(0.0)
        data.wgcN = torch.tensor(0.0)
        data.wgvN = torch.tensor(0.0)

        x = torch.tensor(1.0)
        y = torch.tensor(8.5)
        z = torch.tensor(0.1)

        result = density_2001(x, y, z, data)

        # All components should be zero
        assert result.ne1 == 0
        assert result.ne2 == 0
        assert result.nea == 0
        assert result.negc == 0
        assert result.nelism == 0
        assert result.necN == 0
        assert result.nevN == 0
        assert result.total_density(data) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
