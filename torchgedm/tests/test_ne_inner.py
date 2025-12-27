"""
Tests for ne_inner (inner thin disk annular component)
"""

import torch
import pytest
from torchgedm.ne2001.components.ne_inner import ne_inner
from torchgedm.ne2001.data_loader import NE2001Data


@pytest.fixture
def ne2001_params():
    """Load NE2001 parameters"""
    data = NE2001Data(device='cpu')
    return {
        'n2': data.n2,
        'h2': data.h2,
        'A2': data.A2,
        'F2': data.F2,
    }


def test_ne_inner_at_galactic_center(ne2001_params):
    """Test density at Galactic center (r=0)"""
    x = torch.tensor(0.0)
    y = torch.tensor(0.0)
    z = torch.tensor(0.0)

    ne2, F2 = ne_inner(x, y, z, **ne2001_params)

    # At r=0, far from A2, should have very low density
    # rrarg = (A2/1.8)² ≈ 9, so g2 ≈ exp(-9) ≈ 1.2e-4
    assert ne2 > 0
    assert ne2 < 0.01  # Very small density
    assert F2 == ne2001_params['F2']


def test_ne_inner_peak_at_A2(ne2001_params):
    """Test that density peaks at radius A2"""
    A2 = ne2001_params['A2']
    n2 = ne2001_params['n2']
    h2 = ne2001_params['h2']

    # At r=A2, z=0: should have peak density
    x = torch.tensor(float(A2))
    y = torch.tensor(0.0)
    z = torch.tensor(0.0)

    ne2_peak, _ = ne_inner(x, y, z, **ne2001_params)

    # At peak: g2=exp(0)=1, sech²(0)=1, so ne2 = n2
    expected_peak = float(n2)
    assert abs(ne2_peak - expected_peak) / expected_peak < 1e-6


def test_ne_inner_annular_structure(ne2001_params):
    """Test that density has annular (ring) structure"""
    A2 = ne2001_params['A2']
    z = torch.tensor(0.0)

    # Test at three radii: inside A2, at A2, outside A2
    r_inside = float(A2) - 2.0
    r_peak = float(A2)
    r_outside = float(A2) + 2.0

    ne2_inside, _ = ne_inner(torch.tensor(r_inside), torch.tensor(0.0), z, **ne2001_params)
    ne2_peak, _ = ne_inner(torch.tensor(r_peak), torch.tensor(0.0), z, **ne2001_params)
    ne2_outside, _ = ne_inner(torch.tensor(r_outside), torch.tensor(0.0), z, **ne2001_params)

    # Peak should be higher than both inside and outside
    assert ne2_peak > ne2_inside
    assert ne2_peak > ne2_outside

    # Inside and outside should be roughly symmetric (Gaussian)
    # exp(-((r-A2)/1.8)²) should give similar values for ±2 kpc
    assert abs(ne2_inside - ne2_outside) / ne2_peak < 0.1


def test_ne_inner_vertical_profile(ne2001_params):
    """Test vertical sech² profile"""
    A2 = ne2001_params['A2']
    h2 = ne2001_params['h2']

    # At peak radius, test different heights
    x = torch.tensor(float(A2))
    y = torch.tensor(0.0)

    z_values = torch.tensor([0.0, float(h2), 2.0 * float(h2)])
    ne2_values = []

    for z in z_values:
        ne2, _ = ne_inner(x, y, z, **ne2001_params)
        ne2_values.append(ne2)

    # Density should decrease with |z|
    assert ne2_values[0] > ne2_values[1]
    assert ne2_values[1] > ne2_values[2]

    # Check sech² behavior: sech²(h2/h2) = sech²(1) ≈ 0.42
    expected_ratio = 0.42  # sech²(1)
    actual_ratio = ne2_values[1] / ne2_values[0]
    assert abs(actual_ratio - expected_ratio) < 0.01


def test_ne_inner_zero_beyond_cutoff(ne2001_params):
    """Test that density is zero beyond rrarg=10 cutoff"""
    A2 = ne2001_params['A2']

    # At rrarg=10: (r-A2)² = 10 * 1.8² = 32.4, so r = A2 ± 5.69 kpc
    r_cutoff = float(A2) + 6.0  # Beyond cutoff
    x = torch.tensor(r_cutoff)
    y = torch.tensor(0.0)
    z = torch.tensor(0.0)

    ne2, _ = ne_inner(x, y, z, **ne2001_params)

    # Should be effectively zero (or very small)
    assert ne2 < 1e-4


def test_ne_inner_batched(ne2001_params):
    """Test batched computation"""
    # Create batch of 100 random points
    batch_size = 100
    x = torch.randn(batch_size) * 5
    y = torch.randn(batch_size) * 5
    z = torch.randn(batch_size) * 0.5

    ne2, F2 = ne_inner(x, y, z, **ne2001_params)

    assert ne2.shape == (batch_size,)
    assert torch.all(ne2 >= 0)  # Density should be non-negative
    assert F2 == ne2001_params['F2']


def test_ne_inner_differentiable(ne2001_params):
    """Test that ne_inner is differentiable"""
    x = torch.tensor(5.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)
    z = torch.tensor(0.2, requires_grad=True)

    ne2, _ = ne_inner(x, y, z, **ne2001_params)

    # Compute gradients
    ne2.backward()

    assert x.grad is not None
    assert y.grad is not None
    assert z.grad is not None
    assert torch.all(torch.isfinite(x.grad))
    assert torch.all(torch.isfinite(y.grad))
    assert torch.all(torch.isfinite(z.grad))


def test_ne_inner_z_symmetry(ne2001_params):
    """Test that density is symmetric in z (sech² is even function)"""
    A2 = ne2001_params['A2']
    x = torch.tensor(float(A2))
    y = torch.tensor(0.0)
    z_pos = torch.tensor(0.5)
    z_neg = torch.tensor(-0.5)

    ne2_pos, _ = ne_inner(x, y, z_pos, **ne2001_params)
    ne2_neg, _ = ne_inner(x, y, z_neg, **ne2001_params)

    assert torch.allclose(ne2_pos, ne2_neg, rtol=1e-6)


def test_ne_inner_azimuthal_symmetry(ne2001_params):
    """Test azimuthal symmetry (should only depend on r, not angle)"""
    r = 5.0
    z = torch.tensor(0.0)

    # Test at 4 different angles
    angles = torch.tensor([0.0, torch.pi / 4, torch.pi / 2, 3 * torch.pi / 4])
    densities = []

    for angle in angles:
        x = r * torch.cos(angle)
        y = r * torch.sin(angle)
        ne2, _ = ne_inner(x, y, z, **ne2001_params)
        densities.append(ne2)

    # All densities should be equal (azimuthal symmetry)
    for i in range(1, len(densities)):
        assert torch.allclose(densities[0], densities[i], rtol=1e-6)


def test_ne_inner_gaussian_falloff(ne2001_params):
    """Test Gaussian radial falloff from peak"""
    A2 = ne2001_params['A2']
    z = torch.tensor(0.0)

    # At r = A2 + 1.8 kpc: rrarg = 1, g2 = exp(-1) ≈ 0.368
    r1 = float(A2) + 1.8
    x1 = torch.tensor(r1)
    y1 = torch.tensor(0.0)

    # At r = A2 + 3.6 kpc: rrarg = 4, g2 = exp(-4) ≈ 0.0183
    r2 = float(A2) + 3.6
    x2 = torch.tensor(r2)
    y2 = torch.tensor(0.0)

    ne2_1, _ = ne_inner(x1, y1, z, **ne2001_params)
    ne2_2, _ = ne_inner(x2, y2, z, **ne2001_params)

    # Ratio should be exp(3) ≈ 20.09
    expected_ratio = torch.exp(torch.tensor(3.0))
    actual_ratio = ne2_1 / ne2_2
    assert abs(actual_ratio - expected_ratio) / expected_ratio < 0.01


def test_ne_inner_multidimensional_batch(ne2001_params):
    """Test with multidimensional batching"""
    # Create 10x20 grid
    x = torch.randn(10, 20) * 5
    y = torch.randn(10, 20) * 5
    z = torch.randn(10, 20) * 0.5

    ne2, F2 = ne_inner(x, y, z, **ne2001_params)

    assert ne2.shape == (10, 20)
    assert torch.all(ne2 >= 0)
    assert F2 == ne2001_params['F2']


def test_ne_inner_parameter_broadcasting(ne2001_params):
    """Test that parameters can be broadcast"""
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([0.0, 0.0, 0.0])
    z = torch.tensor([0.0, 0.0, 0.0])

    # Parameters are scalars, should broadcast to match input shape
    ne2, F2 = ne_inner(x, y, z, **ne2001_params)

    assert ne2.shape == (3,)
    assert torch.all(ne2 >= 0)


def test_ne_inner_without_F2(ne2001_params):
    """Test that F2 is optional"""
    x = torch.tensor(5.0)
    y = torch.tensor(3.0)
    z = torch.tensor(0.2)

    # Call without F2
    params_no_F2 = {k: v for k, v in ne2001_params.items() if k != 'F2'}
    ne2 = ne_inner(x, y, z, **params_no_F2)

    # Should return only ne2, not tuple
    assert isinstance(ne2, torch.Tensor)
    assert ne2.shape == ()


def test_ne_inner_scale_length(ne2001_params):
    """Test hardcoded scale length of 1.8 kpc"""
    A2 = ne2001_params['A2']
    z = torch.tensor(0.0)

    # At r = A2, density should be n2 (peak)
    ne2_peak, _ = ne_inner(torch.tensor(float(A2)), torch.tensor(0.0), z, **ne2001_params)

    # At r = A2 + 1.8, density should be n2 * exp(-1)
    r_scale = float(A2) + 1.8
    ne2_scale, _ = ne_inner(torch.tensor(r_scale), torch.tensor(0.0), z, **ne2001_params)

    expected_ratio = torch.exp(torch.tensor(-1.0))
    actual_ratio = ne2_scale / ne2_peak
    assert abs(actual_ratio - expected_ratio) / expected_ratio < 1e-6


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
