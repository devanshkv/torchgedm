"""
Tests for ne_gc (Galactic center component)
"""

import torch
import pytest
from torchgedm.ne2001.components.ne_gc import ne_gc
from torchgedm.ne2001.data_loader import NE2001Data


@pytest.fixture
def ne2001_params():
    """Load NE2001 parameters"""
    data = NE2001Data(device='cpu')
    return {
        'xgc': data.xgc,
        'ygc': data.ygc,
        'zgc': data.zgc,
        'rgc': data.rgc,
        'hgc': data.hgc,
        'negc0': data.negc0,
        'Fgc0': data.Fgc0,
    }


def test_ne_gc_at_gc_center(ne2001_params):
    """Test density at Galactic center"""
    xgc = float(ne2001_params['xgc'])
    ygc = float(ne2001_params['ygc'])
    zgc = float(ne2001_params['zgc'])
    negc0 = float(ne2001_params['negc0'])

    x = torch.tensor(xgc)
    y = torch.tensor(ygc)
    z = torch.tensor(zgc)

    negc, Fgc = ne_gc(x, y, z, **ne2001_params)

    # At center of GC ellipsoid, should have full density
    assert torch.allclose(negc, ne2001_params['negc0'], rtol=1e-6)
    assert Fgc == ne2001_params['Fgc0']


def test_ne_gc_inside_ellipsoid(ne2001_params):
    """Test density inside the ellipsoid"""
    xgc = float(ne2001_params['xgc'])
    ygc = float(ne2001_params['ygc'])
    zgc = float(ne2001_params['zgc'])
    rgc = float(ne2001_params['rgc'])
    hgc = float(ne2001_params['hgc'])
    negc0 = float(ne2001_params['negc0'])

    # Point at half radius, half height: (0.5*rgc)² + (0.5*hgc)² = 0.5 < 1
    x = torch.tensor(xgc + 0.5 * rgc)
    y = torch.tensor(ygc)
    z = torch.tensor(zgc + 0.5 * hgc)

    negc, _ = ne_gc(x, y, z, **ne2001_params)

    # Should have full density (constant inside)
    assert torch.allclose(negc, ne2001_params['negc0'], rtol=1e-6)


def test_ne_gc_outside_radial_extent(ne2001_params):
    """Test density outside radial extent"""
    xgc = float(ne2001_params['xgc'])
    ygc = float(ne2001_params['ygc'])
    zgc = float(ne2001_params['zgc'])
    rgc = float(ne2001_params['rgc'])

    # Point beyond radial extent
    x = torch.tensor(xgc + 2.0 * rgc)
    y = torch.tensor(ygc)
    z = torch.tensor(zgc)

    negc, _ = ne_gc(x, y, z, **ne2001_params)

    # Should be zero
    assert negc == 0.0


def test_ne_gc_outside_vertical_extent(ne2001_params):
    """Test density outside vertical extent"""
    xgc = float(ne2001_params['xgc'])
    ygc = float(ne2001_params['ygc'])
    zgc = float(ne2001_params['zgc'])
    hgc = float(ne2001_params['hgc'])

    # Point beyond vertical extent
    x = torch.tensor(xgc)
    y = torch.tensor(ygc)
    z = torch.tensor(zgc + 2.0 * hgc)

    negc, _ = ne_gc(x, y, z, **ne2001_params)

    # Should be zero
    assert negc == 0.0


def test_ne_gc_on_ellipsoid_boundary(ne2001_params):
    """Test density on ellipsoid boundary"""
    xgc = float(ne2001_params['xgc'])
    ygc = float(ne2001_params['ygc'])
    zgc = float(ne2001_params['zgc'])
    rgc = float(ne2001_params['rgc'])
    hgc = float(ne2001_params['hgc'])

    # Point on boundary: (rr/rgc)² + (zz/hgc)² = 1
    # Use rr = rgc * cos(θ), zz = hgc * sin(θ) for θ = π/4
    import math
    theta = math.pi / 4
    rr = rgc * math.cos(theta)
    zz = hgc * math.sin(theta)

    x = torch.tensor(xgc + rr)
    y = torch.tensor(ygc)
    z = torch.tensor(zgc + zz)

    negc, _ = ne_gc(x, y, z, **ne2001_params)

    # Should have full density (arg = 1, condition is <=)
    assert torch.allclose(negc, ne2001_params['negc0'], rtol=1e-6)


def test_ne_gc_just_outside_ellipsoid(ne2001_params):
    """Test density just outside ellipsoid"""
    xgc = float(ne2001_params['xgc'])
    ygc = float(ne2001_params['ygc'])
    zgc = float(ne2001_params['zgc'])
    rgc = float(ne2001_params['rgc'])
    hgc = float(ne2001_params['hgc'])

    # Point just outside: (rr/rgc)² + (zz/hgc)² = 1.01
    import math
    theta = math.pi / 4
    rr = rgc * math.cos(theta) * 1.01
    zz = hgc * math.sin(theta) * 1.01

    x = torch.tensor(xgc + rr)
    y = torch.tensor(ygc)
    z = torch.tensor(zgc + zz)

    negc, _ = ne_gc(x, y, z, **ne2001_params)

    # Should be zero
    assert negc == 0.0


def test_ne_gc_azimuthal_symmetry(ne2001_params):
    """Test azimuthal symmetry around GC"""
    xgc = float(ne2001_params['xgc'])
    ygc = float(ne2001_params['ygc'])
    zgc = float(ne2001_params['zgc'])
    rgc = float(ne2001_params['rgc'])

    r = 0.5 * rgc
    z = torch.tensor(zgc)

    # Test at 4 different angles
    angles = torch.tensor([0.0, torch.pi / 4, torch.pi / 2, 3 * torch.pi / 4])
    densities = []

    for angle in angles:
        x = xgc + r * torch.cos(angle)
        y = ygc + r * torch.sin(angle)
        negc, _ = ne_gc(x, y, z, **ne2001_params)
        densities.append(negc)

    # All densities should be equal (azimuthal symmetry)
    for i in range(1, len(densities)):
        assert torch.allclose(densities[0], densities[i], rtol=1e-6)


def test_ne_gc_z_symmetry(ne2001_params):
    """Test that density is symmetric in z"""
    xgc = float(ne2001_params['xgc'])
    ygc = float(ne2001_params['ygc'])
    zgc = float(ne2001_params['zgc'])
    hgc = float(ne2001_params['hgc'])

    x = torch.tensor(xgc)
    y = torch.tensor(ygc)
    z_offset = 0.5 * hgc

    z_pos = torch.tensor(zgc + z_offset)
    z_neg = torch.tensor(zgc - z_offset)

    negc_pos, _ = ne_gc(x, y, z_pos, **ne2001_params)
    negc_neg, _ = ne_gc(x, y, z_neg, **ne2001_params)

    assert torch.allclose(negc_pos, negc_neg, rtol=1e-6)


def test_ne_gc_batched(ne2001_params):
    """Test batched computation"""
    xgc = float(ne2001_params['xgc'])
    ygc = float(ne2001_params['ygc'])
    zgc = float(ne2001_params['zgc'])
    rgc = float(ne2001_params['rgc'])
    hgc = float(ne2001_params['hgc'])

    # Create batch of 100 points around GC
    batch_size = 100
    x = xgc + torch.randn(batch_size) * rgc * 0.5
    y = ygc + torch.randn(batch_size) * rgc * 0.5
    z = zgc + torch.randn(batch_size) * hgc * 0.5

    negc, Fgc = ne_gc(x, y, z, **ne2001_params)

    assert negc.shape == (batch_size,)
    assert torch.all(negc >= 0)  # Density should be non-negative
    assert Fgc == ne2001_params['Fgc0']


def test_ne_gc_differentiable(ne2001_params):
    """Test that ne_gc is differentiable"""
    xgc = float(ne2001_params['xgc'])
    ygc = float(ne2001_params['ygc'])
    zgc = float(ne2001_params['zgc'])

    x = torch.tensor(xgc + 0.01, requires_grad=True)
    y = torch.tensor(ygc + 0.01, requires_grad=True)
    z = torch.tensor(zgc + 0.01, requires_grad=True)

    negc, _ = ne_gc(x, y, z, **ne2001_params)

    # Compute gradients
    negc.backward()

    assert x.grad is not None
    assert y.grad is not None
    assert z.grad is not None
    assert torch.all(torch.isfinite(x.grad))
    assert torch.all(torch.isfinite(y.grad))
    assert torch.all(torch.isfinite(z.grad))


def test_ne_gc_multidimensional_batch(ne2001_params):
    """Test with multidimensional batching"""
    xgc = float(ne2001_params['xgc'])
    ygc = float(ne2001_params['ygc'])
    zgc = float(ne2001_params['zgc'])
    rgc = float(ne2001_params['rgc'])
    hgc = float(ne2001_params['hgc'])

    # Create 10x20 grid
    x = xgc + torch.randn(10, 20) * rgc * 0.5
    y = ygc + torch.randn(10, 20) * rgc * 0.5
    z = zgc + torch.randn(10, 20) * hgc * 0.5

    negc, Fgc = ne_gc(x, y, z, **ne2001_params)

    assert negc.shape == (10, 20)
    assert torch.all(negc >= 0)
    assert Fgc == ne2001_params['Fgc0']


def test_ne_gc_without_Fgc0(ne2001_params):
    """Test that Fgc0 is optional"""
    xgc = float(ne2001_params['xgc'])
    ygc = float(ne2001_params['ygc'])
    zgc = float(ne2001_params['zgc'])

    x = torch.tensor(xgc)
    y = torch.tensor(ygc)
    z = torch.tensor(zgc)

    # Call without Fgc0
    params_no_Fgc0 = {k: v for k, v in ne2001_params.items() if k != 'Fgc0'}
    negc = ne_gc(x, y, z, **params_no_Fgc0)

    # Should return only negc, not tuple
    assert isinstance(negc, torch.Tensor)
    assert negc.shape == ()


def test_ne_gc_constant_inside(ne2001_params):
    """Test that density is constant everywhere inside ellipsoid"""
    xgc = float(ne2001_params['xgc'])
    ygc = float(ne2001_params['ygc'])
    zgc = float(ne2001_params['zgc'])
    rgc = float(ne2001_params['rgc'])
    hgc = float(ne2001_params['hgc'])
    negc0 = float(ne2001_params['negc0'])

    # Test 10 random points inside ellipsoid
    torch.manual_seed(42)
    for _ in range(10):
        # Generate random point inside unit sphere
        theta = torch.rand(1) * 2 * torch.pi
        phi = torch.rand(1) * torch.pi
        r_frac = torch.rand(1) ** (1/3)  # Uniform in volume

        # Map to ellipsoid coordinates (ensure arg < 1)
        rr = r_frac * rgc * 0.7  # Use 0.7 to stay well inside
        zz = r_frac * hgc * 0.7

        x = torch.tensor(xgc) + rr * torch.cos(theta)
        y = torch.tensor(ygc) + rr * torch.sin(theta)
        z = torch.tensor(zgc) + zz

        negc, _ = ne_gc(x, y, z, **ne2001_params)

        # Should always be negc0
        assert torch.allclose(negc, ne2001_params['negc0'], rtol=1e-6)


def test_ne_gc_far_from_center(ne2001_params):
    """Test density far from Galactic center"""
    # Test at Sun's position (roughly 8 kpc from GC)
    x = torch.tensor(8.0)
    y = torch.tensor(0.0)
    z = torch.tensor(0.0)

    negc, _ = ne_gc(x, y, z, **ne2001_params)

    # Should be zero (far from small GC region)
    assert negc == 0.0


def test_ne_gc_parameter_values(ne2001_params):
    """Verify loaded parameter values match expected"""
    # From ne_gc.inp file
    assert torch.allclose(ne2001_params['xgc'], torch.tensor(-0.01), atol=1e-6)
    assert torch.allclose(ne2001_params['ygc'], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(ne2001_params['zgc'], torch.tensor(-0.020), atol=1e-6)
    assert torch.allclose(ne2001_params['rgc'], torch.tensor(0.145), atol=1e-6)
    assert torch.allclose(ne2001_params['hgc'], torch.tensor(0.026), atol=1e-6)
    assert torch.allclose(ne2001_params['negc0'], torch.tensor(10.0), atol=1e-6)
    assert torch.allclose(ne2001_params['Fgc0'], torch.tensor(6.0e4), atol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
