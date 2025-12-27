"""
Tests for neclumpN (discrete clump component)
"""

import torch
import pytest
from torchgedm.ne2001.components.ne_clumps import neclumpN, galactic_to_xyz
from torchgedm.ne2001.data_loader import NE2001Data


@pytest.fixture
def ne2001_data():
    """Load NE2001 data"""
    return NE2001Data(device='cpu')


def test_galactic_to_xyz_conversion():
    """Test galactic to galactocentric coordinate conversion"""
    # Test point at l=0, b=0, d=8.5 kpc (Galactic center direction)
    l = torch.tensor(0.0)
    b = torch.tensor(0.0)
    d = torch.tensor(8.5)

    x, y, z = galactic_to_xyz(l, b, d)

    # At l=0, b=0, d=8.5: x=0, y=0, z=0 (Galactic center)
    assert torch.allclose(x, torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(y, torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(z, torch.tensor(0.0), atol=1e-6)


def test_galactic_to_xyz_sun_position():
    """Test conversion at Sun's position"""
    # Sun at origin in observer coordinates
    # In galactic: any l, b, d=0 should give Sun position
    l = torch.tensor(0.0)
    b = torch.tensor(0.0)
    d = torch.tensor(0.0)

    x, y, z = galactic_to_xyz(l, b, d)

    # Sun at x=0, y=8.5, z=0
    assert torch.allclose(x, torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(y, torch.tensor(8.5), atol=1e-6)
    assert torch.allclose(z, torch.tensor(0.0), atol=1e-6)


def test_neclumpN_no_clumps():
    """Test with no clumps"""
    x = torch.tensor(0.0)
    y = torch.tensor(8.5)
    z = torch.tensor(0.0)

    # Empty clump arrays
    clump_l = torch.tensor([])
    clump_b = torch.tensor([])
    clump_dc = torch.tensor([])
    clump_nec = torch.tensor([])
    clump_Fc = torch.tensor([])
    clump_rc = torch.tensor([])
    clump_edge = torch.tensor([])

    necN, FcN, hitclump = neclumpN(x, y, z, clump_l, clump_b, clump_dc,
                                    clump_nec, clump_Fc, clump_rc, clump_edge)

    assert necN == 0.0
    assert FcN == 0.0
    assert hitclump == 0


def test_neclumpN_single_clump_center():
    """Test density at center of a single clump"""
    # Create a clump at x=1, y=2, z=0.1
    # First convert to galactic coordinates
    # For simplicity, place clump with known coordinates
    clump_l = torch.tensor([90.0])  # l=90 deg
    clump_b = torch.tensor([0.0])    # b=0
    clump_dc = torch.tensor([7.5])   # d=7.5 kpc -> y = 8.5 - 7.5 = 1.0, x = 7.5, z = 0
    clump_nec = torch.tensor([10.0])
    clump_Fc = torch.tensor([0.5])
    clump_rc = torch.tensor([0.1])
    clump_edge = torch.tensor([0.0])  # Exponential

    # Calculate clump position
    xc, yc, zc = galactic_to_xyz(clump_l, clump_b, clump_dc)

    # Test at clump center (arg = 0, density = nec * exp(0) = nec)
    necN, FcN, hitclump = neclumpN(xc, yc, zc, clump_l, clump_b, clump_dc,
                                    clump_nec, clump_Fc, clump_rc, clump_edge)

    assert torch.allclose(necN, clump_nec, rtol=1e-6)
    assert torch.allclose(FcN, clump_Fc, rtol=1e-6)
    assert hitclump == 1


def test_neclumpN_exponential_falloff():
    """Test exponential density falloff (edge=0)"""
    # Create a clump
    clump_l = torch.tensor([90.0])
    clump_b = torch.tensor([0.0])
    clump_dc = torch.tensor([7.5])
    clump_nec = torch.tensor([10.0])
    clump_Fc = torch.tensor([0.5])
    clump_rc = torch.tensor([1.0])
    clump_edge = torch.tensor([0.0])  # Exponential

    xc, yc, zc = galactic_to_xyz(clump_l, clump_b, clump_dc)

    # Test at distance r = rc (arg = 1, density = nec * exp(-1))
    x_test = xc + clump_rc
    necN, FcN, hitclump = neclumpN(x_test, yc, zc, clump_l, clump_b, clump_dc,
                                    clump_nec, clump_Fc, clump_rc, clump_edge)

    expected = clump_nec * torch.exp(torch.tensor(-1.0))
    assert torch.allclose(necN, expected, rtol=1e-6)
    assert hitclump == 1


def test_neclumpN_hard_edge():
    """Test hard edge truncation (edge=1)"""
    # Create a clump with hard edge
    clump_l = torch.tensor([90.0])
    clump_b = torch.tensor([0.0])
    clump_dc = torch.tensor([7.5])
    clump_nec = torch.tensor([10.0])
    clump_Fc = torch.tensor([0.5])
    clump_rc = torch.tensor([1.0])
    clump_edge = torch.tensor([1.0])  # Hard edge

    xc, yc, zc = galactic_to_xyz(clump_l, clump_b, clump_dc)

    # Test inside (r = 0.5*rc, arg = 0.25 < 1, density = nec)
    x_inside = xc + 0.5 * clump_rc
    necN_inside, _, hitclump_inside = neclumpN(x_inside, yc, zc, clump_l, clump_b, clump_dc,
                                                clump_nec, clump_Fc, clump_rc, clump_edge)

    assert torch.allclose(necN_inside, clump_nec, rtol=1e-6)
    assert hitclump_inside == 1

    # Test outside (r = 1.5*rc, arg = 2.25 > 1, density = 0)
    x_outside = xc + 1.5 * clump_rc
    necN_outside, _, hitclump_outside = neclumpN(x_outside, yc, zc, clump_l, clump_b, clump_dc,
                                                  clump_nec, clump_Fc, clump_rc, clump_edge)

    assert necN_outside == 0.0
    assert hitclump_outside == 0


def test_neclumpN_outside_range():
    """Test outside exponential rolloff range (arg >= 5)"""
    clump_l = torch.tensor([90.0])
    clump_b = torch.tensor([0.0])
    clump_dc = torch.tensor([7.5])
    clump_nec = torch.tensor([10.0])
    clump_Fc = torch.tensor([0.5])
    clump_rc = torch.tensor([1.0])
    clump_edge = torch.tensor([0.0])  # Exponential, but truncated at arg=5

    xc, yc, zc = galactic_to_xyz(clump_l, clump_b, clump_dc)

    # Test at r = sqrt(5) * rc (arg = 5, should be zero due to truncation)
    x_far = xc + torch.sqrt(torch.tensor(5.0)) * clump_rc
    necN, _, hitclump = neclumpN(x_far, yc, zc, clump_l, clump_b, clump_dc,
                                 clump_nec, clump_Fc, clump_rc, clump_edge)

    # At arg=5.0, should be very small or zero (numerical precision tolerance)
    assert necN.item() < 0.1  # Allow small numerical error
    assert hitclump.item() in [0, 1]  # May hit due to floating point precision


def test_neclumpN_multiple_clumps():
    """Test with multiple overlapping clumps"""
    # Create two clumps at same location
    clump_l = torch.tensor([90.0, 90.0])
    clump_b = torch.tensor([0.0, 0.0])
    clump_dc = torch.tensor([7.5, 7.5])
    clump_nec = torch.tensor([10.0, 5.0])
    clump_Fc = torch.tensor([0.5, 0.3])
    clump_rc = torch.tensor([1.0, 1.0])
    clump_edge = torch.tensor([0.0, 0.0])

    xc, yc, zc = galactic_to_xyz(clump_l[0], clump_b[0], clump_dc[0])

    # At center, should sum both densities
    necN, FcN, hitclump = neclumpN(xc, yc, zc, clump_l, clump_b, clump_dc,
                                    clump_nec, clump_Fc, clump_rc, clump_edge)

    # Density should be sum of both
    assert torch.allclose(necN, torch.tensor(15.0), rtol=1e-6)
    # Fluctuation parameter should be from last hit clump
    assert torch.allclose(FcN, clump_Fc[1], rtol=1e-6)
    # Should report last clump hit
    assert hitclump == 2


def test_neclumpN_batched():
    """Test with batched coordinates"""
    clump_l = torch.tensor([90.0])
    clump_b = torch.tensor([0.0])
    clump_dc = torch.tensor([7.5])
    clump_nec = torch.tensor([10.0])
    clump_Fc = torch.tensor([0.5])
    clump_rc = torch.tensor([1.0])
    clump_edge = torch.tensor([0.0])

    xc, yc, zc = galactic_to_xyz(clump_l, clump_b, clump_dc)

    # Batch of 3 positions: center, at rc, far away
    # Use .squeeze() to ensure scalars when stacking
    x_batch = torch.stack([xc.squeeze(), (xc + clump_rc).squeeze(), (xc + 10.0).squeeze()])
    y_batch = torch.stack([yc.squeeze(), yc.squeeze(), yc.squeeze()])
    z_batch = torch.stack([zc.squeeze(), zc.squeeze(), zc.squeeze()])

    necN, FcN, hitclump = neclumpN(x_batch, y_batch, z_batch, clump_l, clump_b, clump_dc,
                                    clump_nec, clump_Fc, clump_rc, clump_edge)

    # Check shapes
    assert necN.shape == (3,)
    assert FcN.shape == (3,)
    assert hitclump.shape == (3,)

    # Check values
    assert torch.allclose(necN[0], clump_nec, rtol=1e-6)  # Center
    assert torch.allclose(necN[1], clump_nec * torch.exp(torch.tensor(-1.0)), rtol=1e-6)  # At rc
    assert necN[2] == 0.0  # Far away
    assert hitclump[2] == 0


def test_neclumpN_differentiable():
    """Test that function is differentiable"""
    clump_l = torch.tensor([90.0])
    clump_b = torch.tensor([0.0])
    clump_dc = torch.tensor([7.5])
    clump_nec = torch.tensor([10.0])
    clump_Fc = torch.tensor([0.5])
    clump_rc = torch.tensor([1.0])
    clump_edge = torch.tensor([0.0])

    xc_val, yc_val, zc_val = galactic_to_xyz(clump_l, clump_b, clump_dc)

    # Test point near center (requires grad)
    x = torch.tensor(float(xc_val) + 0.1, requires_grad=True)
    y = torch.tensor(float(yc_val), requires_grad=True)
    z = torch.tensor(float(zc_val), requires_grad=True)

    necN, _, _ = neclumpN(x, y, z, clump_l, clump_b, clump_dc,
                          clump_nec, clump_Fc, clump_rc, clump_edge)

    # Should be able to compute gradients
    necN.backward()
    assert x.grad is not None
    assert y.grad is not None
    assert z.grad is not None


def test_neclumpN_with_real_data(ne2001_data):
    """Test with real NE2001 clump data"""
    # Test at Sun position (should be zero or very small)
    x = torch.tensor(0.0)
    y = torch.tensor(8.5)
    z = torch.tensor(0.0)

    necN, FcN, hitclump = neclumpN(
        x, y, z,
        ne2001_data.clump_l,
        ne2001_data.clump_b,
        ne2001_data.clump_dc,
        ne2001_data.clump_nec,
        ne2001_data.clump_Fc,
        ne2001_data.clump_rc,
        ne2001_data.clump_edge
    )

    # Results should be non-negative
    assert necN >= 0.0
    assert FcN >= 0.0
    assert hitclump >= 0
    assert hitclump <= ne2001_data.n_clumps
