"""
Tests for nevoidN (discrete void component)
"""

import torch
import pytest
from torchgedm.ne2001.components.ne_voids import nevoidN, galactic_to_xyz
from torchgedm.ne2001.data_loader import NE2001Data


@pytest.fixture
def ne2001_data():
    """Load NE2001 data"""
    return NE2001Data(device='cpu')


def test_nevoidN_no_voids():
    """Test with no voids"""
    x = torch.tensor(0.0)
    y = torch.tensor(8.5)
    z = torch.tensor(0.0)

    # Empty void arrays
    void_l = torch.tensor([])
    void_b = torch.tensor([])
    void_dv = torch.tensor([])
    void_nev = torch.tensor([])
    void_Fv = torch.tensor([])
    void_aav = torch.tensor([])
    void_bbv = torch.tensor([])
    void_ccv = torch.tensor([])
    void_thvy = torch.tensor([])
    void_thvz = torch.tensor([])
    void_edge = torch.tensor([])

    nevN, FvN, hitvoid, wvoid = nevoidN(
        x, y, z, void_l, void_b, void_dv, void_nev, void_Fv,
        void_aav, void_bbv, void_ccv, void_thvy, void_thvz, void_edge
    )

    assert nevN == 0.0
    assert FvN == 0.0
    assert hitvoid == 0
    assert wvoid == 0


def test_nevoidN_single_void_center():
    """Test density at center of a single void (no rotation)"""
    # Create a spherical void at x=7.5, y=1.0, z=0 (l=90, b=0, d=7.5)
    void_l = torch.tensor([90.0])
    void_b = torch.tensor([0.0])
    void_dv = torch.tensor([7.5])
    void_nev = torch.tensor([5.0])
    void_Fv = torch.tensor([0.3])
    void_aav = torch.tensor([1.0])  # Spherical
    void_bbv = torch.tensor([1.0])
    void_ccv = torch.tensor([1.0])
    void_thvy = torch.tensor([0.0])  # No rotation
    void_thvz = torch.tensor([0.0])
    void_edge = torch.tensor([0.0])  # Exponential

    xv, yv, zv = galactic_to_xyz(void_l, void_b, void_dv)

    # Test at void center (q = 0, density = nev * exp(0) = nev)
    nevN, FvN, hitvoid, wvoid = nevoidN(
        xv, yv, zv, void_l, void_b, void_dv, void_nev, void_Fv,
        void_aav, void_bbv, void_ccv, void_thvy, void_thvz, void_edge
    )

    assert torch.allclose(nevN, void_nev, rtol=1e-6)
    assert torch.allclose(FvN, void_Fv, rtol=1e-6)
    assert hitvoid == 1
    assert wvoid == 1


def test_nevoidN_exponential_falloff():
    """Test exponential density falloff (edge=0)"""
    void_l = torch.tensor([90.0])
    void_b = torch.tensor([0.0])
    void_dv = torch.tensor([7.5])
    void_nev = torch.tensor([5.0])
    void_Fv = torch.tensor([0.3])
    void_aav = torch.tensor([1.0])
    void_bbv = torch.tensor([1.0])
    void_ccv = torch.tensor([1.0])
    void_thvy = torch.tensor([0.0])
    void_thvz = torch.tensor([0.0])
    void_edge = torch.tensor([0.0])

    xv, yv, zv = galactic_to_xyz(void_l, void_b, void_dv)

    # Test at distance r = aav (q = 1, density = nev * exp(-1))
    x_test = xv + void_aav
    nevN, _, hitvoid, _ = nevoidN(
        x_test, yv, zv, void_l, void_b, void_dv, void_nev, void_Fv,
        void_aav, void_bbv, void_ccv, void_thvy, void_thvz, void_edge
    )

    expected = void_nev * torch.exp(torch.tensor(-1.0))
    assert torch.allclose(nevN, expected, rtol=1e-6)
    assert hitvoid == 1


def test_nevoidN_hard_edge():
    """Test hard edge truncation (edge=1)"""
    void_l = torch.tensor([90.0])
    void_b = torch.tensor([0.0])
    void_dv = torch.tensor([7.5])
    void_nev = torch.tensor([5.0])
    void_Fv = torch.tensor([0.3])
    void_aav = torch.tensor([1.0])
    void_bbv = torch.tensor([1.0])
    void_ccv = torch.tensor([1.0])
    void_thvy = torch.tensor([0.0])
    void_thvz = torch.tensor([0.0])
    void_edge = torch.tensor([1.0])  # Hard edge

    xv, yv, zv = galactic_to_xyz(void_l, void_b, void_dv)

    # Test inside (r = 0.5*aav, q = 0.25 < 1, density = nev)
    x_inside = xv + 0.5 * void_aav
    nevN_inside, _, hitvoid_inside, _ = nevoidN(
        x_inside, yv, zv, void_l, void_b, void_dv, void_nev, void_Fv,
        void_aav, void_bbv, void_ccv, void_thvy, void_thvz, void_edge
    )

    assert torch.allclose(nevN_inside, void_nev, rtol=1e-6)
    assert hitvoid_inside == 1

    # Test outside (r = 1.5*aav, q = 2.25 > 1, density = 0)
    x_outside = xv + 1.5 * void_aav
    nevN_outside, _, hitvoid_outside, wvoid_outside = nevoidN(
        x_outside, yv, zv, void_l, void_b, void_dv, void_nev, void_Fv,
        void_aav, void_bbv, void_ccv, void_thvy, void_thvz, void_edge
    )

    assert nevN_outside == 0.0
    assert hitvoid_outside == 0
    assert wvoid_outside == 0


def test_nevoidN_outside_range():
    """Test outside exponential rolloff range (q >= 3)"""
    void_l = torch.tensor([90.0])
    void_b = torch.tensor([0.0])
    void_dv = torch.tensor([7.5])
    void_nev = torch.tensor([5.0])
    void_Fv = torch.tensor([0.3])
    void_aav = torch.tensor([1.0])
    void_bbv = torch.tensor([1.0])
    void_ccv = torch.tensor([1.0])
    void_thvy = torch.tensor([0.0])
    void_thvz = torch.tensor([0.0])
    void_edge = torch.tensor([0.0])

    xv, yv, zv = galactic_to_xyz(void_l, void_b, void_dv)

    # Test at r = sqrt(3) * aav (q = 3, should be zero due to truncation)
    x_far = xv + torch.sqrt(torch.tensor(3.0)) * void_aav
    nevN, _, hitvoid, wvoid = nevoidN(
        x_far, yv, zv, void_l, void_b, void_dv, void_nev, void_Fv,
        void_aav, void_bbv, void_ccv, void_thvy, void_thvz, void_edge
    )

    assert nevN == 0.0
    assert hitvoid == 0
    assert wvoid == 0


def test_nevoidN_ellipsoidal():
    """Test ellipsoidal void (no rotation)"""
    void_l = torch.tensor([90.0])
    void_b = torch.tensor([0.0])
    void_dv = torch.tensor([7.5])
    void_nev = torch.tensor([5.0])
    void_Fv = torch.tensor([0.3])
    void_aav = torch.tensor([2.0])  # Major axis
    void_bbv = torch.tensor([1.0])  # Minor axis
    void_ccv = torch.tensor([1.0])  # Minor axis
    void_thvy = torch.tensor([0.0])
    void_thvz = torch.tensor([0.0])
    void_edge = torch.tensor([1.0])  # Hard edge

    xv, yv, zv = galactic_to_xyz(void_l, void_b, void_dv)

    # Test at (aav, 0, 0) from center: q = 1, should be inside
    x_test = xv + void_aav
    nevN_major, _, hitvoid_major, _ = nevoidN(
        x_test, yv, zv, void_l, void_b, void_dv, void_nev, void_Fv,
        void_aav, void_bbv, void_ccv, void_thvy, void_thvz, void_edge
    )

    # On boundary: q = (2/2)² = 1, should be inside (<=)
    assert torch.allclose(nevN_major, void_nev, rtol=1e-6)
    assert hitvoid_major == 1

    # Test at (0, bbv, 0): q = 1, should be inside
    y_test = yv + void_bbv
    nevN_minor, _, hitvoid_minor, _ = nevoidN(
        xv, y_test, zv, void_l, void_b, void_dv, void_nev, void_Fv,
        void_aav, void_bbv, void_ccv, void_thvy, void_thvz, void_edge
    )

    assert torch.allclose(nevN_minor, void_nev, rtol=1e-6)
    assert hitvoid_minor == 1


def test_nevoidN_rotation_y_axis():
    """Test rotation around y-axis"""
    void_l = torch.tensor([90.0])
    void_b = torch.tensor([0.0])
    void_dv = torch.tensor([7.5])
    void_nev = torch.tensor([5.0])
    void_Fv = torch.tensor([0.3])
    void_aav = torch.tensor([2.0])
    void_bbv = torch.tensor([1.0])
    void_ccv = torch.tensor([1.0])
    void_thvy = torch.tensor([90.0])  # 90 degree rotation around y
    void_thvz = torch.tensor([0.0])
    void_edge = torch.tensor([1.0])

    xv, yv, zv = galactic_to_xyz(void_l, void_b, void_dv)

    # After 90° rotation around y: x->z, z->-x
    # Major axis now along z instead of x
    # Point at (0, 0, aav) should be inside after rotation
    z_test = zv + void_aav
    nevN, _, hitvoid, _ = nevoidN(
        xv, yv, z_test, void_l, void_b, void_dv, void_nev, void_Fv,
        void_aav, void_bbv, void_ccv, void_thvy, void_thvz, void_edge
    )

    # Should be on boundary (inside with hard edge)
    assert torch.allclose(nevN, void_nev, rtol=1e-5)
    assert hitvoid == 1


def test_nevoidN_batched():
    """Test with batched coordinates"""
    void_l = torch.tensor([90.0])
    void_b = torch.tensor([0.0])
    void_dv = torch.tensor([7.5])
    void_nev = torch.tensor([5.0])
    void_Fv = torch.tensor([0.3])
    void_aav = torch.tensor([1.0])
    void_bbv = torch.tensor([1.0])
    void_ccv = torch.tensor([1.0])
    void_thvy = torch.tensor([0.0])
    void_thvz = torch.tensor([0.0])
    void_edge = torch.tensor([0.0])

    xv, yv, zv = galactic_to_xyz(void_l, void_b, void_dv)

    # Batch of 3 positions: center, at aav, far away
    # Use .squeeze() to ensure scalars when stacking
    x_batch = torch.stack([xv.squeeze(), (xv + void_aav).squeeze(), (xv + 10.0).squeeze()])
    y_batch = torch.stack([yv.squeeze(), yv.squeeze(), yv.squeeze()])
    z_batch = torch.stack([zv.squeeze(), zv.squeeze(), zv.squeeze()])

    nevN, FvN, hitvoid, wvoid = nevoidN(
        x_batch, y_batch, z_batch, void_l, void_b, void_dv, void_nev, void_Fv,
        void_aav, void_bbv, void_ccv, void_thvy, void_thvz, void_edge
    )

    # Check shapes
    assert nevN.shape == (3,)
    assert FvN.shape == (3,)
    assert hitvoid.shape == (3,)
    assert wvoid.shape == (3,)

    # Check values
    assert torch.allclose(nevN[0], void_nev, rtol=1e-6)  # Center
    assert torch.allclose(nevN[1], void_nev * torch.exp(torch.tensor(-1.0)), rtol=1e-6)  # At aav
    assert nevN[2] == 0.0  # Far away
    assert hitvoid[2] == 0
    assert wvoid[2] == 0


def test_nevoidN_only_one_void_active():
    """Test that only one void contributes (unlike clumps which sum)"""
    # Create two overlapping voids at same location
    void_l = torch.tensor([90.0, 90.0])
    void_b = torch.tensor([0.0, 0.0])
    void_dv = torch.tensor([7.5, 7.5])
    void_nev = torch.tensor([5.0, 3.0])
    void_Fv = torch.tensor([0.3, 0.2])
    void_aav = torch.tensor([1.0, 1.0])
    void_bbv = torch.tensor([1.0, 1.0])
    void_ccv = torch.tensor([1.0, 1.0])
    void_thvy = torch.tensor([0.0, 0.0])
    void_thvz = torch.tensor([0.0, 0.0])
    void_edge = torch.tensor([0.0, 0.0])

    xv, yv, zv = galactic_to_xyz(void_l[0], void_b[0], void_dv[0])

    # At center, should only get density from one void (not sum)
    nevN, FvN, hitvoid, wvoid = nevoidN(
        xv, yv, zv, void_l, void_b, void_dv, void_nev, void_Fv,
        void_aav, void_bbv, void_ccv, void_thvy, void_thvz, void_edge
    )

    # Should get density from last void only (not sum)
    assert torch.allclose(nevN, void_nev[1], rtol=1e-6)
    assert torch.allclose(FvN, void_Fv[1], rtol=1e-6)
    assert hitvoid == 2
    assert wvoid == 1


def test_nevoidN_differentiable():
    """Test that function is differentiable"""
    void_l = torch.tensor([90.0])
    void_b = torch.tensor([0.0])
    void_dv = torch.tensor([7.5])
    void_nev = torch.tensor([5.0])
    void_Fv = torch.tensor([0.3])
    void_aav = torch.tensor([1.0])
    void_bbv = torch.tensor([1.0])
    void_ccv = torch.tensor([1.0])
    void_thvy = torch.tensor([0.0])
    void_thvz = torch.tensor([0.0])
    void_edge = torch.tensor([0.0])

    xv_val, yv_val, zv_val = galactic_to_xyz(void_l, void_b, void_dv)

    # Test point near center (requires grad)
    x = torch.tensor(float(xv_val) + 0.1, requires_grad=True)
    y = torch.tensor(float(yv_val), requires_grad=True)
    z = torch.tensor(float(zv_val), requires_grad=True)

    nevN, _, _, _ = nevoidN(
        x, y, z, void_l, void_b, void_dv, void_nev, void_Fv,
        void_aav, void_bbv, void_ccv, void_thvy, void_thvz, void_edge
    )

    # Should be able to compute gradients
    nevN.backward()
    assert x.grad is not None
    assert y.grad is not None
    assert z.grad is not None


def test_nevoidN_with_real_data(ne2001_data):
    """Test with real NE2001 void data"""
    # Test at Sun position
    x = torch.tensor(0.0)
    y = torch.tensor(8.5)
    z = torch.tensor(0.0)

    nevN, FvN, hitvoid, wvoid = nevoidN(
        x, y, z,
        ne2001_data.void_l,
        ne2001_data.void_b,
        ne2001_data.void_dv,
        ne2001_data.void_nev,
        ne2001_data.void_Fv,
        ne2001_data.void_aav,
        ne2001_data.void_bbv,
        ne2001_data.void_ccv,
        ne2001_data.void_thvy,
        ne2001_data.void_thvz,
        ne2001_data.void_edge
    )

    # Results should be non-negative
    assert nevN >= 0.0
    assert FvN >= 0.0
    assert hitvoid >= 0
    assert hitvoid <= ne2001_data.n_voids
    assert wvoid >= 0
    assert wvoid <= 1
