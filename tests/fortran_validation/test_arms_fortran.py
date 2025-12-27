"""
Validation test for ne_arms (spiral arms) against FORTRAN NE2001.

This test validates the ne_arms implementation by:
1. Generating 10,000 random galactocentric (x,y,z) points
2. Comparing PyTorch vs FORTRAN outputs for:
   - nea: spiral arms electron density (cm^-3)
   - fa: spiral arms fluctuation parameter
   - whicharm: which arm dominates (0-5, 0=none)
3. Computing avg/max/rel differences
4. Asserting all differences < 1e-6

Requires: Compiled ne21c FORTRAN wrapper
"""

import numpy as np
import torch
import pytest
import sys

# Try to import ne21c (FORTRAN wrapper)
sys.path.insert(0, 'ne21c')
try:
    import ne21c
    HAS_NE21C = True
except ImportError:
    HAS_NE21C = False

from torchgedm.ne2001.data_loader import NE2001Data
from torchgedm.ne2001.components.ne_arms import ne_arms, generate_spiral_arm_paths


@pytest.fixture
def ne2001_data():
    """Load NE2001 data once for all tests."""
    return NE2001Data(device='cpu')


@pytest.fixture
def spiral_arm_paths(ne2001_data):
    """Generate spiral arm paths once for all tests."""
    data = ne2001_data
    arm_x, arm_y, arm_kmax = generate_spiral_arm_paths(
        data.arm_a,
        data.arm_rmin,
        data.arm_thmin,
        data.arm_extent,
    )
    return arm_x, arm_y, arm_kmax


@pytest.fixture
def test_points():
    """Generate random test points."""
    np.random.seed(42)
    torch.manual_seed(42)

    n_points = 10000
    x_coords = np.random.uniform(-20, 20, n_points).astype(np.float32)
    y_coords = np.random.uniform(-20, 20, n_points).astype(np.float32)
    z_coords = np.random.uniform(-5, 5, n_points).astype(np.float32)

    return x_coords, y_coords, z_coords


@pytest.mark.skipif(not HAS_NE21C, reason="ne21c FORTRAN wrapper not available")
def test_ne_arms_vs_fortran(ne2001_data, spiral_arm_paths, test_points):
    """Compare ne_arms against FORTRAN implementation."""
    data = ne2001_data
    arm_x, arm_y, arm_kmax = spiral_arm_paths
    x_coords, y_coords, z_coords = test_points
    n_points = len(x_coords)

    # Call FORTRAN ne_arms via ne21c
    nea_fortran = np.zeros(n_points, dtype=np.float32)
    fa_fortran = np.zeros(n_points, dtype=np.float32)
    whicharm_fortran = np.zeros(n_points, dtype=np.int32)

    for i in range(n_points):
        result = ne21c.density_xyz(float(x_coords[i]), float(y_coords[i]), float(z_coords[i]))
        nea_fortran[i] = result['nea']
        fa_fortran[i] = result['fa']
        whicharm_fortran[i] = int(result['whicharm'])

    # Call PyTorch ne_arms
    x_torch = torch.from_numpy(x_coords)
    y_torch = torch.from_numpy(y_coords)
    z_torch = torch.from_numpy(z_coords)

    nea_torch, fa_torch, whicharm_torch, _ = ne_arms(
        x_torch, y_torch, z_torch,
        arm_x, arm_y, arm_kmax,
        data.na, data.ha, data.wa,
        data.Aa, data.Fa,
        data.narm, data.warm,
        data.harm, data.farm,
    )

    # Compare nea (spiral arms density)
    nea_torch_np = nea_torch.cpu().numpy()
    abs_diff_nea = np.abs(nea_torch_np - nea_fortran)
    max_diff_nea = np.max(abs_diff_nea)
    avg_diff_nea = np.mean(abs_diff_nea)

    assert max_diff_nea < 1e-6, \
        f"nea max diff {max_diff_nea:.2e} exceeds tolerance 1e-6"

    # Compare fa (fluctuation parameter)
    fa_torch_np = fa_torch.cpu().numpy()
    abs_diff_fa = np.abs(fa_torch_np - fa_fortran)
    max_diff_fa = np.max(abs_diff_fa)

    assert max_diff_fa < 1e-6, \
        f"fa max diff {max_diff_fa:.2e} exceeds tolerance 1e-6"

    # Compare whicharm (arm identification) - should be exact match
    whicharm_torch_np = whicharm_torch.cpu().numpy()
    arm_matches = np.sum(whicharm_torch_np == whicharm_fortran)
    arm_match_pct = 100 * arm_matches / n_points

    assert arm_match_pct == 100.0, \
        f"Arm identification: {arm_match_pct:.2f}% accurate (expected 100%)"


@pytest.mark.skipif(not HAS_NE21C, reason="ne21c FORTRAN wrapper not available")
def test_ne_arms_nonzero_coverage(ne2001_data, spiral_arm_paths, test_points):
    """Verify reasonable coverage of non-zero density regions."""
    data = ne2001_data
    arm_x, arm_y, arm_kmax = spiral_arm_paths
    x_coords, y_coords, z_coords = test_points

    x_torch = torch.from_numpy(x_coords)
    y_torch = torch.from_numpy(y_coords)
    z_torch = torch.from_numpy(z_coords)

    nea_torch, _, _, _ = ne_arms(
        x_torch, y_torch, z_torch,
        arm_x, arm_y, arm_kmax,
        data.na, data.ha, data.wa,
        data.Aa, data.Fa,
        data.narm, data.warm,
        data.harm, data.farm,
    )

    # Check that a reasonable number of points have non-zero density
    nonzero_count = torch.count_nonzero(nea_torch).item()
    nonzero_pct = 100 * nonzero_count / len(x_coords)

    # At least 5% of random points should be in spiral arms
    assert nonzero_pct >= 5.0, \
        f"Only {nonzero_pct:.1f}% of points in spiral arms (expected >= 5%)"


def test_ne_arms_output_shapes(ne2001_data, spiral_arm_paths):
    """Test that output shapes are correct."""
    data = ne2001_data
    arm_x, arm_y, arm_kmax = spiral_arm_paths

    # Single point
    x = torch.tensor(0.0)
    y = torch.tensor(0.0)
    z = torch.tensor(0.0)

    nea, fa, whicharm, waa = ne_arms(
        x, y, z,
        arm_x, arm_y, arm_kmax,
        data.na, data.ha, data.wa,
        data.Aa, data.Fa,
        data.narm, data.warm,
        data.harm, data.farm,
    )

    assert nea.shape == torch.Size([]), "Single point should return scalar"
    assert fa.shape == torch.Size([]), "Single point should return scalar"
    assert whicharm.shape == torch.Size([]), "Single point should return scalar"
    assert waa.shape == torch.Size([]), "Single point should return scalar"

    # Batch of points
    x_batch = torch.tensor([0.0, 1.0, 2.0])
    y_batch = torch.tensor([0.0, 1.0, 2.0])
    z_batch = torch.tensor([0.0, 0.0, 0.0])

    nea_b, fa_b, whicharm_b, waa_b = ne_arms(
        x_batch, y_batch, z_batch,
        arm_x, arm_y, arm_kmax,
        data.na, data.ha, data.wa,
        data.Aa, data.Fa,
        data.narm, data.warm,
        data.harm, data.farm,
    )

    assert nea_b.shape == (3,), "Batch should return correct shape"
    assert fa_b.shape == (3,), "Batch should return correct shape"
    assert whicharm_b.shape == (3,), "Batch should return correct shape"
    assert waa_b.shape == (3,), "Batch should return correct shape"


def test_ne_arms_non_negative(ne2001_data, spiral_arm_paths):
    """Test that all outputs are non-negative."""
    data = ne2001_data
    arm_x, arm_y, arm_kmax = spiral_arm_paths

    # Test various points
    np.random.seed(42)
    x = torch.from_numpy(np.random.uniform(-20, 20, 100).astype(np.float32))
    y = torch.from_numpy(np.random.uniform(-20, 20, 100).astype(np.float32))
    z = torch.from_numpy(np.random.uniform(-5, 5, 100).astype(np.float32))

    nea, fa, whicharm, waa = ne_arms(
        x, y, z,
        arm_x, arm_y, arm_kmax,
        data.na, data.ha, data.wa,
        data.Aa, data.Fa,
        data.narm, data.warm,
        data.harm, data.farm,
    )

    assert torch.all(nea >= 0), "nea should be non-negative"
    assert torch.all(fa >= 0), "fa should be non-negative"
    assert torch.all(whicharm >= 0), "whicharm should be non-negative"
    assert torch.all(whicharm <= 5), "whicharm should be <= 5"
    assert torch.all(waa >= 0), "waa should be non-negative"
    assert torch.all(waa <= 1), "waa should be <= 1"
