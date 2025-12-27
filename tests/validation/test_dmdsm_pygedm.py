"""
Validation script for dmdsm implementation against pygedm.

Compares PyTorch implementation against original FORTRAN/C code via pygedm.
Tests with a grid of sky directions and distances to ensure numerical accuracy.
"""

import torch
import numpy as np
import pytest
from torchgedm.ne2001 import dist_to_dm, dm_to_dist, NE2001Data

pytest.importorskip("pygedm", reason="pygedm not available for comparison")
import pygedm


@pytest.fixture
def ne2001_data():
    """Load NE2001 data once for all tests."""
    return NE2001Data()


# Test cases: (l_deg, b_deg, dist_kpc, description)
DIST_TO_DM_CASES = [
    (0.0, 0.0, 0.5, "GC direction, 0.5 kpc"),
    (0.0, 0.0, 1.0, "GC direction, 1 kpc"),
    (0.0, 0.0, 5.0, "GC direction, 5 kpc"),
    (180.0, 0.0, 1.0, "Anti-GC, 1 kpc"),
    (180.0, 0.0, 5.0, "Anti-GC, 5 kpc"),
    (90.0, 0.0, 2.0, "Tangent point, 2 kpc"),
    (270.0, 0.0, 2.0, "Opposite tangent, 2 kpc"),
    (45.0, 0.0, 3.0, "l=45, in plane, 3 kpc"),
    (0.0, 45.0, 1.0, "Above plane, 1 kpc"),
    (0.0, -30.0, 2.0, "Below plane, 2 kpc"),
    (135.0, 15.0, 1.5, "l=135, b=15, 1.5 kpc"),
    (225.0, -20.0, 2.5, "l=225, b=-20, 2.5 kpc"),
]


# Test cases: (l_deg, b_deg, dm_pc_cm3, description)
DM_TO_DIST_CASES = [
    (0.0, 0.0, 10.0, "GC direction, DM=10"),
    (0.0, 0.0, 50.0, "GC direction, DM=50"),
    (0.0, 0.0, 100.0, "GC direction, DM=100"),
    (180.0, 0.0, 10.0, "Anti-GC, DM=10"),
    (180.0, 0.0, 50.0, "Anti-GC, DM=50"),
    (90.0, 0.0, 50.0, "Tangent, DM=50"),
    (270.0, 0.0, 50.0, "Opposite tangent, DM=50"),
    (45.0, 0.0, 75.0, "l=45, DM=75"),
    (0.0, 45.0, 20.0, "Above plane, DM=20"),
    (0.0, -30.0, 40.0, "Below plane, DM=40"),
    (135.0, 15.0, 30.0, "l=135, b=15, DM=30"),
    (225.0, -20.0, 60.0, "l=225, b=-20, DM=60"),
]


@pytest.mark.parametrize("l_deg,b_deg,dist_kpc,desc", DIST_TO_DM_CASES)
def test_dist_to_dm(ne2001_data, l_deg, b_deg, dist_kpc, desc):
    """Compare dist_to_dm implementations."""
    data = ne2001_data

    # PyGEDM (original FORTRAN/C implementation)
    dm_pygedm, _ = pygedm.dist_to_dm(l_deg, b_deg, dist_kpc * 1000, method='ne2001')
    dm_pygedm_val = dm_pygedm.value

    # PyTorch implementation
    l = torch.tensor(l_deg)
    b = torch.tensor(b_deg)
    dist = torch.tensor(dist_kpc)
    dm_torch = dist_to_dm(l, b, dist, data)
    dm_torch_val = dm_torch.item()

    # Compare
    abs_error = abs(dm_torch_val - dm_pygedm_val)
    rel_error = abs_error / max(abs(dm_pygedm_val), 1e-6)

    # Assert 0.1% tolerance
    assert rel_error < 1e-3, \
        f"{desc}: rel_error {rel_error*100:.4f}% exceeds 0.1% " \
        f"(PyGEDM={dm_pygedm_val:.3f}, PyTorch={dm_torch_val:.3f})"


@pytest.mark.parametrize("l_deg,b_deg,dm_val,desc", DM_TO_DIST_CASES)
def test_dm_to_dist(ne2001_data, l_deg, b_deg, dm_val, desc):
    """Compare dm_to_dist implementations."""
    data = ne2001_data

    # PyGEDM (original FORTRAN/C implementation)
    dist_pygedm, _ = pygedm.dm_to_dist(l_deg, b_deg, dm_val, method='ne2001')
    dist_pygedm_kpc = dist_pygedm.to('kpc').value

    # PyTorch implementation
    l = torch.tensor(l_deg)
    b = torch.tensor(b_deg)
    dm = torch.tensor(dm_val)
    dist_torch, is_limit = dm_to_dist(l, b, dm, data)
    dist_torch_kpc = dist_torch.item()

    # Compare
    abs_error = abs(dist_torch_kpc - dist_pygedm_kpc)
    rel_error = abs_error / max(abs(dist_pygedm_kpc), 1e-6)

    # Assert 0.1% tolerance
    assert rel_error < 1e-3, \
        f"{desc}: rel_error {rel_error*100:.4f}% exceeds 0.1% " \
        f"(PyGEDM={dist_pygedm_kpc:.3f}, PyTorch={dist_torch_kpc:.3f})"


@pytest.mark.parametrize("l_deg,b_deg,dist_orig", [
    (0.0, 0.0, 0.5),
    (0.0, 0.0, 2.0),
    (180.0, 0.0, 3.0),
    (90.0, 0.0, 1.5),
    (45.0, 30.0, 2.0),
])
def test_round_trip_consistency(ne2001_data, l_deg, b_deg, dist_orig):
    """Test round-trip consistency: dist -> DM -> dist should recover original."""
    data = ne2001_data

    l = torch.tensor(l_deg)
    b = torch.tensor(b_deg)
    d = torch.tensor(dist_orig)

    # dist -> DM
    dm = dist_to_dm(l, b, d, data)

    # DM -> dist
    dist_final, is_limit = dm_to_dist(l, b, dm, data)

    error = abs(dist_final.item() - dist_orig) / dist_orig

    # Assert 1% tolerance for round-trip
    assert error < 0.01, \
        f"Round-trip error {error*100:.4f}% exceeds 1% " \
        f"(orig={dist_orig:.3f}, final={dist_final.item():.3f})"


def test_dist_to_dm_batched(ne2001_data):
    """Test batched dist_to_dm computation."""
    data = ne2001_data

    # Create batch
    l_batch = torch.tensor([0.0, 180.0, 90.0])
    b_batch = torch.tensor([0.0, 0.0, 0.0])
    dist_batch = torch.tensor([1.0, 1.0, 2.0])

    dm_batch = dist_to_dm(l_batch, b_batch, dist_batch, data)

    # Verify shape
    assert dm_batch.shape == (3,), "Batched output should have correct shape"

    # Verify each element individually
    for i in range(3):
        dm_single = dist_to_dm(
            torch.tensor(l_batch[i].item()),
            torch.tensor(b_batch[i].item()),
            torch.tensor(dist_batch[i].item()),
            data
        )
        assert torch.allclose(dm_batch[i], dm_single), \
            f"Batched result at index {i} doesn't match single computation"


def test_dm_to_dist_batched(ne2001_data):
    """Test batched dm_to_dist computation."""
    data = ne2001_data

    # Create batch
    l_batch = torch.tensor([0.0, 180.0, 90.0])
    b_batch = torch.tensor([0.0, 0.0, 0.0])
    dm_batch = torch.tensor([10.0, 10.0, 50.0])

    dist_batch, is_limit_batch = dm_to_dist(l_batch, b_batch, dm_batch, data)

    # Verify shape
    assert dist_batch.shape == (3,), "Batched output should have correct shape"
    assert is_limit_batch.shape == (3,), "Batched is_limit should have correct shape"

    # Verify each element individually
    for i in range(3):
        dist_single, is_limit_single = dm_to_dist(
            torch.tensor(l_batch[i].item()),
            torch.tensor(b_batch[i].item()),
            torch.tensor(dm_batch[i].item()),
            data
        )
        assert torch.allclose(dist_batch[i], dist_single), \
            f"Batched dist at index {i} doesn't match single computation"
        assert is_limit_batch[i] == is_limit_single, \
            f"Batched is_limit at index {i} doesn't match single computation"
