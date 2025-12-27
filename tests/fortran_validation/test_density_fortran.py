"""
Validate torchgedm DENSITY_2001 against FORTRAN NE2001.

Compares PyTorch implementation with original FORTRAN code
using 10,000 random galactocentric points.

Requires: Compiled ne21c FORTRAN wrapper and Docker environment
"""

import os
import torch
import numpy as np
import pytest
import sys

# Try to import ne21c (FORTRAN wrapper)
sys.path.insert(0, 'ne21c')
try:
    import ne21c
    HAS_NE21C = True
except ImportError:
    HAS_NE21C = False

from torchgedm.ne2001 import density_2001, NE2001Data


@pytest.fixture
def ne2001_data():
    """Load NE2001 data once for all tests."""
    return NE2001Data(device='cpu')


@pytest.fixture
def test_points():
    """Generate random test points."""
    np.random.seed(42)
    torch.manual_seed(42)

    n_points = 10000
    x_np = np.random.uniform(-20, 20, n_points)
    y_np = np.random.uniform(-20, 20, n_points)
    z_np = np.random.uniform(-5, 5, n_points)

    return x_np, y_np, z_np


def compute_error_stats(torch_val, fortran_val):
    """Compute error statistics."""
    abs_diff = np.abs(torch_val - fortran_val)

    # Compute relative difference (avoiding division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_diff = abs_diff / np.maximum(np.abs(fortran_val), 1e-10)
        rel_diff = np.where(np.isfinite(rel_diff), rel_diff, 0)

    return {
        'avg_abs': np.mean(abs_diff),
        'max_abs': np.max(abs_diff),
        'avg_rel': np.mean(rel_diff) * 100,  # as percentage
        'max_rel': np.max(rel_diff) * 100,
    }


@pytest.mark.skipif(not HAS_NE21C, reason="ne21c FORTRAN wrapper not available")
def test_density_total_vs_fortran(ne2001_data, test_points):
    """Compare total density against FORTRAN."""
    data = ne2001_data
    x_np, y_np, z_np = test_points
    n_points = len(x_np)

    x_torch = torch.from_numpy(x_np).float()
    y_torch = torch.from_numpy(y_np).float()
    z_torch = torch.from_numpy(z_np).float()

    # Compute PyTorch density
    result_torch = density_2001(x_torch, y_torch, z_torch, data)

    # Compute FORTRAN density
    ne1_fortran = np.zeros(n_points)
    ne2_fortran = np.zeros(n_points)
    nea_fortran = np.zeros(n_points)
    negc_fortran = np.zeros(n_points)
    nelism_fortran = np.zeros(n_points)
    necN_fortran = np.zeros(n_points)
    nevN_fortran = np.zeros(n_points)

    for i in range(n_points):
        result = ne21c.density_xyz(x_np[i], y_np[i], z_np[i])
        ne1_fortran[i] = result['ne1']
        ne2_fortran[i] = result['ne2']
        nea_fortran[i] = result['nea']
        negc_fortran[i] = result['negc']
        nelism_fortran[i] = result['nelism']
        necN_fortran[i] = result['necn']
        nevN_fortran[i] = result['nevn']

    # Compute total densities
    total_torch = (result_torch.ne1 + result_torch.ne2 + result_torch.nea +
                   result_torch.negc + result_torch.nelism +
                   result_torch.necN + result_torch.nevN).numpy()
    total_fortran = (ne1_fortran + ne2_fortran + nea_fortran +
                     negc_fortran + nelism_fortran + necN_fortran + nevN_fortran)

    # Compute errors
    stats = compute_error_stats(total_torch, total_fortran)

    # Assert tolerances
    assert stats['avg_abs'] < 1e-5, \
        f"Average abs error {stats['avg_abs']:.2e} exceeds 1e-5"
    assert stats['max_abs'] < 1e-2, \
        f"Max abs error {stats['max_abs']:.2e} exceeds 1e-2"


@pytest.mark.skipif(not HAS_NE21C, reason="ne21c FORTRAN wrapper not available")
@pytest.mark.parametrize("component,tol_avg,tol_max", [
    ("ne1", 1e-6, 1e-3),   # thick disk
    ("ne2", 1e-6, 1e-3),   # thin disk
    ("nea", 1e-6, 1e-3),   # spiral arms
    ("negc", 1e-6, 1e-3),  # Galactic center
    ("nelism", 1e-6, 1e-3), # Local ISM
    ("necN", 1e-6, 1e-3),  # discrete clumps
    ("nevN", 1e-6, 1e-3),  # voids
])
def test_density_components_vs_fortran(ne2001_data, test_points, component, tol_avg, tol_max):
    """Compare individual density components against FORTRAN."""
    data = ne2001_data
    x_np, y_np, z_np = test_points
    n_points = len(x_np)

    x_torch = torch.from_numpy(x_np).float()
    y_torch = torch.from_numpy(y_np).float()
    z_torch = torch.from_numpy(z_np).float()

    # Compute PyTorch density
    result_torch = density_2001(x_torch, y_torch, z_torch, data)

    # Compute FORTRAN density
    fortran_component = np.zeros(n_points)
    for i in range(n_points):
        result = ne21c.density_xyz(x_np[i], y_np[i], z_np[i])
        # Map component name to FORTRAN result key
        fortran_key = component if component not in ['necN', 'nevN'] else component.lower()
        fortran_component[i] = result[fortran_key]

    # Get PyTorch component
    torch_component = getattr(result_torch, component).numpy()

    # Compute errors
    stats = compute_error_stats(torch_component, fortran_component)

    # Assert tolerances
    assert stats['avg_abs'] < tol_avg, \
        f"{component}: avg abs error {stats['avg_abs']:.2e} exceeds {tol_avg:.0e}"
    assert stats['max_abs'] < tol_max, \
        f"{component}: max abs error {stats['max_abs']:.2e} exceeds {tol_max:.0e}"


@pytest.mark.skipif(not HAS_NE21C, reason="ne21c FORTRAN wrapper not available")
def test_arm_identification_vs_fortran(ne2001_data, test_points):
    """Compare arm identification against FORTRAN."""
    data = ne2001_data
    x_np, y_np, z_np = test_points
    n_points = len(x_np)

    x_torch = torch.from_numpy(x_np).float()
    y_torch = torch.from_numpy(y_np).float()
    z_torch = torch.from_numpy(z_np).float()

    # Compute PyTorch density
    result_torch = density_2001(x_torch, y_torch, z_torch, data)
    whicharm_torch = result_torch.whicharm.numpy()

    # Compute FORTRAN density
    whicharm_fortran = np.zeros(n_points, dtype=int)
    for i in range(n_points):
        result = ne21c.density_xyz(x_np[i], y_np[i], z_np[i])
        whicharm_fortran[i] = int(result['whicharm'])

    # Check arm identification accuracy
    arm_matches = np.sum(whicharm_torch == whicharm_fortran)
    arm_accuracy = 100 * arm_matches / n_points

    # Should be > 99% accurate (allow for minor boundary differences)
    assert arm_accuracy > 99.0, \
        f"Arm identification only {arm_accuracy:.2f}% accurate (expected > 99%)"


def test_density_output_shapes(ne2001_data):
    """Test that output shapes are correct."""
    data = ne2001_data

    # Single point
    x = torch.tensor(0.0)
    y = torch.tensor(0.0)
    z = torch.tensor(0.0)

    result = density_2001(x, y, z, data)

    assert result.ne1.shape == torch.Size([]), "Single point should return scalar"
    assert result.ne2.shape == torch.Size([]), "Single point should return scalar"
    assert result.nea.shape == torch.Size([]), "Single point should return scalar"

    # Batch of points
    x_batch = torch.tensor([0.0, 1.0, 2.0])
    y_batch = torch.tensor([0.0, 1.0, 2.0])
    z_batch = torch.tensor([0.0, 0.0, 0.0])

    result_batch = density_2001(x_batch, y_batch, z_batch, data)

    assert result_batch.ne1.shape == (3,), "Batch should return correct shape"
    assert result_batch.ne2.shape == (3,), "Batch should return correct shape"
    assert result_batch.nea.shape == (3,), "Batch should return correct shape"


def test_density_non_negative(ne2001_data):
    """Test that all density components are non-negative."""
    data = ne2001_data

    np.random.seed(42)
    x = torch.from_numpy(np.random.uniform(-20, 20, 100).astype(np.float32))
    y = torch.from_numpy(np.random.uniform(-20, 20, 100).astype(np.float32))
    z = torch.from_numpy(np.random.uniform(-5, 5, 100).astype(np.float32))

    result = density_2001(x, y, z, data)

    assert torch.all(result.ne1 >= 0), "ne1 should be non-negative"
    assert torch.all(result.ne2 >= 0), "ne2 should be non-negative"
    assert torch.all(result.nea >= 0), "nea should be non-negative"
    assert torch.all(result.negc >= 0), "negc should be non-negative"
    assert torch.all(result.nelism >= 0), "nelism should be non-negative"
    assert torch.all(result.necN >= 0), "necN should be non-negative"
    assert torch.all(result.nevN >= 0), "nevN should be non-negative"
