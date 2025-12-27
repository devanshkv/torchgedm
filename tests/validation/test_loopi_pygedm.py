"""
Comparison test for neLOOPI against original pygedm implementation

This test generates a grid of points around Loop I and compares
the PyTorch implementation against the original C implementation.
"""

import numpy as np
import pytest
import torch
from torchgedm.ne2001.data_loader import NE2001Data
from torchgedm.ne2001.components.ne_loopi import neLOOPI

# Try to import pygedm for comparison
try:
    from pygedm import calculate_electron_density_xyz
    HAS_PYGEDM = True
except ImportError:
    HAS_PYGEDM = False


@pytest.mark.skipif(not HAS_PYGEDM, reason="pygedm not available")
def test_loopi_vs_pygedm():
    """Compare neLOOPI against original pygedm implementation"""

    # Load data
    data = NE2001Data(device='cpu')

    # Get Loop I parameters to create a grid around it
    x_center = data.xlpI.item()
    y_center = data.ylpI.item()
    z_center = data.zlpI.item()
    r_max = (data.rlpI + data.drlpI).item()

    # Create grid around Loop I
    # Use a reasonable grid: -0.5 to +0.5 kpc around center in each direction
    n_points = 20
    x_range = np.linspace(x_center - 0.5, x_center + 0.5, n_points)
    y_range = np.linspace(y_center - 0.5, y_center + 0.5, n_points)
    z_range = np.linspace(z_center - 0.5, z_center + 0.5, n_points)

    # Create 3D grid
    x_grid_np, y_grid_np, z_grid_np = np.meshgrid(x_range, y_range, z_range, indexing='ij')

    # Flatten for testing
    x_flat = x_grid_np.flatten()
    y_flat = y_grid_np.flatten()
    z_flat = z_grid_np.flatten()

    n_test = len(x_flat)

    # Get PyTorch results
    x_torch = torch.from_numpy(x_flat).float()
    y_torch = torch.from_numpy(y_flat).float()
    z_torch = torch.from_numpy(z_flat).float()

    ne_torch, F_torch, w_torch = neLOOPI(x_torch, y_torch, z_torch, data)
    ne_torch_np = ne_torch.numpy()

    # Get original pygedm results
    # Note: calculate_electron_density_xyz returns total density as Quantity
    # In the Loop I region, LISM (including Loop I) dominates
    ne_original = np.zeros(n_test)

    for i in range(n_test):
        result = calculate_electron_density_xyz(
            x_flat[i], y_flat[i], z_flat[i]
        )
        # Convert Quantity to value (in cm^-3)
        ne_original[i] = result.value if hasattr(result, 'value') else float(result)

    # Compare
    # Filter for points where either implementation has non-zero density
    mask = (ne_torch_np > 0) | (ne_original > 0)
    n_inside = mask.sum()

    assert n_inside > 0, "No points inside Loop I in test grid"

    ne_torch_inside = ne_torch_np[mask]
    ne_original_inside = ne_original[mask]

    # Calculate differences
    abs_diff = np.abs(ne_torch_inside - ne_original_inside)
    rel_diff = abs_diff / (np.abs(ne_original_inside) + 1e-10)

    # Check acceptance criteria
    avg_rel = rel_diff.mean()
    max_rel = rel_diff.max()

    # Assert tolerances (< 1e-6 relative error)
    assert avg_rel < 1e-6, f"Average relative diff {avg_rel:.3e} exceeds tolerance 1e-6"
    assert max_rel < 1e-6, f"Max relative diff {max_rel:.3e} exceeds tolerance 1e-6"
