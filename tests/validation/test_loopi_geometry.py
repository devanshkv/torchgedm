"""
Validation test for neLOOPI implementation geometry and behavior.

Tests:
1. Correct geometry (spherical distance)
2. Truncation at z < 0
3. Three regions (inner, shell, outside)
4. Parameter matching with expected values
"""

import torch
import pytest
from torchgedm.ne2001.data_loader import NE2001Data
from torchgedm.ne2001.components.ne_loopi import neLOOPI


@pytest.fixture
def ne2001_data():
    """Load NE2001 data once for all tests."""
    return NE2001Data(device='cpu')


@pytest.fixture
def loopi_params(ne2001_data):
    """Extract Loop I parameters."""
    data = ne2001_data
    return {
        'x_center': data.xlpI.item(),
        'y_center': data.ylpI.item(),
        'z_center': data.zlpI.item(),
        'r_inner': data.rlpI.item(),
        'r_outer': (data.rlpI + data.drlpI).item(),
        'ne_inner': data.nelpI.item(),
        'ne_shell': data.dnelpI.item(),
        'F_inner': data.FlpI.item(),
        'F_shell': data.dFlpI.item(),
    }


def test_center_point(ne2001_data, loopi_params):
    """Test center point is in inner region."""
    data = ne2001_data
    p = loopi_params

    x = torch.tensor(p['x_center'])
    y = torch.tensor(p['y_center'])
    z = torch.tensor(p['z_center'])
    ne, F, w = neLOOPI(x, y, z, data)

    assert ne.item() == p['ne_inner'], "Center should have inner density"
    assert F.item() == p['F_inner'], "Center should have inner fluctuation"
    assert w.item() == 1, "Center should have weight 1"


def test_inner_volume(ne2001_data, loopi_params):
    """Test point in inner volume (r < r_inner)."""
    data = ne2001_data
    p = loopi_params

    r_test = p['r_inner'] * 0.5
    x = torch.tensor(p['x_center'] + r_test)
    y = torch.tensor(p['y_center'])
    z = torch.tensor(p['z_center'])
    ne, F, w = neLOOPI(x, y, z, data)

    assert ne.item() == p['ne_inner'], "Inner volume should have inner density"
    assert F.item() == p['F_inner'], "Inner volume should have inner fluctuation"
    assert w.item() == 1, "Inner volume should have weight 1"


def test_shell_region(ne2001_data, loopi_params):
    """Test point in shell (r_inner < r < r_outer)."""
    data = ne2001_data
    p = loopi_params

    r_test = (p['r_inner'] + p['r_outer']) / 2
    x = torch.tensor(p['x_center'])
    y = torch.tensor(p['y_center'])
    z = torch.tensor(p['z_center'] + r_test)
    ne, F, w = neLOOPI(x, y, z, data)

    assert ne.item() == p['ne_shell'], "Shell should have shell density"
    assert F.item() == p['F_shell'], "Shell should have shell fluctuation"
    assert w.item() == 1, "Shell should have weight 1"


def test_outside_region(ne2001_data, loopi_params):
    """Test point outside (r > r_outer)."""
    data = ne2001_data
    p = loopi_params

    r_test = p['r_outer'] * 1.5
    x = torch.tensor(p['x_center'] + r_test)
    y = torch.tensor(p['y_center'])
    z = torch.tensor(p['z_center'])
    ne, F, w = neLOOPI(x, y, z, data)

    assert ne.item() == 0, "Outside should have zero density"
    assert F.item() == 0, "Outside should have zero fluctuation"
    assert w.item() == 0, "Outside should have zero weight"


def test_z_truncation(ne2001_data, loopi_params):
    """Test truncation at z < 0."""
    data = ne2001_data
    p = loopi_params

    x = torch.tensor(p['x_center'])
    y = torch.tensor(p['y_center'])
    z = torch.tensor(-0.1)  # Below z=0
    ne, F, w = neLOOPI(x, y, z, data)

    assert ne.item() == 0, "z<0 should have zero density"
    assert F.item() == 0, "z<0 should have zero fluctuation"
    assert w.item() == 0, "z<0 should have zero weight"


def test_spherical_symmetry(ne2001_data, loopi_params):
    """Test spherical symmetry (same distance, different directions)."""
    data = ne2001_data
    p = loopi_params

    r_test = p['r_inner'] * 0.5

    # Point in +x direction
    x1 = torch.tensor(p['x_center'] + r_test)
    y1 = torch.tensor(p['y_center'])
    z1 = torch.tensor(p['z_center'])
    ne1, F1, w1 = neLOOPI(x1, y1, z1, data)

    # Point in +y direction
    x2 = torch.tensor(p['x_center'])
    y2 = torch.tensor(p['y_center'] + r_test)
    z2 = torch.tensor(p['z_center'])
    ne2, F2, w2 = neLOOPI(x2, y2, z2, data)

    # Point in +z direction
    x3 = torch.tensor(p['x_center'])
    y3 = torch.tensor(p['y_center'])
    z3 = torch.tensor(p['z_center'] + r_test)
    ne3, F3, w3 = neLOOPI(x3, y3, z3, data)

    assert ne1 == ne2 == ne3, "Density should be spherically symmetric"
    assert F1 == F2 == F3, "Fluctuation should be spherically symmetric"
    assert w1 == w2 == w3, "Weight should be spherically symmetric"


def test_boundary_r_inner(ne2001_data, loopi_params):
    """Test boundary at r = r_inner."""
    data = ne2001_data
    p = loopi_params

    r_test = p['r_inner']
    x = torch.tensor(p['x_center'])
    y = torch.tensor(p['y_center'])
    z = torch.tensor(p['z_center'] + r_test)
    ne, F, w = neLOOPI(x, y, z, data)

    # At boundary, should be in inner region (r <= r_inner)
    assert ne.item() == p['ne_inner'], "Boundary at r_inner should have inner density"
    assert w.item() == 1, "Boundary at r_inner should have weight 1"


def test_near_r_outer_boundary(ne2001_data, loopi_params):
    """Test point just inside r = r_outer."""
    data = ne2001_data
    p = loopi_params

    r_test = p['r_outer'] * 0.99  # Slightly inside to avoid FP precision issues
    x = torch.tensor(p['x_center'])
    y = torch.tensor(p['y_center'])
    z = torch.tensor(p['z_center'] + r_test)
    ne, F, w = neLOOPI(x, y, z, data)

    # Should be in shell region
    assert ne.item() == p['ne_shell'], "Near r_outer should have shell density"
    assert w.item() == 1, "Near r_outer should have weight 1"


def test_batched_computation(ne2001_data, loopi_params):
    """Test batched computation consistency."""
    data = ne2001_data
    p = loopi_params

    shell_r = (p['r_inner'] + p['r_outer']) / 2 * 0.95  # Avoid boundary issues

    x_batch = torch.tensor([p['x_center'], p['x_center'] + p['r_inner'] * 0.5, p['x_center']])
    y_batch = torch.tensor([p['y_center'], p['y_center'], p['y_center']])
    z_batch = torch.tensor([p['z_center'], p['z_center'], p['z_center'] + shell_r])

    ne_batch, F_batch, w_batch = neLOOPI(x_batch, y_batch, z_batch, data)

    expected_ne = torch.tensor([p['ne_inner'], p['ne_inner'], p['ne_shell']])
    expected_F = torch.tensor([p['F_inner'], p['F_inner'], p['F_shell']])
    expected_w = torch.tensor([1, 1, 1])

    assert torch.allclose(ne_batch, expected_ne), "Batched ne doesn't match expected"
    assert torch.allclose(F_batch, expected_F), "Batched F doesn't match expected"
    assert torch.all(w_batch == expected_w), "Batched w doesn't match expected"
