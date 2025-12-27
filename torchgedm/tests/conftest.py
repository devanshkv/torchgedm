"""
Pytest configuration and fixtures for torchgedm tests
"""

import pytest
import torch
import numpy as np


@pytest.fixture
def device():
    """Return CPU device (GPU testing will be added later)"""
    return torch.device('cpu')


@pytest.fixture
def pygedm_ne2001():
    """Fixture to load original pygedm NE2001 implementation"""
    try:
        import pygedm
        return pygedm
    except ImportError:
        pytest.skip("pygedm not available")


def calc_avg_diff(torch_val, original_val):
    """Calculate average absolute difference"""
    if isinstance(torch_val, torch.Tensor):
        torch_val = torch_val.detach().cpu().numpy()
    if isinstance(original_val, torch.Tensor):
        original_val = original_val.detach().cpu().numpy()

    return np.mean(np.abs(torch_val - original_val))


def calc_max_diff(torch_val, original_val):
    """Calculate maximum absolute difference"""
    if isinstance(torch_val, torch.Tensor):
        torch_val = torch_val.detach().cpu().numpy()
    if isinstance(original_val, torch.Tensor):
        original_val = original_val.detach().cpu().numpy()

    return np.max(np.abs(torch_val - original_val))


def calc_rel_diff(torch_val, original_val, epsilon=1e-10):
    """Calculate relative difference percentage"""
    if isinstance(torch_val, torch.Tensor):
        torch_val = torch_val.detach().cpu().numpy()
    if isinstance(original_val, torch.Tensor):
        original_val = original_val.detach().cpu().numpy()

    # Avoid division by zero
    denominator = np.maximum(np.abs(original_val), epsilon)
    rel_diff = np.abs(torch_val - original_val) / denominator
    return np.mean(rel_diff)


def assert_within_tolerance(torch_val, original_val, tol=1e-6, name="values"):
    """
    Assert that torch and original values match within tolerance

    Args:
        torch_val: Value from PyTorch implementation
        original_val: Value from original implementation
        tol: Tolerance threshold (default 1e-6)
        name: Name of the value being compared (for error messages)
    """
    avg_diff = calc_avg_diff(torch_val, original_val)
    max_diff = calc_max_diff(torch_val, original_val)
    rel_diff = calc_rel_diff(torch_val, original_val)

    print(f"\n{name} comparison:")
    print(f"  Average absolute difference: {avg_diff:.2e}")
    print(f"  Maximum absolute difference: {max_diff:.2e}")
    print(f"  Relative difference: {rel_diff:.2e}")

    assert avg_diff < tol, f"{name}: Average diff {avg_diff:.2e} exceeds tolerance {tol:.2e}"
    assert max_diff < tol, f"{name}: Max diff {max_diff:.2e} exceeds tolerance {tol:.2e}"
    assert rel_diff < tol, f"{name}: Relative diff {rel_diff:.2e} exceeds tolerance {tol:.2e}"


# Make utility functions available as fixtures too
@pytest.fixture
def comparison_utils():
    """Fixture providing comparison utility functions"""
    return {
        'calc_avg_diff': calc_avg_diff,
        'calc_max_diff': calc_max_diff,
        'calc_rel_diff': calc_rel_diff,
        'assert_within_tolerance': assert_within_tolerance,
    }
