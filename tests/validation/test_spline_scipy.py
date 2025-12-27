"""
Validation script comparing PyTorch spline implementation with SciPy CubicSpline.

Compares the PyTorch cubic spline against:
1. SciPy's CubicSpline (natural boundary conditions)
2. Reference test cases with known outputs
"""

import numpy as np
import torch
import pytest
from scipy.interpolate import CubicSpline
from torchgedm.ne2001.spline import cspline


def calc_stats(torch_val, reference_val):
    """Calculate error statistics."""
    if isinstance(torch_val, torch.Tensor):
        torch_val = torch_val.detach().cpu().numpy()
    if isinstance(reference_val, torch.Tensor):
        reference_val = reference_val.detach().cpu().numpy()

    abs_diff = np.abs(torch_val - reference_val)
    rel_diff = abs_diff / (np.abs(reference_val) + 1e-10)

    return {
        'avg_abs': np.mean(abs_diff),
        'max_abs': np.max(abs_diff),
        'avg_rel': np.mean(rel_diff) * 100,  # as percentage
        'max_rel': np.max(rel_diff) * 100,
    }


def test_against_scipy_basic():
    """Test against SciPy's CubicSpline implementation."""
    n_knots = 10
    n_eval = 100

    # Generate test data
    x = np.linspace(0, 2*np.pi, n_knots)
    y = np.sin(x) + 0.5 * np.cos(2*x)
    x_eval = np.linspace(0, 2*np.pi, n_eval)

    # Convert to torch
    x_torch = torch.from_numpy(x).float()
    y_torch = torch.from_numpy(y).float()
    x_eval_torch = torch.from_numpy(x_eval).float()

    # Compute with our implementation
    y_eval_torch, _ = cspline(x_torch, y_torch, x_eval_torch)

    # Compute with SciPy (natural boundary conditions)
    scipy_spline = CubicSpline(x, y, bc_type='natural')
    y_eval_scipy = scipy_spline(x_eval)

    # Compare
    stats = calc_stats(y_eval_torch, y_eval_scipy)

    # Assert tolerances
    assert stats['max_abs'] < 5e-6, f"Max abs diff {stats['max_abs']:.2e} exceeds tolerance"
    assert stats['avg_rel'] < 0.001, f"Avg rel diff {stats['avg_rel']:.4f}% exceeds 0.001%"


def test_spiral_arm_data():
    """Test on realistic spiral arm data."""
    # Simulate spiral arm control points (theta vs radius)
    theta_knots = np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]) * np.pi / 180
    r_knots = np.array([8.0, 9.2, 10.5, 11.8, 13.0, 14.1, 15.0])

    # Fine evaluation grid
    theta_eval = np.linspace(0, np.pi, 200)

    # Convert to torch
    theta_torch = torch.from_numpy(theta_knots).float()
    r_torch = torch.from_numpy(r_knots).float()
    theta_eval_torch = torch.from_numpy(theta_eval).float()

    # Compute with our implementation
    r_eval_torch, _ = cspline(theta_torch, r_torch, theta_eval_torch)

    # Compute with SciPy
    scipy_spline = CubicSpline(theta_knots, r_knots, bc_type='natural')
    r_eval_scipy = scipy_spline(theta_eval)

    # Compare
    stats = calc_stats(r_eval_torch, r_eval_scipy)

    assert stats['max_abs'] < 5e-6, f"Max abs diff {stats['max_abs']:.2e} exceeds tolerance"
    assert stats['avg_rel'] < 0.001, f"Avg rel diff {stats['avg_rel']:.4f}% exceeds 0.001%"


def test_batched_operations():
    """Test batched spline computation."""
    n_arms = 5
    n_knots = 8
    n_eval = 150

    # Generate multiple spiral arms
    theta_knots = np.linspace(0, 2*np.pi, n_knots)
    np.random.seed(42)  # For reproducibility
    r_knots = np.stack([
        np.linspace(8, 15, n_knots) + np.random.randn() * 0.5
        for _ in range(n_arms)
    ])
    theta_eval = np.linspace(0, 2*np.pi, n_eval)

    # Convert to torch (batch dimension first)
    theta_torch = torch.from_numpy(theta_knots).float().unsqueeze(0).expand(n_arms, -1)
    r_torch = torch.from_numpy(r_knots).float()
    theta_eval_torch = torch.from_numpy(theta_eval).float().unsqueeze(0).expand(n_arms, -1)

    # Compute with our implementation (batched)
    r_eval_torch, _ = cspline(theta_torch, r_torch, theta_eval_torch)

    # Compute with SciPy (loop over arms)
    r_eval_scipy = np.stack([
        CubicSpline(theta_knots, r_knots[i], bc_type='natural')(theta_eval)
        for i in range(n_arms)
    ])

    # Compare
    stats = calc_stats(r_eval_torch, r_eval_scipy)

    assert stats['max_abs'] < 5e-6, f"Max abs diff {stats['max_abs']:.2e} exceeds tolerance"
    assert stats['avg_rel'] < 0.001, f"Avg rel diff {stats['avg_rel']:.4f}% exceeds 0.001%"


def test_linear_function():
    """Test linear function (should be exact)."""
    x = np.linspace(0, 10, 5)
    y = 2 * x + 3
    x_eval = np.linspace(0, 10, 50)

    x_torch = torch.from_numpy(x).float()
    y_torch = torch.from_numpy(y).float()
    x_eval_torch = torch.from_numpy(x_eval).float()

    y_eval_torch, _ = cspline(x_torch, y_torch, x_eval_torch)
    y_eval_expected = 2 * x_eval + 3

    stats = calc_stats(y_eval_torch, y_eval_expected)

    # Linear functions should be interpolated exactly by cubic splines
    assert stats['max_abs'] < 1e-6, f"Linear function: max abs diff {stats['max_abs']:.2e}"


def test_minimum_knots():
    """Test minimum knots (n=2)."""
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    x_eval = np.array([0.5])

    x_torch = torch.from_numpy(x).float()
    y_torch = torch.from_numpy(y).float()
    x_eval_torch = torch.from_numpy(x_eval).float()

    y_eval_torch, _ = cspline(x_torch, y_torch, x_eval_torch)

    scipy_spline = CubicSpline(x, y, bc_type='natural')
    y_eval_scipy = scipy_spline(x_eval)

    stats = calc_stats(y_eval_torch, y_eval_scipy)

    assert stats['max_abs'] < 5e-6, f"Min knots: max abs diff {stats['max_abs']:.2e}"


def test_many_knots():
    """Test many knots (n=50)."""
    x = np.linspace(0, 10, 50)
    y = np.sin(x) * np.exp(-x/10)
    x_eval = np.linspace(0, 10, 500)

    x_torch = torch.from_numpy(x).float()
    y_torch = torch.from_numpy(y).float()
    x_eval_torch = torch.from_numpy(x_eval).float()

    y_eval_torch, _ = cspline(x_torch, y_torch, x_eval_torch)

    scipy_spline = CubicSpline(x, y, bc_type='natural')
    y_eval_scipy = scipy_spline(x_eval)

    stats = calc_stats(y_eval_torch, y_eval_scipy)

    assert stats['max_abs'] < 5e-6, f"Many knots: max abs diff {stats['max_abs']:.2e}"
    assert stats['avg_rel'] < 0.001, f"Many knots: avg rel diff {stats['avg_rel']:.4f}%"


@pytest.mark.parametrize("n_knots,n_eval", [
    (5, 50),
    (10, 100),
    (20, 200),
])
def test_parametric_sizes(n_knots, n_eval):
    """Test various knot and evaluation sizes."""
    x = np.linspace(0, 2*np.pi, n_knots)
    y = np.sin(x)
    x_eval = np.linspace(0, 2*np.pi, n_eval)

    x_torch = torch.from_numpy(x).float()
    y_torch = torch.from_numpy(y).float()
    x_eval_torch = torch.from_numpy(x_eval).float()

    y_eval_torch, _ = cspline(x_torch, y_torch, x_eval_torch)

    scipy_spline = CubicSpline(x, y, bc_type='natural')
    y_eval_scipy = scipy_spline(x_eval)

    stats = calc_stats(y_eval_torch, y_eval_scipy)

    assert stats['max_abs'] < 5e-6, \
        f"n_knots={n_knots}, n_eval={n_eval}: max abs diff {stats['max_abs']:.2e}"
