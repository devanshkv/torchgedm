"""
Test that torchgedm package imports correctly
"""

import pytest


def test_import_torchgedm():
    """Test that we can import torchgedm"""
    import torchgedm
    assert torchgedm.__version__ == "0.1.0"


def test_import_ne2001_module():
    """Test that we can import ne2001 submodule"""
    import torchgedm.ne2001
    assert torchgedm.ne2001 is not None


def test_import_components():
    """Test that we can import components submodule"""
    import torchgedm.ne2001.components
    assert torchgedm.ne2001.components is not None


def test_torch_available():
    """Test that PyTorch is available"""
    import torch
    assert torch.__version__ is not None
    print(f"PyTorch version: {torch.__version__}")


def test_device_fixture(device):
    """Test the device fixture from conftest"""
    import torch
    assert isinstance(device, torch.device)
    assert device.type == 'cpu'


def test_comparison_utils_fixture(comparison_utils):
    """Test that comparison utils are available"""
    assert 'calc_avg_diff' in comparison_utils
    assert 'calc_max_diff' in comparison_utils
    assert 'calc_rel_diff' in comparison_utils
    assert 'assert_within_tolerance' in comparison_utils
