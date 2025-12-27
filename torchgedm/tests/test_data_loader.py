"""
Test data loader for NE2001 parameters
"""

import pytest
import torch
from torchgedm.ne2001.data_loader import NE2001Data


def test_data_loader_initialization(device):
    """Test that data loader initializes without errors"""
    data = NE2001Data(device=str(device))
    assert data is not None
    assert data.device == device


def test_gal01_parameters_loaded(device):
    """Test that gal01.inp parameters are loaded correctly"""
    data = NE2001Data(device=str(device))

    # Check weights exist
    assert hasattr(data, 'wg1')
    assert hasattr(data, 'wg2')
    assert hasattr(data, 'wga')
    assert hasattr(data, 'wggc')
    assert hasattr(data, 'wglism')
    assert hasattr(data, 'wgcN')
    assert hasattr(data, 'wgvN')

    # Check component 1 parameters
    assert hasattr(data, 'n1h1')
    assert hasattr(data, 'h1')
    assert hasattr(data, 'A1')
    assert hasattr(data, 'F1')

    # Check component 2 parameters
    assert hasattr(data, 'n2')
    assert hasattr(data, 'h2')
    assert hasattr(data, 'A2')
    assert hasattr(data, 'F2')

    # Check spiral arm parameters
    assert hasattr(data, 'narm')
    assert hasattr(data, 'warm')
    assert hasattr(data, 'harm')
    assert hasattr(data, 'farm')

    # Verify arm parameters have 5 elements
    assert data.narm.shape == (5,)
    assert data.warm.shape == (5,)
    assert data.harm.shape == (5,)
    assert data.farm.shape == (5,)

    # Check that all values are on correct device
    assert data.wg1.device == device
    assert data.narm.device == device


def test_ne_arms_parameters_loaded(device):
    """Test that ne_arms_log_mod.inp parameters are loaded"""
    data = NE2001Data(device=str(device))

    assert hasattr(data, 'arm_a')
    assert hasattr(data, 'arm_rmin')
    assert hasattr(data, 'arm_thmin')
    assert hasattr(data, 'arm_extent')

    # Should have 5 spiral arms
    assert data.arm_a.shape == (5,)
    assert data.arm_rmin.shape == (5,)
    assert data.arm_thmin.shape == (5,)
    assert data.arm_extent.shape == (5,)

    assert data.arm_a.device == device


def test_ne_gc_parameters_loaded(device):
    """Test that ne_gc.inp parameters are loaded"""
    data = NE2001Data(device=str(device))

    assert hasattr(data, 'xgc')
    assert hasattr(data, 'ygc')
    assert hasattr(data, 'zgc')
    assert hasattr(data, 'rgc')
    assert hasattr(data, 'hgc')
    assert hasattr(data, 'negc0')
    assert hasattr(data, 'Fgc0')

    assert data.xgc.device == device


def test_nelism_parameters_loaded(device):
    """Test that nelism.inp parameters are loaded"""
    data = NE2001Data(device=str(device))

    # LDR parameters
    assert hasattr(data, 'aldr')
    assert hasattr(data, 'bldr')
    assert hasattr(data, 'cldr')
    assert hasattr(data, 'xldr')
    assert hasattr(data, 'yldr')
    assert hasattr(data, 'zldr')
    assert hasattr(data, 'thetaldr')
    assert hasattr(data, 'neldr')
    assert hasattr(data, 'Fldr')

    # LSB parameters
    assert hasattr(data, 'alsb')
    assert hasattr(data, 'blsb')
    assert hasattr(data, 'clsb')
    assert hasattr(data, 'xlsb')
    assert hasattr(data, 'ylsb')
    assert hasattr(data, 'zlsb')

    # LHB parameters
    assert hasattr(data, 'alhb')
    assert hasattr(data, 'blhb')
    assert hasattr(data, 'clhb')

    # Loop I parameters
    assert hasattr(data, 'xlpI')
    assert hasattr(data, 'ylpI')
    assert hasattr(data, 'zlpI')
    assert hasattr(data, 'rlpI')

    assert data.aldr.device == device


def test_clumps_loaded(device):
    """Test that neclumpN.NE2001.dat is loaded"""
    data = NE2001Data(device=str(device))

    assert hasattr(data, 'clump_l')
    assert hasattr(data, 'clump_b')
    assert hasattr(data, 'clump_nec')
    assert hasattr(data, 'clump_Fc')
    assert hasattr(data, 'clump_dc')
    assert hasattr(data, 'clump_rc')
    assert hasattr(data, 'clump_edge')
    assert hasattr(data, 'n_clumps')

    # Should have 175 clumps (176 lines - 1 header)
    assert data.n_clumps == 175
    assert data.clump_l.shape == (175,)
    assert data.clump_b.shape == (175,)

    assert data.clump_l.device == device


def test_voids_loaded(device):
    """Test that nevoidN.NE2001.dat is loaded"""
    data = NE2001Data(device=str(device))

    assert hasattr(data, 'void_l')
    assert hasattr(data, 'void_b')
    assert hasattr(data, 'void_dv')
    assert hasattr(data, 'void_nev')
    assert hasattr(data, 'void_Fv')
    assert hasattr(data, 'void_aav')
    assert hasattr(data, 'void_bbv')
    assert hasattr(data, 'void_ccv')
    assert hasattr(data, 'void_thvy')
    assert hasattr(data, 'void_thvz')
    assert hasattr(data, 'void_edge')
    assert hasattr(data, 'n_voids')

    # Should have 17 voids (18 lines - 1 header)
    assert data.n_voids == 17
    assert data.void_l.shape == (17,)
    assert data.void_b.shape == (17,)

    assert data.void_l.device == device


def test_device_transfer():
    """Test that .to(device) works"""
    data = NE2001Data(device='cpu')

    # Check initially on CPU
    assert data.wg1.device == torch.device('cpu')
    assert data.narm.device == torch.device('cpu')
    assert data.clump_l.device == torch.device('cpu')

    # .to() should return self
    result = data.to('cpu')
    assert result is data


def test_repr():
    """Test string representation"""
    data = NE2001Data(device='cpu')
    repr_str = repr(data)

    assert 'NE2001Data' in repr_str
    assert 'device=' in repr_str
    assert 'n_clumps=175' in repr_str
    assert 'n_voids=17' in repr_str


def test_parameter_values(device):
    """Test that some known parameter values are correct"""
    data = NE2001Data(device=str(device))

    # From gal01.inp, we know some values:
    # n1h1: 0.033
    # h1: 0.97
    # A1: 17.5
    # F1: 0.18

    assert abs(data.n1h1.item() - 0.033) < 1e-6
    assert abs(data.h1.item() - 0.97) < 1e-6
    assert abs(data.A1.item() - 17.5) < 1e-6
    assert abs(data.F1.item() - 0.18) < 1e-6

    # From ne_gc.inp:
    # xgc: -0.01
    # negc0: 10.0

    assert abs(data.xgc.item() - (-0.01)) < 1e-6
    assert abs(data.negc0.item() - 10.0) < 1e-6


def test_all_tensors_are_tensors(device):
    """Verify that all loaded data are PyTorch tensors"""
    data = NE2001Data(device=str(device))

    # Check a sample of attributes
    assert isinstance(data.wg1, torch.Tensor)
    assert isinstance(data.narm, torch.Tensor)
    assert isinstance(data.arm_a, torch.Tensor)
    assert isinstance(data.xgc, torch.Tensor)
    assert isinstance(data.aldr, torch.Tensor)
    assert isinstance(data.clump_l, torch.Tensor)
    assert isinstance(data.void_l, torch.Tensor)


def test_tensor_dtypes(device):
    """Verify tensor dtypes are appropriate"""
    data = NE2001Data(device=str(device))

    # Most tensors should be float32 or float64
    assert data.narm.dtype in [torch.float32, torch.float64]
    assert data.arm_a.dtype in [torch.float32, torch.float64]
    assert data.clump_l.dtype == torch.float32  # Explicitly set to float32
    assert data.void_l.dtype == torch.float32   # Explicitly set to float32
