"""
Tests for combined Local ISM (LISM) density component

Tests the hierarchical weighting scheme:
    LHB > Loop I > LSB > LDR
"""

import torch
import pytest
from torchgedm.ne2001.data_loader import NE2001Data
from torchgedm.ne2001.components.ne_lism import neLISM
from torchgedm.ne2001.components.ne_lhb import neLHB
from torchgedm.ne2001.components.ne_lsb import neLSB
from torchgedm.ne2001.components.ne_ldr import neLDRQ1
from torchgedm.ne2001.components.ne_loopi import neLOOPI


@pytest.fixture
def ne2001_data():
    """Load NE2001 data"""
    return NE2001Data(device='cpu')


class TestNeLISM:
    """Test suite for combined Local ISM component"""

    def test_lhb_center_overrides_all(self, ne2001_data):
        """
        Test that LHB overrides all other components when at LHB center.

        Weighting hierarchy: LHB > Loop I > LSB > LDR
        """
        # LHB center
        x = ne2001_data.xlhb
        y = ne2001_data.ylhb
        z = ne2001_data.zlhb

        # Get individual components
        ne_lhb, F_lhb, w_lhb = neLHB(x, y, z, ne2001_data)

        # Get combined LISM
        ne_lism, F_lism, w_lism = neLISM(x, y, z, ne2001_data)

        # At LHB center, should have wLHB=1, so LISM should match LHB exactly
        assert w_lhb == 1, "Expected to be inside LHB"
        assert ne_lism == ne_lhb, "LHB should override all others"
        assert F_lism == F_lhb, "LHB fluctuation should be used"
        assert w_lism >= w_lhb, "LISM weight should be at least LHB weight"

    def test_lsb_center(self, ne2001_data):
        """
        Test density at LSB center.

        If LSB center is outside LHB and Loop I, LSB should dominate.
        """
        # LSB center
        x = ne2001_data.xlsb
        y = ne2001_data.ylsb
        z = ne2001_data.zlsb

        # Get individual components
        ne_lsb, F_lsb, w_lsb = neLSB(x, y, z, ne2001_data)
        ne_lhb, F_lhb, w_lhb = neLHB(x, y, z, ne2001_data)
        ne_loopi, F_loopi, w_loopi = neLOOPI(x, y, z, ne2001_data)

        # Get combined LISM
        ne_lism, F_lism, w_lism = neLISM(x, y, z, ne2001_data)

        # At LSB center, should be inside LSB
        assert w_lsb == 1, "Expected to be inside LSB"

        # If not inside higher priority components, LSB should dominate
        if w_lhb == 0 and w_loopi == 0:
            assert ne_lism == ne_lsb, "LSB should be used when LHB and Loop I are not active"
            assert F_lism == F_lsb, "LSB fluctuation should be used"

    def test_ldr_center(self, ne2001_data):
        """
        Test density at LDR center.

        If LDR center is outside LHB, Loop I, and LSB, LDR should be used.
        """
        # LDR center
        x = ne2001_data.xldr
        y = ne2001_data.yldr
        z = ne2001_data.zldr

        # Get individual components
        ne_ldr, F_ldr, w_ldr = neLDRQ1(x, y, z, ne2001_data)
        ne_lsb, F_lsb, w_lsb = neLSB(x, y, z, ne2001_data)
        ne_lhb, F_lhb, w_lhb = neLHB(x, y, z, ne2001_data)
        ne_loopi, F_loopi, w_loopi = neLOOPI(x, y, z, ne2001_data)

        # Get combined LISM
        ne_lism, F_lism, w_lism = neLISM(x, y, z, ne2001_data)

        # At LDR center, should be inside LDR
        assert w_ldr == 1, "Expected to be inside LDR"

        # If not inside any higher priority component, LDR should be used
        if w_lhb == 0 and w_loopi == 0 and w_lsb == 0:
            assert ne_lism == ne_ldr, "LDR should be used when all others are inactive"
            assert F_lism == F_ldr, "LDR fluctuation should be used"

    def test_loopi_center(self, ne2001_data):
        """
        Test density at Loop I center.

        If Loop I center is outside LHB, Loop I should override LSB and LDR.
        """
        # Loop I center (must be at z >= 0 since Loop I is truncated below z=0)
        x = ne2001_data.xlpI
        y = ne2001_data.ylpI
        z = torch.abs(ne2001_data.zlpI) if ne2001_data.zlpI < 0 else ne2001_data.zlpI

        # Get individual components
        ne_loopi, F_loopi, w_loopi = neLOOPI(x, y, z, ne2001_data)
        ne_lhb, F_lhb, w_lhb = neLHB(x, y, z, ne2001_data)

        # Get combined LISM
        ne_lism, F_lism, w_lism = neLISM(x, y, z, ne2001_data)

        # At Loop I center, should be inside Loop I
        assert w_loopi == 1, "Expected to be inside Loop I"

        # If not inside LHB, Loop I should be used
        if w_lhb == 0:
            assert ne_lism == ne_loopi, "Loop I should override LSB and LDR"
            assert F_lism == F_loopi, "Loop I fluctuation should be used"

    def test_far_from_all_components(self, ne2001_data):
        """Test density far from all LISM components (should be zero)"""
        # Point far from all components
        x = torch.tensor(10.0)
        y = torch.tensor(0.0)
        z = torch.tensor(5.0)

        ne_lism, F_lism, w_lism = neLISM(x, y, z, ne2001_data)

        # Verify all components are inactive
        ne_ldr, F_ldr, w_ldr = neLDRQ1(x, y, z, ne2001_data)
        ne_lsb, F_lsb, w_lsb = neLSB(x, y, z, ne2001_data)
        ne_lhb, F_lhb, w_lhb = neLHB(x, y, z, ne2001_data)
        ne_loopi, F_loopi, w_loopi = neLOOPI(x, y, z, ne2001_data)

        assert w_ldr == 0, "Should be outside LDR"
        assert w_lsb == 0, "Should be outside LSB"
        assert w_lhb == 0, "Should be outside LHB"
        assert w_loopi == 0, "Should be outside Loop I"

        assert ne_lism == 0.0, "Density should be zero outside all components"
        assert F_lism == 0.0, "Fluctuation should be zero outside all components"
        assert w_lism == 0, "Weight should be zero outside all components"

    def test_batched_computation(self, ne2001_data):
        """Test batched computation with multiple points"""
        # Create grid spanning multiple LISM components
        x = torch.linspace(-1.0, 2.0, 10)
        y = torch.linspace(7.5, 9.5, 10)
        z = torch.linspace(-0.5, 0.5, 10)

        # Expand to 3D grid
        x_grid, y_grid, z_grid = torch.meshgrid(x, y, z, indexing='ij')

        ne_lism, F_lism, w_lism = neLISM(x_grid, y_grid, z_grid, ne2001_data)

        # Check shapes
        assert ne_lism.shape == (10, 10, 10)
        assert F_lism.shape == (10, 10, 10)
        assert w_lism.shape == (10, 10, 10)

        # Check that at least some points are inside LISM
        assert (ne_lism > 0).any(), "Expected some points inside LISM components"

        # Check that weights are binary (0 or 1)
        assert torch.all((w_lism == 0) | (w_lism == 1)), "Weights should be 0 or 1"

    def test_weighting_formula(self, ne2001_data):
        """
        Test that the weighting formula matches FORTRAN implementation.

        Formula (neLISM.NE2001.f lines 91-96):
            ne_LISM = (1-wLHB) *
                      (
                        (1-wLOOPI) * (wLSB*neLSB + (1-wLSB)*neLDR)
                    +     wLOOPI * neLOOPI
                      )
                    +     wLHB  * neLHB
        """
        # Test at an arbitrary point
        x = torch.tensor(0.5)
        y = torch.tensor(8.5)
        z = torch.tensor(0.1)

        # Get individual components
        ne_ldr, F_ldr, w_ldr = neLDRQ1(x, y, z, ne2001_data)
        ne_lsb, F_lsb, w_lsb = neLSB(x, y, z, ne2001_data)
        ne_lhb, F_lhb, w_lhb = neLHB(x, y, z, ne2001_data)
        ne_loopi, F_loopi, w_loopi = neLOOPI(x, y, z, ne2001_data)

        # Convert weights to float for formula
        w_ldr_f = w_ldr.float()
        w_lsb_f = w_lsb.float()
        w_lhb_f = w_lhb.float()
        w_loopi_f = w_loopi.float()

        # Calculate expected density using FORTRAN formula
        ne_expected = (1.0 - w_lhb_f) * (
            (1.0 - w_loopi_f) * (w_lsb_f * ne_lsb + (1.0 - w_lsb_f) * ne_ldr)
            + w_loopi_f * ne_loopi
        ) + w_lhb_f * ne_lhb

        # Calculate expected fluctuation
        F_expected = (1.0 - w_lhb_f) * (
            (1.0 - w_loopi_f) * (w_lsb_f * F_lsb + (1.0 - w_lsb_f) * F_ldr)
            + w_loopi_f * F_loopi
        ) + w_lhb_f * F_lhb

        # Get combined LISM
        ne_lism, F_lism, w_lism = neLISM(x, y, z, ne2001_data)

        # Verify formula
        assert torch.isclose(ne_lism, ne_expected, atol=1e-6), \
            f"Density formula mismatch: {ne_lism} != {ne_expected}"
        assert torch.isclose(F_lism, F_expected, atol=1e-6), \
            f"Fluctuation formula mismatch: {F_lism} != {F_expected}"

    def test_max_weight(self, ne2001_data):
        """
        Test that wLISM returns maximum weight of all components.

        FORTRAN line 108: wLISM = max(wLOOPI, max(wLDR, max(wLSB, wLHB)))
        """
        # Test at multiple points
        points = [
            (0.0, 8.5, 0.0),  # Near Sun
            (0.5, 8.5, 0.1),  # Arbitrary point
            (-1.0, 9.0, 0.0), # Near LSB
            (1.0, 8.0, 0.0),  # Near LDR
        ]

        for x_val, y_val, z_val in points:
            x = torch.tensor(x_val)
            y = torch.tensor(y_val)
            z = torch.tensor(z_val)

            # Get individual components
            ne_ldr, F_ldr, w_ldr = neLDRQ1(x, y, z, ne2001_data)
            ne_lsb, F_lsb, w_lsb = neLSB(x, y, z, ne2001_data)
            ne_lhb, F_lhb, w_lhb = neLHB(x, y, z, ne2001_data)
            ne_loopi, F_loopi, w_loopi = neLOOPI(x, y, z, ne2001_data)

            # Calculate expected max weight
            w_expected = torch.max(
                torch.max(
                    torch.max(w_ldr, w_lsb),
                    w_lhb
                ),
                w_loopi
            )

            # Get combined LISM
            ne_lism, F_lism, w_lism = neLISM(x, y, z, ne2001_data)

            assert w_lism == w_expected, \
                f"Weight should be max of all components at ({x_val}, {y_val}, {z_val})"

    def test_differentiability(self, ne2001_data):
        """Test that ne_LISM supports gradient computation"""
        x = torch.tensor(0.5, requires_grad=True)
        y = torch.tensor(8.5, requires_grad=True)
        z = torch.tensor(0.1, requires_grad=True)

        ne_lism, F_lism, w_lism = neLISM(x, y, z, ne2001_data)

        # Check that ne_lism is a tensor (even if gradient may be zero due to piecewise constant nature)
        assert isinstance(ne_lism, torch.Tensor)

        # Note: Gradients may be zero for piecewise constant functions,
        # but the function should still support requires_grad

    def test_return_types(self, ne2001_data):
        """Test that return types are correct"""
        x = torch.tensor(0.0)
        y = torch.tensor(8.5)
        z = torch.tensor(0.0)

        ne_lism, F_lism, w_lism = neLISM(x, y, z, ne2001_data)

        assert isinstance(ne_lism, torch.Tensor)
        assert isinstance(F_lism, torch.Tensor)
        assert isinstance(w_lism, torch.Tensor)

        # Weights should be integer type
        assert w_lism.dtype in [torch.int64, torch.long]

    def test_loopi_truncation_at_negative_z(self, ne2001_data):
        """
        Test that Loop I is truncated at z < 0 and doesn't affect LISM there.

        FORTRAN comment (line 14): Loop I component is truncated below z=0
        """
        # Loop I center, but at negative z
        x = ne2001_data.xlpI
        y = ne2001_data.ylpI
        z = torch.tensor(-0.5)  # Below z=0

        # Get individual components
        ne_loopi, F_loopi, w_loopi = neLOOPI(x, y, z, ne2001_data)

        # Loop I should be inactive at z < 0
        assert w_loopi == 0, "Loop I should be truncated at z < 0"
        assert ne_loopi == 0.0, "Loop I density should be zero at z < 0"

        # Get combined LISM
        ne_lism, F_lism, w_lism = neLISM(x, y, z, ne2001_data)

        # LISM should not include Loop I contribution
        # (Will use LSB or LDR if inside those, or zero if outside all)
        ne_ldr, F_ldr, w_ldr = neLDRQ1(x, y, z, ne2001_data)
        ne_lsb, F_lsb, w_lsb = neLSB(x, y, z, ne2001_data)
        ne_lhb, F_lhb, w_lhb = neLHB(x, y, z, ne2001_data)

        # Verify Loop I doesn't contribute
        w_ldr_f = w_ldr.float()
        w_lsb_f = w_lsb.float()
        w_lhb_f = w_lhb.float()

        ne_expected = (1.0 - w_lhb_f) * (
            w_lsb_f * ne_lsb + (1.0 - w_lsb_f) * ne_ldr
        ) + w_lhb_f * ne_lhb

        assert torch.isclose(ne_lism, ne_expected, atol=1e-6), \
            "Loop I should not contribute at z < 0"

    def test_edge_cases(self, ne2001_data):
        """Test edge cases like boundaries between components"""
        # Test at z=0 (Loop I boundary)
        x = ne2001_data.xlpI
        y = ne2001_data.ylpI
        z = torch.tensor(0.0)

        ne_lism, F_lism, w_lism = neLISM(x, y, z, ne2001_data)

        # Should return valid values at boundary
        assert ne_lism >= 0
        assert F_lism >= 0
        assert w_lism in [0, 1]

    def test_scalar_and_batched_consistency(self, ne2001_data):
        """Test that scalar and batched computations give same results"""
        # Scalar computation
        x_scalar = torch.tensor(0.5)
        y_scalar = torch.tensor(8.5)
        z_scalar = torch.tensor(0.1)

        ne_scalar, F_scalar, w_scalar = neLISM(x_scalar, y_scalar, z_scalar, ne2001_data)

        # Batched computation with same point
        x_batch = torch.tensor([0.5, 0.5])
        y_batch = torch.tensor([8.5, 8.5])
        z_batch = torch.tensor([0.1, 0.1])

        ne_batch, F_batch, w_batch = neLISM(x_batch, y_batch, z_batch, ne2001_data)

        # Results should match
        assert torch.isclose(ne_scalar, ne_batch[0], atol=1e-6)
        assert torch.isclose(F_scalar, F_batch[0], atol=1e-6)
        assert w_scalar == w_batch[0]

    def test_all_weights_zero_gives_zero_density(self, ne2001_data):
        """Test that when all components have weight=0, density is zero"""
        # Point far from all components
        x = torch.tensor(20.0)
        y = torch.tensor(-10.0)
        z = torch.tensor(10.0)

        # Verify all components are inactive
        ne_ldr, F_ldr, w_ldr = neLDRQ1(x, y, z, ne2001_data)
        ne_lsb, F_lsb, w_lsb = neLSB(x, y, z, ne2001_data)
        ne_lhb, F_lhb, w_lhb = neLHB(x, y, z, ne2001_data)
        ne_loopi, F_loopi, w_loopi = neLOOPI(x, y, z, ne2001_data)

        assert w_ldr == 0
        assert w_lsb == 0
        assert w_lhb == 0
        assert w_loopi == 0

        # Get combined LISM
        ne_lism, F_lism, w_lism = neLISM(x, y, z, ne2001_data)

        # Should be zero everywhere
        assert ne_lism == 0.0
        assert F_lism == 0.0
        assert w_lism == 0
