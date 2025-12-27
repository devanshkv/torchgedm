"""
Tests for scattering functions

Tests the scattering calculation functions against known values
from the FORTRAN implementation.
"""

import pytest
import torch
import math
from torchgedm.ne2001.scattering import (
    tauiss, scintbw, scintime, specbroad,
    theta_xgal, theta_gal, emission_measure, transition_frequency
)


class TestTauiss:
    """Test TAUISS function (pulse broadening)."""

    def test_basic_calculation(self):
        """Test basic TAUISS calculation."""
        d = torch.tensor(1.0)    # 1 kpc
        sm = torch.tensor(292.0)  # 292 kpc m^{-20/3}
        nu = torch.tensor(1.0)    # 1 GHz

        tau = tauiss(d, sm, nu)

        # tauiss = 1.0 * (292/292)**1.2 * 1 * 1**(-4.4) = 1.0
        expected = 1.0
        assert torch.isclose(tau, torch.tensor(expected), rtol=1e-6)

    def test_frequency_scaling(self):
        """Test frequency scaling: tau ~ nu^{-4.4}."""
        d = torch.tensor(1.0)
        sm = torch.tensor(100.0)
        nu1 = torch.tensor(1.0)
        nu2 = torch.tensor(2.0)

        tau1 = tauiss(d, sm, nu1)
        tau2 = tauiss(d, sm, nu2)

        # tau2 / tau1 should be (nu1 / nu2)^4.4 = 0.5^4.4
        ratio = tau2 / tau1
        expected_ratio = (nu1 / nu2)**4.4
        assert torch.isclose(ratio, expected_ratio, rtol=1e-6)

    def test_distance_scaling(self):
        """Test distance scaling: tau ~ d."""
        d1 = torch.tensor(1.0)
        d2 = torch.tensor(2.0)
        sm = torch.tensor(100.0)
        nu = torch.tensor(1.0)

        tau1 = tauiss(d1, sm, nu)
        tau2 = tauiss(d2, sm, nu)

        # tau2 / tau1 should be d2 / d1 = 2.0
        ratio = tau2 / tau1
        assert torch.isclose(ratio, torch.tensor(2.0), rtol=1e-6)

    def test_sm_scaling(self):
        """Test SM scaling: tau ~ (sm)^1.2."""
        d = torch.tensor(1.0)
        sm1 = torch.tensor(100.0)
        sm2 = torch.tensor(200.0)
        nu = torch.tensor(1.0)

        tau1 = tauiss(d, sm1, nu)
        tau2 = tauiss(d, sm2, nu)

        # tau2 / tau1 should be (sm2/sm1)^1.2
        ratio = tau2 / tau1
        expected_ratio = (sm2 / sm1)**1.2
        assert torch.isclose(ratio, expected_ratio, rtol=1e-6)

    def test_differentiable(self):
        """Test that tauiss is differentiable."""
        d = torch.tensor(1.0, requires_grad=True)
        sm = torch.tensor(100.0, requires_grad=True)
        nu = torch.tensor(1.0, requires_grad=True)

        tau = tauiss(d, sm, nu)
        tau.backward()

        assert d.grad is not None
        assert sm.grad is not None
        assert nu.grad is not None


class TestScintbw:
    """Test SCINTBW function (scintillation bandwidth)."""

    def test_basic_calculation(self):
        """Test basic SCINTBW calculation."""
        d = torch.tensor(1.0)
        sm = torch.tensor(292.0)
        nu = torch.tensor(1.0)

        bw = scintbw(d, sm, nu)

        # scintbw = 1.16 / (2 * pi * tauiss_ms)
        # tauiss_ms = 1000.0 for these parameters
        c1 = 1.16
        tauiss_ms = 1000.0
        expected = c1 / (2.0 * math.pi * tauiss_ms)
        assert torch.isclose(bw, torch.tensor(expected), rtol=1e-6)

    def test_relationship_to_tauiss(self):
        """Test relationship: scintbw ~ 1 / tauiss."""
        d = torch.tensor(1.0)
        sm = torch.tensor(100.0)
        nu = torch.tensor(1.0)

        bw1 = scintbw(d, sm, nu)
        bw2 = scintbw(d, sm * 2, nu)  # Double SM

        # If SM doubles, tauiss increases by 2^1.2, so bw decreases by 2^1.2
        ratio = bw2 / bw1
        expected_ratio = 1.0 / (2.0**1.2)
        assert torch.isclose(ratio, torch.tensor(expected_ratio), rtol=1e-5)


class TestScintime:
    """Test SCINTIME function (scintillation timescale)."""

    def test_basic_calculation(self):
        """Test basic SCINTIME calculation."""
        sm = torch.tensor(100.0)
        nu = torch.tensor(1.0)
        vperp = torch.tensor(100.0)

        t_scint = scintime(sm, nu, vperp)

        # scintime = 3.3 * nu^1.2 * sm^(-0.6) * (100/vperp)
        # = 3.3 * 1 * 100^(-0.6) * 1
        expected = 3.3 * 100.0**(-0.6)
        assert torch.isclose(t_scint, torch.tensor(expected), rtol=1e-6)

    def test_velocity_scaling(self):
        """Test velocity scaling: scintime ~ 1/vperp."""
        sm = torch.tensor(100.0)
        nu = torch.tensor(1.0)
        vperp1 = torch.tensor(100.0)
        vperp2 = torch.tensor(200.0)

        t1 = scintime(sm, nu, vperp1)
        t2 = scintime(sm, nu, vperp2)

        # t2 / t1 should be vperp1 / vperp2 = 0.5
        ratio = t2 / t1
        assert torch.isclose(ratio, torch.tensor(0.5), rtol=1e-6)

    def test_frequency_scaling(self):
        """Test frequency scaling: scintime ~ nu^1.2."""
        sm = torch.tensor(100.0)
        nu1 = torch.tensor(1.0)
        nu2 = torch.tensor(2.0)
        vperp = torch.tensor(100.0)

        t1 = scintime(sm, nu1, vperp)
        t2 = scintime(sm, nu2, vperp)

        # t2 / t1 should be (nu2/nu1)^1.2 = 2^1.2
        ratio = t2 / t1
        expected_ratio = 2.0**1.2
        assert torch.isclose(ratio, torch.tensor(expected_ratio), rtol=1e-6)

    def test_sm_scaling(self):
        """Test SM scaling: scintime ~ sm^(-0.6)."""
        sm1 = torch.tensor(100.0)
        sm2 = torch.tensor(200.0)
        nu = torch.tensor(1.0)
        vperp = torch.tensor(100.0)

        t1 = scintime(sm1, nu, vperp)
        t2 = scintime(sm2, nu, vperp)

        # t2 / t1 should be (sm1/sm2)^0.6
        ratio = t2 / t1
        expected_ratio = (sm1 / sm2)**0.6
        assert torch.isclose(ratio, torch.tensor(expected_ratio), rtol=1e-6)


class TestSpecbroad:
    """Test SPECBROAD function (spectral broadening)."""

    def test_basic_calculation(self):
        """Test basic SPECBROAD calculation."""
        sm = torch.tensor(100.0)
        nu = torch.tensor(1.0)
        vperp = torch.tensor(100.0)

        bw = specbroad(sm, nu, vperp)

        # specbroad = 0.097 * nu^(-1.2) * sm^0.6 * (vperp/100)
        # = 0.097 * 1 * 100^0.6 * 1
        expected = 0.097 * 100.0**0.6
        assert torch.isclose(bw, torch.tensor(expected), rtol=1e-6)

    def test_inverse_relationship_to_scintime(self):
        """Test relationship: specbroad ~ vperp, scintime ~ 1/vperp."""
        sm = torch.tensor(100.0)
        nu = torch.tensor(1.0)
        vperp1 = torch.tensor(100.0)
        vperp2 = torch.tensor(200.0)

        bw1 = specbroad(sm, nu, vperp1)
        bw2 = specbroad(sm, nu, vperp2)
        t1 = scintime(sm, nu, vperp1)
        t2 = scintime(sm, nu, vperp2)

        # bw2/bw1 * t2/t1 should equal 1 (reciprocal relationship)
        product = (bw2 / bw1) * (t2 / t1)
        assert torch.isclose(product, torch.tensor(1.0), rtol=1e-6)


class TestThetaGalXgal:
    """Test THETA_GAL and THETA_XGAL functions (angular broadening)."""

    def test_theta_xgal(self):
        """Test THETA_XGAL calculation."""
        sm = torch.tensor(100.0)
        nu = torch.tensor(1.0)

        theta = theta_xgal(sm, nu)

        # theta_xgal = 128 * sm^0.6 * nu^(-2.2)
        expected = 128.0 * 100.0**0.6
        assert torch.isclose(theta, torch.tensor(expected), rtol=1e-6)

    def test_theta_gal(self):
        """Test THETA_GAL calculation."""
        sm = torch.tensor(100.0)
        nu = torch.tensor(1.0)

        theta = theta_gal(sm, nu)

        # theta_gal = 71 * sm^0.6 * nu^(-2.2)
        expected = 71.0 * 100.0**0.6
        assert torch.isclose(theta, torch.tensor(expected), rtol=1e-6)

    def test_ratio_xgal_to_gal(self):
        """Test ratio between extragalactic and galactic broadening."""
        sm = torch.tensor(100.0)
        nu = torch.tensor(1.0)

        theta_x = theta_xgal(sm, nu)
        theta_g = theta_gal(sm, nu)

        # Ratio should be 128/71
        ratio = theta_x / theta_g
        expected_ratio = 128.0 / 71.0
        assert torch.isclose(ratio, torch.tensor(expected_ratio), rtol=1e-6)

    def test_frequency_scaling(self):
        """Test frequency scaling: theta ~ nu^(-2.2)."""
        sm = torch.tensor(100.0)
        nu1 = torch.tensor(1.0)
        nu2 = torch.tensor(2.0)

        theta1 = theta_gal(sm, nu1)
        theta2 = theta_gal(sm, nu2)

        # theta2 / theta1 should be (nu1/nu2)^2.2 = 0.5^2.2
        ratio = theta2 / theta1
        expected_ratio = (nu1 / nu2)**2.2
        assert torch.isclose(ratio, torch.tensor(expected_ratio), rtol=1e-6)


class TestEmissionMeasure:
    """Test EMISSION_MEASURE function."""

    def test_basic_calculation(self):
        """Test basic emission measure calculation."""
        sm = torch.tensor(100.0)

        em = emission_measure(sm)

        # Should return a positive value
        assert em > 0

    def test_proportional_to_sm(self):
        """Test that EM is proportional to SM."""
        sm1 = torch.tensor(100.0)
        sm2 = torch.tensor(200.0)

        em1 = emission_measure(sm1)
        em2 = emission_measure(sm2)

        # em2 / em1 should be 2.0 (linear relationship)
        ratio = em2 / em1
        assert torch.isclose(ratio, torch.tensor(2.0), rtol=1e-6)


class TestTransitionFrequency:
    """Test TRANSITION_FREQUENCY function."""

    def test_basic_calculation(self):
        """Test basic transition frequency calculation."""
        sm = torch.tensor(100.0)
        smtau = torch.tensor(50.0)
        smtheta = torch.tensor(75.0)
        d = torch.tensor(5.0)

        nu_t = transition_frequency(sm, smtau, smtheta, d)

        # Should return a positive frequency in GHz
        assert nu_t > 0

    def test_scaling_with_sm(self):
        """Test scaling with SM: nu_t ~ SM^(6/17)."""
        sm1 = torch.tensor(100.0)
        sm2 = torch.tensor(200.0)
        smtau = torch.tensor(50.0)
        smtheta = torch.tensor(75.0)
        d = torch.tensor(5.0)

        # Keep ratios constant
        nu1 = transition_frequency(sm1, smtau, smtheta, d)
        nu2 = transition_frequency(sm2, smtau * 2, smtheta * 2, d)

        # nu2 / nu1 should be approximately 2^(6/17)
        ratio = nu2 / nu1
        expected_ratio = 2.0**(6.0 / 17.0)
        assert torch.isclose(ratio, torch.tensor(expected_ratio), rtol=1e-3)


class TestBatchedOperations:
    """Test that scattering functions work with batched tensors."""

    def test_tauiss_batched(self):
        """Test TAUISS with batched inputs."""
        d = torch.tensor([1.0, 2.0, 3.0])
        sm = torch.tensor([100.0, 100.0, 100.0])
        nu = torch.tensor([1.0, 1.0, 1.0])

        tau = tauiss(d, sm, nu)

        assert tau.shape == (3,)
        # Check linear scaling with distance
        assert torch.isclose(tau[1] / tau[0], torch.tensor(2.0), rtol=1e-6)
        assert torch.isclose(tau[2] / tau[0], torch.tensor(3.0), rtol=1e-6)

    def test_scintime_batched(self):
        """Test SCINTIME with batched inputs."""
        sm = torch.tensor([100.0, 200.0, 300.0])
        nu = torch.tensor([1.0, 1.0, 1.0])
        vperp = torch.tensor([100.0, 100.0, 100.0])

        t = scintime(sm, nu, vperp)

        assert t.shape == (3,)
        # Check that increasing SM decreases scintillation time
        assert t[1] < t[0]
        assert t[2] < t[1]
