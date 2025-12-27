"""
Scattering functions for NE2001 Model

This module implements scattering time scales and related quantities
from Cordes & Rickett (1998, ApJ).

Reference:
    scattering98.f (FORTRAN source)
"""

import torch
import math


def tauiss(
    d: torch.Tensor,
    sm: torch.Tensor,
    nu: torch.Tensor
) -> torch.Tensor:
    """
    Calculate pulse broadening time (scintillation timescale).

    Calculates the pulse broadening time in seconds from distance,
    scattering measure, and radio frequency.

    Args:
        d: Pulsar distance (kpc)
        sm: Scattering measure (kpc m^{-20/3})
        nu: Radio frequency (GHz)

    Returns:
        Pulse broadening time (seconds)

    Note:
        Direct port from FORTRAN code scattering98.f:
        TAUISS = 1000. * (sm / 292.)**1.2 * d * nu**(-4.4)

        The factor of 1000 in FORTRAN converts from seconds to milliseconds.
        We return seconds directly for consistency with modern units.

    Examples:
        >>> d = torch.tensor(1.0)    # 1 kpc
        >>> sm = torch.tensor(100.0) # 100 kpc m^{-20/3}
        >>> nu = torch.tensor(1.0)   # 1 GHz
        >>> tau = tauiss(d, sm, nu)
        >>> print(f"Pulse broadening: {tau:.3f} s")
    """
    # Original formula with conversion factor removed
    # tauiss_ms = 1000. * (sm / 292.)**1.2 * d * nu**(-4.4)
    # Return in seconds instead of milliseconds
    tauiss_s = 1.0 * (sm / 292.0)**1.2 * d * nu**(-4.4)
    return tauiss_s


def scintbw(
    d: torch.Tensor,
    sm: torch.Tensor,
    nu: torch.Tensor
) -> torch.Tensor:
    """
    Calculate scintillation bandwidth.

    Calculates the scintillation bandwidth in kHz from distance,
    scattering measure, and radio frequency.

    Args:
        d: Pulsar distance (kpc)
        sm: Scattering measure (kpc m^{-20/3})
        nu: Radio frequency (GHz)

    Returns:
        Scintillation bandwidth (kHz)

    Note:
        Direct port from FORTRAN code scattering98.f:
        c1 = 1.16  (for uniform, Kolmogorov medium)
        tauiss_ms = 1000. * (sm / 292.)**1.2 * d * nu**(-4.4)
        scintbw = c1 / (2. * pi * tauiss_ms)  [kHz]

    Examples:
        >>> d = torch.tensor(1.0)    # 1 kpc
        >>> sm = torch.tensor(100.0) # 100 kpc m^{-20/3}
        >>> nu = torch.tensor(1.0)   # 1 GHz
        >>> bw = scintbw(d, sm, nu)
        >>> print(f"Scintillation bandwidth: {bw:.3f} kHz")
    """
    # Constant for uniform Kolmogorov medium
    c1 = 1.16

    # Calculate pulse broadening time in milliseconds (as in FORTRAN)
    tauiss_ms = 1000.0 * (sm / 292.0)**1.2 * d * nu**(-4.4)

    # Calculate scintillation bandwidth in kHz
    scintbw_khz = c1 / (2.0 * math.pi * tauiss_ms)

    return scintbw_khz


def scintime(
    sm: torch.Tensor,
    nu: torch.Tensor,
    vperp: torch.Tensor
) -> torch.Tensor:
    """
    Calculate scintillation timescale.

    Calculates the scintillation time for given scattering measure,
    frequency, and transverse velocity.

    Args:
        sm: Scattering measure (kpc m^{-20/3})
        nu: Radio frequency (GHz)
        vperp: Pulsar transverse speed (km/s)

    Returns:
        Scintillation time (seconds)

    Note:
        Direct port from FORTRAN code scattering98.f:
        scintime = 3.3 * nu**1.2 * sm**(-0.6) * (100./vperp)

        Usage: Should be called with sm = smtau for appropriate
        line of sight weighting.

        Reference: eqn (46) of Cordes & Lazio 1991, ApJ, 376, 123.
        The coefficient was updated from 2.3 to 3.3 for consistency
        with Cordes & Rickett (1998).

    Examples:
        >>> sm = torch.tensor(100.0)  # 100 kpc m^{-20/3}
        >>> nu = torch.tensor(1.0)    # 1 GHz
        >>> vperp = torch.tensor(100.0) # 100 km/s
        >>> t_scint = scintime(sm, nu, vperp)
        >>> print(f"Scintillation time: {t_scint:.3f} s")
    """
    # Direct port from FORTRAN
    scintime_s = 3.3 * nu**1.2 * sm**(-0.6) * (100.0 / vperp)

    return scintime_s


def specbroad(
    sm: torch.Tensor,
    nu: torch.Tensor,
    vperp: torch.Tensor
) -> torch.Tensor:
    """
    Calculate spectral broadening bandwidth.

    Calculates the bandwidth of spectral broadening for given
    scattering measure, frequency, and transverse velocity.

    Args:
        sm: Scattering measure (kpc m^{-20/3})
        nu: Radio frequency (GHz)
        vperp: Pulsar transverse speed (km/s)

    Returns:
        Spectral broadening bandwidth (Hz)

    Note:
        Direct port from FORTRAN code scattering98.f:
        specbroad = 0.097 * nu**(-1.2) * sm**0.6 * (vperp/100.)

        The coefficient was changed from 0.14 Hz (Cordes & Lazio 1991)
        to 0.097 to conform with SCINTIME and Cordes & Rickett (1998).

        Usage: Should be called with sm = smtau for appropriate
        line of sight weighting.

        Reference: eqn (47) of Cordes & Lazio 1991, ApJ, 376, 123.

    Examples:
        >>> sm = torch.tensor(100.0)  # 100 kpc m^{-20/3}
        >>> nu = torch.tensor(1.0)    # 1 GHz
        >>> vperp = torch.tensor(100.0) # 100 km/s
        >>> bw = specbroad(sm, nu, vperp)
        >>> print(f"Spectral broadening: {bw:.3f} Hz")
    """
    # Direct port from FORTRAN
    specbroad_hz = 0.097 * nu**(-1.2) * sm**0.6 * (vperp / 100.0)

    return specbroad_hz


def theta_xgal(
    sm: torch.Tensor,
    nu: torch.Tensor
) -> torch.Tensor:
    """
    Calculate angular broadening for extragalactic source.

    Calculates angular broadening for an extragalactic source
    of plane waves.

    Args:
        sm: Scattering measure (kpc m^{-20/3})
        nu: Radio frequency (GHz)

    Returns:
        Angular broadening FWHM (milliarcseconds)

    Note:
        Direct port from FORTRAN code scattering98.f:
        theta_xgal = 128. * sm**0.6 * nu**(-2.2)

    Examples:
        >>> sm = torch.tensor(100.0)  # 100 kpc m^{-20/3}
        >>> nu = torch.tensor(1.0)    # 1 GHz
        >>> theta = theta_xgal(sm, nu)
        >>> print(f"Angular broadening: {theta:.3f} mas")
    """
    theta_mas = 128.0 * sm**0.6 * nu**(-2.2)
    return theta_mas


def theta_gal(
    sm: torch.Tensor,
    nu: torch.Tensor
) -> torch.Tensor:
    """
    Calculate angular broadening for galactic source.

    Calculates angular broadening for a galactic source
    of spherical waves.

    Args:
        sm: Scattering measure (kpc m^{-20/3})
        nu: Radio frequency (GHz)

    Returns:
        Angular broadening FWHM (milliarcseconds)

    Note:
        Direct port from FORTRAN code scattering98.f:
        theta_gal = 71. * sm**0.6 * nu**(-2.2)

    Examples:
        >>> sm = torch.tensor(100.0)  # 100 kpc m^{-20/3}
        >>> nu = torch.tensor(1.0)    # 1 GHz
        >>> theta = theta_gal(sm, nu)
        >>> print(f"Angular broadening: {theta:.3f} mas")
    """
    theta_mas = 71.0 * sm**0.6 * nu**(-2.2)
    return theta_mas


def emission_measure(
    sm: torch.Tensor
) -> torch.Tensor:
    """
    Calculate emission measure from scattering measure.

    Calculates the emission measure from the scattering measure
    using an assumed outer scale and spectral index of the
    wavenumber spectrum.

    Args:
        sm: Scattering measure (kpc m^{-20/3})

    Returns:
        Emission measure (pc cm^{-6})

    Note:
        For a wavenumber spectrum P_n(q) = q^{-alpha} from q_0 to q_1,
        the mean square electron density is approximately:

        <n_e^2> ≈ 4π * [C_n^2 / (alpha - 3)] * q_0^{3 - alpha}

        assuming (q_0 / q_1)^{3-alpha} >> 1.

        Parameters:
            router = 1 pc (outer scale)
            alpha = 11/3 (spectral index for Kolmogorov turbulence)
            pc = 3.086e18 cm

        Jim Cordes, 18 Dec 1989

    Examples:
        >>> sm = torch.tensor(100.0)  # 100 kpc m^{-20/3}
        >>> em = emission_measure(sm)
        >>> print(f"Emission measure: {em:.3f} pc cm^-6")
    """
    # Parameters from FORTRAN
    router = 1.0  # outer scale = 1 pc
    pc = 3.086e18  # pc in cm
    alpha = 11.0 / 3.0  # = 3.6666667
    pi = math.pi

    # Direct port from FORTRAN
    em = (sm *
          ((4.0 * pi * 1000.0) / (alpha - 3.0)) *
          (router * pc / (2.0 * pi))**(alpha - 3.0) *
          (0.01)**(20.0 / 3.0))

    return em


def transition_frequency(
    sm: torch.Tensor,
    smtau: torch.Tensor,
    smtheta: torch.Tensor,
    dintegrate: torch.Tensor
) -> torch.Tensor:
    """
    Calculate transition frequency between weak and strong scattering.

    Returns the transition frequency between weak and strong scattering
    regimes.

    Args:
        sm: Scattering measure integral[C_n^2] (kpc m^{-20/3})
        smtau: Weighted SM integral[(s/D)(1-s/D) C_n^2] (kpc m^{-20/3})
        smtheta: Weighted SM integral[(1-s/D) C_n^2] (kpc m^{-20/3})
        dintegrate: Distance used to integrate C_n^2 (kpc)

    Returns:
        Transition frequency (GHz)

    Note:
        Direct port from FORTRAN code scattering98.f.

        Formula:
        ν_t = 318 GHz * ξ^{10/17} * SM^{6/17} * D_eff^{5/17}

        where:
        - D_eff = effective path length through medium
        - D_eff = ∫ ds s C_n^2 / ∫ ds C_n^2
        - ξ = (2π)^{-1/2} = 0.3989 (Fresnel scale definition factor)

        D_eff can be calculated using:
        D_eff = dintegrate * (sm - smtau/6 - smtheta/3) / sm

        Jim Cordes, 28 March 2001

    Examples:
        >>> sm = torch.tensor(100.0)
        >>> smtau = torch.tensor(50.0)
        >>> smtheta = torch.tensor(75.0)
        >>> d = torch.tensor(5.0)
        >>> nu_t = transition_frequency(sm, smtau, smtheta, d)
        >>> print(f"Transition frequency: {nu_t:.3f} GHz")
    """
    # Parameters from FORTRAN
    xi = 0.3989  # (2*pi)^{-1/2} = Fresnel scale definition factor
    coefficient = 318.0  # GHz; see NE2001 paper

    # Calculate effective distance
    deff = (dintegrate * (sm - smtau / 6.0 - smtheta / 3.0)) / sm

    # Calculate transition frequency
    nu_trans = (coefficient *
                xi**(10.0 / 17.0) *
                sm**(6.0 / 17.0) *
                deff**(5.0 / 17.0))

    return nu_trans
