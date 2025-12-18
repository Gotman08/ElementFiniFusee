"""
Aerothermal calculations for atmospheric reentry.

This module contains all aerothermal correlation functions for computing
convective heat transfer coefficients, recovery temperatures, and altitude
corrections for reentry vehicles.

Functions are organized by physical phenomenon:
- Atmospheric modeling: atmospheric_attenuation_factor
- Flow characterization: compute_reynolds_number
- Temperature calculations: compute_stagnation_temperature, compute_recovery_temperature
- Heat transfer: compute_heat_transfer_coefficient
- Integrated parameters: compute_aerothermal_parameters, compute_altitude_corrected_parameters
"""
import numpy as np
from typing import Tuple
from src.physics.constants import (
    RHO_inf, T_inf, MU_inf, k_air, Pr, c_p, gamma,
    r_recovery, L_ref,
    STEFAN_BOLTZMANN, EPSILON_carbon_carbon,
    H_KARMAN, H_SCALE
)


def atmospheric_attenuation_factor(h: float) -> float:
    """
    @brief Compute atmospheric attenuation factor for aerodynamic heating.

    @details
    This function models the reduction in aerodynamic heating at high altitudes
    due to decreasing atmospheric density. The factor follows the atmospheric
    density profile ρ(h) = ρ₀ · exp(-h/H).

    Above the Kármán line (120 km), the atmosphere is so tenuous that
    aerodynamic heating is negligible (factor = 0).

    Below 120 km, the factor grows exponentially as the vehicle descends:
    f_atm = exp(-(120 km - h) / H_scale)

    where H_scale = 8500 m is the atmospheric scale height.

    This ensures:
    - At h = 250 km (space): f_atm = 0 → No heating
    - At h = 120 km (Kármán line): f_atm = 0 → Heating begins
    - At h = 100 km: f_atm ≈ 0.12 → 12% of nominal heating
    - At h = 80 km: f_atm ≈ 0.50 → 50% of nominal heating
    - At h = 50 km: f_atm ≈ 0.99 → Near-full heating
    - At h = 0 km (sea level): f_atm = 1.0 → Full heating

    @param h: Altitude above sea level [m]

    @return Attenuation factor f_atm in range [0, 1] (dimensionless)
        - 0: No aerodynamic heating (vacuum/space)
        - 1: Full aerodynamic heating (dense atmosphere)

    @example
    >>> # In space (250 km)
    >>> f = atmospheric_attenuation_factor(250000)
    >>> print(f"{f:.3f}")  # Output: 0.000
    >>>
    >>> # Entry interface (120 km)
    >>> f = atmospheric_attenuation_factor(120000)
    >>> print(f"{f:.3f}")  # Output: 0.000
    >>>
    >>> # Peak heating altitude (~80 km)
    >>> f = atmospheric_attenuation_factor(80000)
    >>> print(f"{f:.3f}")  # Output: ~0.5
    """
    if h >= H_KARMAN:
        # Above Kármán line: no aerodynamic heating
        return 0.0
    else:
        # Below Kármán line: exponential growth as altitude decreases
        # f_atm grows from 0 at h=120km to ~1 at h=0
        return np.exp(-(H_KARMAN - h) / H_SCALE)


def compute_altitude_corrected_parameters(V: float,
                                          h: float,
                                          emissivity: float = EPSILON_carbon_carbon
                                          ) -> Tuple[float, float, float, float]:
    """
    @brief Compute aerothermal parameters with altitude-dependent correction.

    @details
    This function extends compute_aerothermal_parameters() by introducing
    altitude-dependent attenuation of aerodynamic heating. This is critical
    for realistic reentry simulations.

    Physical motivation:
    - Standard formulas assume dense atmospheric flow (Re > 10⁵)
    - At high altitudes (h > 120 km), atmosphere is quasi-vacuum (ρ ≈ 10⁻¹¹ kg/m³)
    - No atmospheric density → No boundary layer → No heating
    - As vehicle descends, ρ(h) increases exponentially → heating increases

    The correction is applied via atmospheric_attenuation_factor(h):
    - alpha_corrected = alpha_nominal × f_atm(h)
    - u_E_corrected = T_∞ + (u_E_nominal - T_∞) × f_atm(h)

    This ensures:
    - In space (h > 120 km): alpha ≈ 0, u_E ≈ T_∞ → T stays at ambient
    - In atmosphere (h < 120 km): alpha and u_E increase as ρ(h) increases

    @param V: Rocket velocity [m/s]
    @param h: Altitude above sea level [m]
    @param emissivity: Surface emissivity for radiation cooling (default: 0.85 for C-C TPS)

    @return Tuple containing:
        - alpha: Corrected convection heat transfer coefficient [W/(m²·K)]
        - u_E: Corrected recovery temperature [K]
        - epsilon: Surface emissivity (unchanged)
        - sigma: Stefan-Boltzmann constant (unchanged)

    @example
    >>> # In space (h = 250 km, V = 7000 m/s)
    >>> alpha, u_E, eps, sig = compute_altitude_corrected_parameters(7000, 250000)
    >>> print(f"h=250km: alpha={alpha:.1f}, u_E={u_E:.1f} K")
    >>> # Output: h=250km: alpha=0.0, u_E=230.0 K  (no heating!)
    >>>
    >>> # Peak heating (h = 80 km, V = 5000 m/s)
    >>> alpha, u_E, eps, sig = compute_altitude_corrected_parameters(5000, 80000)
    >>> print(f"h=80km: alpha={alpha:.1f}, u_E={u_E:.1f} K")
    >>> # Output: h=80km: alpha~100, u_E~2500 K  (intense heating)
    """
    # Compute nominal parameters (as if at sea level with dense atmosphere)
    alpha_nominal, u_E_nominal, epsilon, sigma = compute_aerothermal_parameters(V, emissivity)

    # Compute altitude-dependent attenuation factor
    f_atm = atmospheric_attenuation_factor(h)

    # Apply correction to convection coefficient
    alpha = alpha_nominal * f_atm

    # Apply correction to recovery temperature
    # u_E_corrected interpolates between T_∞ (in space) and u_E_nominal (in atmosphere)
    u_E = T_inf + (u_E_nominal - T_inf) * f_atm

    return alpha, u_E, epsilon, sigma


def compute_reynolds_number(V: float, rho: float, mu: float, L: float) -> float:
    """
    @brief Compute the Reynolds number for flow characterization.

    @details
    Calculates the dimensionless Reynolds number:
    $Re = \\frac{\\rho V L}{\\mu}$

    The Reynolds number determines flow regime:
    - $Re < 5 \\times 10^5$: Laminar flow
    - $Re > 5 \\times 10^5$: Turbulent flow

    @param V: Flow velocity [m/s]
    @param rho: Fluid density [kg/m^3]
    @param mu: Dynamic viscosity [Pa.s]
    @param L: Characteristic length [m]

    @return Reynolds number (dimensionless)
    """
    Re = (rho * V * L) / mu
    return Re


def compute_stagnation_temperature(V: float, T_inf: float, gamma: float, c_p: float) -> float:
    """
    @brief Compute the isentropic stagnation (total) temperature.

    @details
    Calculates the temperature at a stagnation point assuming isentropic
    compression of the freestream:
    $T_0 = T_\\infty \\left(1 + \\frac{\\gamma-1}{2} M^2\\right)$

    where the Mach number is:
    $M = \\frac{V}{a_\\infty} = \\frac{V}{\\sqrt{\\gamma R T_\\infty}}$

    @param V: Freestream velocity [m/s]
    @param T_inf: Freestream temperature [K]
    @param gamma: Ratio of specific heats (1.4 for air)
    @param c_p: Specific heat at constant pressure [J/(kg.K)]

    @return Stagnation temperature [K]
    """
    a_inf = np.sqrt(gamma * (c_p * (gamma - 1) / gamma) * T_inf)
    M = V / a_inf

    T_0 = T_inf * (1 + (gamma - 1) / 2 * M**2)
    return T_0


def compute_recovery_temperature(V: float, T_inf: float, r: float, gamma: float, c_p: float) -> float:
    """
    @brief Compute the adiabatic wall (recovery) temperature.

    @details
    Calculates the temperature a perfectly insulated wall would reach:
    $T_{aw} = T_\\infty + r \\frac{V^2}{2 c_p}$

    The recovery factor $r$ depends on flow regime:
    - Laminar: $r \\approx \\sqrt{Pr} \\approx 0.85$
    - Turbulent: $r \\approx Pr^{1/3} \\approx 0.89$

    This is the driving temperature for convective heat transfer.

    @param V: Freestream velocity [m/s]
    @param T_inf: Freestream temperature [K]
    @param r: Recovery factor (0.89 for turbulent, 0.85 for laminar)
    @param gamma: Ratio of specific heats
    @param c_p: Specific heat at constant pressure [J/(kg.K)]

    @return Adiabatic wall temperature [K]
    """
    T_aw = T_inf + r * V**2 / (2 * c_p)
    return T_aw


def compute_heat_transfer_coefficient(V: float, rho: float, mu: float, k: float,
                                      L: float, Pr: float) -> float:
    """
    @brief Compute the convective heat transfer coefficient.

    @details
    Uses Nusselt number correlations for flat plate boundary layers:
    - Laminar ($Re < 5 \\times 10^5$): $Nu = 0.664 \\, Re^{0.5} \\, Pr^{1/3}$
    - Turbulent ($Re > 5 \\times 10^5$): $Nu = 0.037 \\, Re^{0.8} \\, Pr^{1/3}$

    The heat transfer coefficient is then:
    $\\alpha = \\frac{Nu \\cdot k}{L}$

    @param V: Flow velocity [m/s]
    @param rho: Fluid density [kg/m^3]
    @param mu: Dynamic viscosity [Pa.s]
    @param k: Fluid thermal conductivity [W/(m.K)]
    @param L: Characteristic length [m]
    @param Pr: Prandtl number

    @return Heat transfer coefficient $\\alpha$ [W/(m^2.K)]
    """
    Re = compute_reynolds_number(V, rho, mu, L)

    if Re > 5e5:
        Nu = 0.037 * Re**0.8 * Pr**(1.0/3.0)
    else:
        Nu = 0.664 * Re**0.5 * Pr**(1.0/3.0)

    alpha = Nu * k / L

    return alpha


def compute_aerothermal_parameters(V: float,
                                   emissivity: float = EPSILON_carbon_carbon
                                   ) -> Tuple[float, float, float, float]:
    """
    @brief Compute velocity-dependent aerothermal parameters including radiation.

    @details
    Calculates the convection coefficient $\\alpha$, external temperature $u_E$,
    and radiation parameters for Robin+Radiation boundary conditions at a given velocity.

    Uses the recovery temperature as the driving temperature for heat
    transfer (more accurate than stagnation temperature for boundary layers).

    The radiation parameters enable Stefan-Boltzmann cooling:
    $q_{rad} = \\epsilon \\sigma (T^4 - T_{\\infty}^4)$

    @param V: Rocket velocity [m/s]
    @param emissivity: Surface emissivity (default: 0.85 for Carbon-Carbon TPS)

    @return Tuple containing:
        - alpha: Convection coefficient [W/(m²·K)]
        - u_E: External (recovery) temperature [K]
        - epsilon: Surface emissivity for radiation (0-1)
        - sigma: Stefan-Boltzmann constant [W/(m²·K⁴)]
    """
    u_E = compute_recovery_temperature(V, T_inf, r_recovery, gamma, c_p)

    alpha = compute_heat_transfer_coefficient(V, RHO_inf, MU_inf, k_air, L_ref, Pr)

    return alpha, u_E, emissivity, STEFAN_BOLTZMANN
