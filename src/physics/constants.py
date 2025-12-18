"""
Physical constants for aerospace thermal analysis.

This module centralizes all physical constants used across the FEM thermal
analysis project. It serves as the single source of truth to avoid duplication
and ensure consistency.

All constants are documented with their physical meaning and units.
"""

# =============================================================================
# ATMOSPHERIC PROPERTIES (at ~30 km altitude)
# =============================================================================

RHO_inf = 0.02        # kg/m³ - air density
"""Air density at reference altitude (~30 km)."""

T_inf = 230.0         # K - ambient temperature
"""Ambient atmospheric temperature."""

MU_inf = 1.5e-5       # Pa·s - dynamic viscosity
"""Dynamic viscosity of air."""

k_air = 0.02          # W/(m·K) - thermal conductivity
"""Thermal conductivity of air."""

Pr = 0.71             # - Prandtl number
"""Prandtl number for air (dimensionless)."""

c_p = 1005.0          # J/(kg·K) - specific heat at constant pressure
"""Specific heat capacity of air at constant pressure."""

gamma = 1.4           # - ratio of specific heats
"""Ratio of specific heats (c_p / c_v) for air."""

r_recovery = 0.89     # - turbulent recovery factor
"""Recovery factor for turbulent boundary layer."""

# =============================================================================
# REFERENCE GEOMETRY
# =============================================================================

L_ref = 0.5           # m - characteristic length
"""Characteristic length for Reynolds number calculations."""

# =============================================================================
# MATERIAL PROPERTIES (default: Aluminum-like)
# =============================================================================

KAPPA_material = 160.0  # W/(m·K) - thermal conductivity
"""Thermal conductivity of structure material (default: aluminum)."""

# =============================================================================
# RADIATION CONSTANTS
# =============================================================================

STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m²·K⁴)
"""Stefan-Boltzmann constant for thermal radiation."""

EPSILON_carbon_carbon = 0.85        # - emissivity
"""Emissivity of carbon-carbon thermal protection system (TPS)."""

# =============================================================================
# ATMOSPHERIC MODELING
# =============================================================================

H_KARMAN = 120000.0   # m - Kármán line (boundary of space)
"""Altitude defining the boundary of space."""

H_SCALE = 8500.0      # m - atmospheric scale height
"""Exponential scale height for atmospheric density decay."""
