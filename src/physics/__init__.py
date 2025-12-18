"""
@file __init__.py
@brief Physics module initialization
@author HPC-Code-Documenter
@date 2025

@details
The physics module implements aerothermal models and simulation workflows
for rocket reentry thermal analysis.

Aerothermal models (parametric_study.py):
- compute_aerothermal_parameters: Calculate $\\alpha(V)$ and $T_E(V)$
- parametric_velocity_study: Execute velocity sweep analysis
- Physical constants: air properties, material properties

Reentry profiles (reentry_profile.py):
- generate_reentry_profile: Create velocity vs time trajectories
- generate_altitude_profile: Derive altitude from velocity
- compute_dynamic_pressure: Calculate $q = \\frac{1}{2}\\rho V^2$

Key physical models:
- Reynolds number: $Re = \\frac{\\rho V L}{\\mu}$
- Recovery temperature: $T_{aw} = T_\\infty + r \\frac{V^2}{2 c_p}$
- Nusselt correlations for heat transfer coefficient

Supported reentry configurations:
- "nose-first": Classical reentry (nose leads)
- "tail-first": Retropropulsive landing (base leads)

@example
>>> from src.physics import (
...     parametric_velocity_study,
...     generate_reentry_profile
... )
>>> times, velocities = generate_reentry_profile(V_initial=7000)
>>> V_list, T_max, sols = parametric_velocity_study("mesh.msh", velocities)
"""

from src.physics.parametric_study import (
    parametric_velocity_study,
    compute_aerothermal_parameters,
    KAPPA_material,
)
from src.physics.reentry_profile import (
    generate_reentry_profile,
    generate_altitude_profile,
    compute_dynamic_pressure,
)

__all__ = [
    "parametric_velocity_study",
    "compute_aerothermal_parameters",
    "KAPPA_material",
    "generate_reentry_profile",
    "generate_altitude_profile",
    "compute_dynamic_pressure",
]
