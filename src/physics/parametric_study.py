"""
@file parametric_study.py
@brief Parametric velocity study for rocket reentry thermal analysis
@author HPC-Code-Documenter
@date 2025

@details
This module implements aerothermal models and parametric studies for
rocket reentry scenarios. It computes velocity-dependent boundary conditions
and solves the thermal problem across a range of entry velocities.

Physical models implemented:
- Reynolds number: $Re = \\frac{\\rho V L}{\\mu}$
- Stagnation temperature: $T_0 = T_\\infty \\left(1 + \\frac{\\gamma-1}{2} M^2\\right)$
- Recovery temperature: $T_{aw} = T_\\infty + r \\frac{V^2}{2 c_p}$
- Heat transfer coefficient via Nusselt correlations

Supports two reentry modes:
- "nose-first": Classical reentry with nose cone leading
- "tail-first": Retropropulsive reentry with base leading (SpaceX-style)

Reference conditions:
- Altitude: ~30 km
- Air density: 0.02 kg/m^3
- Ambient temperature: 230 K
"""
import numpy as np
from typing import Tuple, List
from src.mesh.mesh_reader import Mesh, read_gmsh_mesh, create_node_mapping
from src.core.assembly import assemble_global_system
from src.core.boundary_conditions import apply_dirichlet_conditions
from src.core.solver import solve_linear_system
from src.core.nonlinear_solver import picard_iteration
from src.physics.constants import KAPPA_material, EPSILON_carbon_carbon
from src.physics.aerothermal import (
    atmospheric_attenuation_factor,
    compute_altitude_corrected_parameters,
    compute_reynolds_number,
    compute_stagnation_temperature,
    compute_recovery_temperature,
    compute_heat_transfer_coefficient,
    compute_aerothermal_parameters,
)


# Aerothermal functions have been moved to src/physics/aerothermal.py
# Import them from there instead of defining them here


def parametric_velocity_study(mesh_file: str,
                              velocity_range: np.ndarray,
                              base_temperature: float = 300.0,
                              mode: str = "tail-first",
                              include_radiation: bool = True,
                              emissivity: float = EPSILON_carbon_carbon) -> Tuple[List[float], List[float], List[np.ndarray]]:
    """
    @brief Perform parametric thermal study over a range of velocities.

    @details
    For each velocity in the specified range:
    1. Compute aerothermal parameters $\\alpha(V)$, $T_E(V)$, and radiation parameters
    2. Assemble the FEM system with Robin+Radiation boundary conditions
    3. Apply Dirichlet conditions (if applicable)
    4. Solve the nonlinear system (if radiation enabled) or linear system
    5. Extract maximum temperature

    The study supports two reentry configurations:
    - "nose-first": External surface receives heating, base at fixed temperature
    - "tail-first": Base receives heating (in attack), flanks/nose in wake

    **NEW**: Includes Stefan-Boltzmann radiation cooling by default.
    This dramatically reduces temperatures at high velocities by adding
    radiative heat dissipation: $q_{rad} = \\epsilon \\sigma (T^4 - T_{\\infty}^4)$

    @param mesh_file: Path to the GMSH mesh file (.msh)
    @param velocity_range: NumPy array of velocities to analyze [m/s]
    @param base_temperature: Reference temperature for Dirichlet BC [K]
    @param mode: Reentry mode - "nose-first" or "tail-first"
    @param include_radiation: Enable Stefan-Boltzmann radiation cooling (default: True)
    @param emissivity: Surface emissivity for radiation (default: 0.85 for Carbon-Carbon)

    @return Tuple containing:
        - velocities: List of analyzed velocities
        - T_max_list: List of maximum temperatures for each velocity
        - solutions: List of complete temperature field arrays

    @example
    >>> velocities = np.linspace(1000, 7000, 10)
    >>> # With radiation (default, realistic temperatures)
    >>> V_list, T_max, sols = parametric_velocity_study(
    ...     "rocket.msh", velocities, mode="tail-first", include_radiation=True
    ... )
    >>> # Without radiation (for comparison)
    >>> V_list_no_rad, T_max_no_rad, _ = parametric_velocity_study(
    ...     "rocket.msh", velocities, mode="tail-first", include_radiation=False
    ... )
    """
    print("=" * 70)
    print("ETUDE PARAMETRIQUE EN VITESSE")
    print("=" * 70)
    print(f"Mode de rentree: {mode.upper()}")
    print(f"Physique de radiation: {'ACTIVEE' if include_radiation else 'DESACTIVEE'}")
    if include_radiation:
        print(f"  - Emissivite: {emissivity:.2f}")
        print(f"  - Methode: Iteration de Picard (nonlineaire)")
    print(f"Lecture du maillage: {mesh_file}")

    mesh = read_gmsh_mesh(mesh_file)
    node_to_dof, num_dofs = create_node_mapping(mesh)

    print(f"Maillage charge: {len(mesh.nodes)} noeuds, {num_dofs} DOFs")
    print(f"Plage de vitesses: {velocity_range[0]:.0f} - {velocity_range[-1]:.0f} m/s")
    print(f"Nombre de cas: {len(velocity_range)}")

    if mode == "tail-first":
        print("\nConditions aux limites (TAIL-FIRST):")
        print("  - BASE (Physical ID 1): Robin + Radiation - Flux entrant (zone d'attaque)")
        print("  - FLANCS: Neumann (isoles, dans le sillage)")
        print("  - OGIVE: Neumann (protegee)")
    else:
        print("\nConditions aux limites (NOSE-FIRST):")
        print("  - SURFACE EXT (Physical ID 1): Robin + Radiation - Flux entrant")
        print("  - BASE (Physical ID 2): Dirichlet - T fixee")

    print("=" * 70)

    velocities = []
    T_max_list = []
    solutions = []

    for idx, V in enumerate(velocity_range):
        print(f"\n[Cas {idx+1}/{len(velocity_range)}] Vitesse V = {V:.0f} m/s")
        print("-" * 70)

        # Compute aerothermal parameters at nominal altitude (~30 km)
        # Note: For full trajectory with varying altitude, use compute_altitude_corrected_parameters()
        alpha, u_E, epsilon, sigma = compute_aerothermal_parameters(V, emissivity)
        print(f"  alpha(V) = {alpha:.2f} W/m^2.K")
        print(f"  u_E(V) = {u_E:.2f} K")
        if include_radiation:
            print(f"  epsilon = {epsilon:.2f}, sigma = {sigma:.3e} W/m^2.K^4")

        # Setup boundary conditions
        if mode == "tail-first":
            robin_boundaries = {
                1: (alpha, u_E)
            }
            dirichlet_boundaries = {}

        else:
            robin_boundaries = {
                1: (alpha, u_E)
            }
            dirichlet_boundaries = {
                2: base_temperature
            }

        # Solve thermal problem (nonlinear with radiation or linear without)
        if include_radiation:
            # Setup radiation boundaries
            radiation_boundaries = {
                1: (epsilon, T_inf, sigma)
            }

            # Solve nonlinear problem with Picard iteration
            U, info = picard_iteration(
                mesh, node_to_dof, KAPPA_material,
                robin_boundaries, radiation_boundaries,
                dirichlet_boundaries=dirichlet_boundaries if dirichlet_boundaries else None,
                max_iter=100,
                tol_abs=10.0,  # Relaxed tolerance: 10 K
                tol_rel=1e-3,  # Relaxed relative tolerance: 0.1%
                relaxation=0.3,  # Strong under-relaxation for stability
                verbose=True  # Enable verbose logging to see convergence
            )

            # Log convergence info
            if info['converged']:
                print(f"  [Convergence] {info['iterations']} iterations, "
                      f"residual={info['final_residual']:.2e} K")
            else:
                print(f"  [WARNING] Non-convergence après {info['iterations']} iterations, "
                      f"residual={info['final_residual']:.2e} K")

        else:
            # Linear solve (convection only, no radiation)
            A, F = assemble_global_system(mesh, node_to_dof, KAPPA_material, robin_boundaries)

            if dirichlet_boundaries:
                A_bc, F_bc = apply_dirichlet_conditions(A, F, mesh, node_to_dof, dirichlet_boundaries)
            else:
                A_bc, F_bc = A.copy(), F.copy()

            U = solve_linear_system(A_bc, F_bc, method='direct')

        T_max = np.max(U)
        T_min = np.min(U)

        if mode == "tail-first":
            print(f"  [OK] T_max = {T_max:.2f} K (base), T_min = {T_min:.2f} K (ogive)")
        else:
            print(f"  [OK] Temperature maximale: T_max = {T_max:.2f} K")

        velocities.append(V)
        T_max_list.append(T_max)
        solutions.append(U)

    print("\n" + "=" * 70)
    print("ETUDE PARAMETRIQUE TERMINEE")
    print("=" * 70)

    return velocities, T_max_list, solutions


if __name__ == '__main__':
    print("Module d'etude parametrique en vitesse")
    print("Utiliser dans le script principal avec un maillage .msh")
