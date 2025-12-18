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


RHO_inf = 0.02
T_inf = 230.0
MU_inf = 1.5e-5
k_air = 0.02
Pr = 0.71
c_p = 1005.0
gamma = 1.4

r_recovery = 0.89

KAPPA_material = 160.0

L_ref = 0.5


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


def compute_aerothermal_parameters(V: float) -> Tuple[float, float]:
    """
    @brief Compute velocity-dependent aerothermal parameters.

    @details
    Calculates the convection coefficient $\\alpha$ and external temperature
    $u_E$ for Robin boundary conditions at a given velocity.

    Uses the recovery temperature as the driving temperature for heat
    transfer (more accurate than stagnation temperature for boundary layers).

    @param V: Rocket velocity [m/s]

    @return Tuple containing:
        - alpha: Convection coefficient [W/(m^2.K)]
        - u_E: External (recovery) temperature [K]
    """
    u_E = compute_recovery_temperature(V, T_inf, r_recovery, gamma, c_p)

    alpha = compute_heat_transfer_coefficient(V, RHO_inf, MU_inf, k_air, L_ref, Pr)

    return alpha, u_E


def parametric_velocity_study(mesh_file: str,
                              velocity_range: np.ndarray,
                              base_temperature: float = 300.0,
                              mode: str = "tail-first") -> Tuple[List[float], List[float], List[np.ndarray]]:
    """
    @brief Perform parametric thermal study over a range of velocities.

    @details
    For each velocity in the specified range:
    1. Compute aerothermal parameters $\\alpha(V)$ and $T_E(V)$
    2. Assemble the FEM system with these boundary conditions
    3. Apply Dirichlet conditions (if applicable)
    4. Solve the linear system
    5. Extract maximum temperature

    The study supports two reentry configurations:
    - "nose-first": External surface receives heating, base at fixed temperature
    - "tail-first": Base receives heating (in attack), flanks/nose in wake

    @param mesh_file: Path to the GMSH mesh file (.msh)
    @param velocity_range: NumPy array of velocities to analyze [m/s]
    @param base_temperature: Reference temperature for Dirichlet BC [K]
    @param mode: Reentry mode - "nose-first" or "tail-first"

    @return Tuple containing:
        - velocities: List of analyzed velocities
        - T_max_list: List of maximum temperatures for each velocity
        - solutions: List of complete temperature field arrays

    @example
    >>> velocities = np.linspace(1000, 7000, 10)
    >>> V_list, T_max, sols = parametric_velocity_study(
    ...     "rocket.msh", velocities, mode="tail-first"
    ... )
    """
    print("=" * 70)
    print("ETUDE PARAMETRIQUE EN VITESSE")
    print("=" * 70)
    print(f"Mode de rentree: {mode.upper()}")
    print(f"Lecture du maillage: {mesh_file}")

    mesh = read_gmsh_mesh(mesh_file)
    node_to_dof, num_dofs = create_node_mapping(mesh)

    print(f"Maillage charge: {len(mesh.nodes)} noeuds, {num_dofs} DOFs")
    print(f"Plage de vitesses: {velocity_range[0]:.0f} - {velocity_range[-1]:.0f} m/s")
    print(f"Nombre de cas: {len(velocity_range)}")

    if mode == "tail-first":
        print("\nConditions aux limites (TAIL-FIRST):")
        print("  - BASE (Physical ID 1): Robin - Flux entrant (zone d'attaque)")
        print("  - FLANCS: Neumann (isoles, dans le sillage)")
        print("  - OGIVE: Neumann (protegee)")
    else:
        print("\nConditions aux limites (NOSE-FIRST):")
        print("  - SURFACE EXT (Physical ID 1): Robin - Flux entrant")
        print("  - BASE (Physical ID 2): Dirichlet - T fixee")

    print("=" * 70)

    velocities = []
    T_max_list = []
    solutions = []

    for idx, V in enumerate(velocity_range):
        print(f"\n[Cas {idx+1}/{len(velocity_range)}] Vitesse V = {V:.0f} m/s")
        print("-" * 70)

        alpha, u_E = compute_aerothermal_parameters(V)
        print(f"  alpha(V) = {alpha:.2f} W/m^2.K")
        print(f"  u_E(V) = {u_E:.2f} K")

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
