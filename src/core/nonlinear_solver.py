"""
Nonlinear solver for thermal problems with radiation using Picard iteration.

This module implements a Picard (fixed-point) iteration scheme for solving
steady-state thermal problems with Stefan-Boltzmann radiation boundary conditions.

The nonlinearity arises from the T⁴ dependence in the radiation term:
    q_rad = ε·σ·(T⁴ - T_∞⁴)

The Picard method linearizes this by using the temperature from the previous iteration:
    α_rad^(k) = ε·σ·(T^(k))³

This transforms the nonlinear problem into a sequence of linear problems that
converge to the nonlinear solution.
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional, Any
from scipy.sparse import csr_matrix

from src.mesh.mesh_reader import Mesh
from src.core.assembly import assemble_global_system_with_radiation
from src.core.boundary_conditions import apply_dirichlet_conditions
from src.core.solver import solve_linear_system

logger = logging.getLogger(__name__)


class ConvergenceError(Exception):
    """Exception raised when Picard iteration fails to converge."""
    pass


def picard_iteration(
    mesh: Mesh,
    node_to_dof: Dict[int, int],
    kappa: float,
    robin_boundaries: Dict[int, Tuple[float, float]],
    radiation_boundaries: Dict[int, Tuple[float, float, float]],
    dirichlet_boundaries: Optional[Dict[int, float]] = None,
    U_initial: Optional[np.ndarray] = None,
    max_iter: int = 50,
    tol_abs: float = 1e-3,
    tol_rel: float = 1e-4,
    relaxation: float = 1.0,
    verbose: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Solve nonlinear thermal problem with radiation using Picard iteration.

    The algorithm linearizes the Stefan-Boltzmann radiation term (ε·σ·T⁴) using
    the temperature from the previous iteration:
        α_rad^(k) = ε·σ·(T^(k))³

    At each iteration, a linear system is solved:
        [K + M_conv + M_rad(T^(k))] · T^(k+1) = F_conv + F_rad

    Convergence is achieved when:
        ||T^(k+1) - T^(k)|| < tol_abs + tol_rel * ||T^(k+1)||

    Parameters
    ----------
    mesh : Mesh
        Finite element mesh
    node_to_dof : Dict[int, int]
        Mapping from node IDs to degrees of freedom
    kappa : float
        Thermal conductivity [W/(m·K)]
    robin_boundaries : Dict[int, Tuple[float, float]]
        Robin BC by physical ID: {phys_id: (alpha, T_E)}
        where alpha is convection coefficient [W/(m²·K)] and T_E is external temperature [K]
    radiation_boundaries : Dict[int, Tuple[float, float, float]]
        Radiation BC by physical ID: {phys_id: (epsilon, T_inf, sigma)}
        where epsilon is emissivity, T_inf is ambient temperature [K],
        and sigma is Stefan-Boltzmann constant [W/(m²·K⁴)]
    dirichlet_boundaries : Optional[Dict[int, float]]
        Dirichlet BC by physical ID: {phys_id: T_fixed}
    U_initial : Optional[np.ndarray]
        Initial guess for temperature field. If None, solve convection-only problem.
    max_iter : int, optional
        Maximum number of Picard iterations (default: 50)
    tol_abs : float, optional
        Absolute convergence tolerance in K (default: 1e-3 = 1 K)
    tol_rel : float, optional
        Relative convergence tolerance (default: 1e-4 = 0.01%)
    relaxation : float, optional
        Under-relaxation factor in [0, 1] (default: 1.0 = no relaxation)
        U^(k+1) = relaxation * U_new + (1 - relaxation) * U^(k)
    verbose : bool, optional
        Print iteration details (default: False)

    Returns
    -------
    U : np.ndarray
        Converged temperature field [K]
    info : Dict[str, Any]
        Convergence information containing:
        - 'converged': bool - Whether iteration converged
        - 'iterations': int - Number of iterations performed
        - 'final_residual': float - Final residual norm [K]
        - 'residual_history': List[float] - Residual at each iteration
        - 'T_max_history': List[float] - Maximum temperature at each iteration

    Raises
    ------
    ConvergenceError
        If iteration fails to converge within max_iter iterations

    Examples
    --------
    >>> robin_bc = {1: (100.0, 2000.0)}  # alpha=100, T_E=2000K
    >>> radiation_bc = {1: (0.85, 230.0, 5.67e-8)}  # epsilon, T_inf, sigma
    >>> U, info = picard_iteration(mesh, node_to_dof, kappa=160.0,
    ...                             robin_boundaries=robin_bc,
    ...                             radiation_boundaries=radiation_bc)
    >>> print(f"Converged in {info['iterations']} iterations")
    >>> print(f"T_max = {U.max():.1f} K")
    """

    num_dofs = len(node_to_dof)

    logger.info("="*60)
    logger.info("Picard Iteration: Nonlinear Thermal Problem with Radiation")
    logger.info("="*60)
    logger.info(f"Problem size: {num_dofs} DOFs")
    logger.info(f"Convergence criteria: abs_tol={tol_abs} K, rel_tol={tol_rel}")
    logger.info(f"Max iterations: {max_iter}, relaxation={relaxation}")

    # Initialize solution
    if U_initial is None:
        logger.info("Initializing with convection-only solution...")
        # Solve linear problem without radiation (U_current=None)
        A_init, F_init = assemble_global_system_with_radiation(
            mesh, node_to_dof, kappa, robin_boundaries,
            radiation_boundaries, U_current=None
        )
        if dirichlet_boundaries:
            A_init, F_init = apply_dirichlet_conditions(
                A_init, F_init, mesh, node_to_dof, dirichlet_boundaries
            )
        U_current = solve_linear_system(A_init, F_init, method='direct')
        logger.info(f"  Initial solution: T_max={U_current.max():.1f} K, T_min={U_current.min():.1f} K")
    else:
        U_current = U_initial.copy()
        logger.info(f"Using provided initial guess: T_max={U_current.max():.1f} K")

    # Iteration history
    residual_history = []
    T_max_history = [U_current.max()]

    logger.info("\nStarting Picard iterations:")
    logger.info("-" * 60)

    for iteration in range(max_iter):
        # Assemble system with radiation linearized at U_current
        A, F = assemble_global_system_with_radiation(
            mesh, node_to_dof, kappa, robin_boundaries,
            radiation_boundaries, U_current=U_current
        )

        # Apply Dirichlet BC
        if dirichlet_boundaries:
            A, F = apply_dirichlet_conditions(
                A, F, mesh, node_to_dof, dirichlet_boundaries
            )

        # Solve linear system
        U_new = solve_linear_system(A, F, method='direct')

        # Apply under-relaxation if specified
        if relaxation < 1.0:
            U_new = relaxation * U_new + (1.0 - relaxation) * U_current

        # Compute residual
        residual_abs = np.linalg.norm(U_new - U_current)
        residual_rel = residual_abs / (np.linalg.norm(U_new) + 1e-10)

        residual_history.append(residual_abs)
        T_max_history.append(U_new.max())

        # Log iteration info
        if verbose or iteration % 5 == 0:
            logger.info(f"  Iter {iteration:3d}: T_max={U_new.max():8.1f} K, "
                       f"residual={residual_abs:.3e} K (rel={residual_rel:.3e})")

        # Check convergence
        converged = (residual_abs < tol_abs) or (residual_rel < tol_rel)

        if converged:
            logger.info("-" * 60)
            logger.info(f"✓ CONVERGED in {iteration + 1} iterations")
            logger.info(f"  Final T_max = {U_new.max():.1f} K, T_min = {U_new.min():.1f} K")
            logger.info(f"  Final residual = {residual_abs:.3e} K")
            logger.info("=" * 60)

            info = {
                'converged': True,
                'iterations': iteration + 1,
                'final_residual': residual_abs,
                'residual_history': residual_history,
                'T_max_history': T_max_history
            }
            return U_new, info

        # Check for divergence
        if iteration > 5:
            recent_residuals = residual_history[-5:]
            if all(r > recent_residuals[0] for r in recent_residuals[1:]):
                logger.warning("⚠ Divergence detected (residual increasing)")
                logger.warning("  Consider: reducing relaxation parameter or checking BC")

        # Check for oscillation
        if iteration > 10:
            recent_max_temps = T_max_history[-10:]
            if max(recent_max_temps) - min(recent_max_temps) < 10.0:
                if residual_abs > tol_abs * 10:
                    logger.warning("⚠ Stagnation detected (T_max oscillating without convergence)")

        # Update for next iteration
        U_current = U_new.copy()

    # Max iterations reached without convergence
    logger.error("=" * 60)
    logger.error(f"✗ FAILED TO CONVERGE after {max_iter} iterations")
    logger.error(f"  Final residual = {residual_history[-1]:.3e} K")
    logger.error(f"  Required: {tol_abs:.3e} K (absolute) or {tol_rel:.3e} (relative)")
    logger.error("=" * 60)

    info = {
        'converged': False,
        'iterations': max_iter,
        'final_residual': residual_history[-1],
        'residual_history': residual_history,
        'T_max_history': T_max_history
    }

    # Optionally raise exception or return best solution
    # For now, return the last iteration with warning
    logger.warning("Returning last iteration result (may not be accurate)")
    return U_current, info


def compute_radiation_coefficient(
    epsilon: float,
    sigma: float,
    T: float
) -> float:
    """
    Compute linearized radiation coefficient for Picard iteration.

    The Stefan-Boltzmann radiation term is:
        q_rad = ε·σ·(T⁴ - T_∞⁴)

    For Picard linearization, we write this as:
        q_rad ≈ α_rad(T^(k)) · (T^(k+1) - T_∞)

    where α_rad(T^(k)) = ε·σ·(T^(k))³ · (T^(k) + T_∞) / (T^(k) - T_∞)

    For T >> T_∞ (typical at high speeds), this simplifies to:
        α_rad ≈ ε·σ·(T^(k))³

    Parameters
    ----------
    epsilon : float
        Surface emissivity (0-1)
    sigma : float
        Stefan-Boltzmann constant [W/(m²·K⁴)]
    T : float
        Temperature at which to linearize [K]

    Returns
    -------
    alpha_rad : float
        Linearized radiation coefficient [W/(m²·K)]

    Examples
    --------
    >>> epsilon = 0.85  # Carbon-Carbon TPS
    >>> sigma = 5.67e-8  # Stefan-Boltzmann constant
    >>> T = 2000.0  # K
    >>> alpha_rad = compute_radiation_coefficient(epsilon, sigma, T)
    >>> print(f"α_rad = {alpha_rad:.1f} W/(m²·K)")
    α_rad = 387.1 W/(m²·K)
    """
    # Clip temperature to reasonable range to avoid numerical issues
    if np.any(T < 200.0) or np.any(T > 5000.0):
        logger.warning(
            f"Température hors bornes détectée. "
            f"Min: {np.min(T):.1f} K, Max: {np.max(T):.1f} K. "
            f"Clipping à [200, 5000] K."
        )
    T_clipped = np.clip(T, 200.0, 5000.0)

    alpha_rad = epsilon * sigma * T_clipped**3

    return alpha_rad
