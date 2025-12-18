"""
@file solver.py
@brief Linear system solver for finite element analysis
@author HPC-Code-Documenter
@date 2025

@details
This module provides solvers for the linear system $A \\cdot U = F$ arising
from finite element discretization of thermal problems.

Supported solution methods:
1. Direct solver (LU factorization): Best for small to medium systems
2. Conjugate Gradient (CG): For symmetric positive definite matrices
3. GMRES: For general non-symmetric matrices

The module includes residual verification and solution statistics extraction
for quality assurance of computed results.

Performance considerations:
- Direct method: $O(N^3)$ complexity but robust
- CG/GMRES: $O(N \\cdot k)$ where $k$ is iteration count, better for large sparse systems
"""
import logging
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, cg, gmres
from typing import Dict
from numpy.typing import NDArray

from src.utils.exceptions import SolverError, ValidationError

logger = logging.getLogger(__name__)


def solve_linear_system(A: csr_matrix,
                        F: NDArray[np.float64],
                        method: str = 'direct') -> NDArray[np.float64]:
    """
    @brief Solve the linear system A * U = F using specified method.

    @details
    Solves the assembled finite element system using one of three methods:
    - Direct: LU factorization via SuperLU, robust but $O(N^3)$ complexity
    - CG: Conjugate Gradient, requires $A$ to be symmetric positive definite
    - GMRES: Generalized Minimal Residual, works for non-symmetric systems

    After solving, the method computes the relative residual:
    $r = \\frac{\\|A \\cdot U - F\\|}{\\|F\\|}$

    A warning is issued if $r > 10^{-4}$, indicating potential numerical issues.

    @param A: Sparse CSR matrix of shape (N, N) - the global stiffness matrix
    @param F: NumPy array of shape (N,) - the global load vector
    @param method: Solution method, one of:
        - 'direct': LU factorization (default, recommended for N < 10000)
        - 'cg': Conjugate Gradient (for symmetric positive definite systems)
        - 'gmres': GMRES (for general systems)

    @return NumPy array of shape (N,) containing the solution vector U (temperatures)

    @raises ValidationError: If matrix is not square, dimensions mismatch, or unknown method
    @raises SolverError: If iterative solver fails to converge or encounters an error

    @example
    >>> # Direct solution (default)
    >>> U = solve_linear_system(A, F)
    >>> # Using conjugate gradient for large SPD system
    >>> U = solve_linear_system(A, F, method='cg')
    """
    if A.shape[0] != A.shape[1]:
        raise ValidationError(f"La matrice A doit être carrée, reçu {A.shape}")

    if A.shape[0] != len(F):
        raise ValidationError(f"Dimensions incompatibles: A={A.shape}, F={len(F)}")

    if method not in ['direct', 'cg', 'gmres']:
        raise ValidationError(f"Méthode inconnue: {method}. Valides: direct, cg, gmres")

    logger.info(f"Résolution du système ({A.shape[0]} DOFs) - Méthode: {method}")

    t_start = time.time()

    if method == 'direct':
        U = spsolve(A, F)
        # Validate return type (spsolve can return sparse matrix if singular)
        if not isinstance(U, np.ndarray):
            raise SolverError(
                f"spsolve a retourné {type(U)} au lieu de ndarray. "
                f"La matrice peut être singulière."
            )

    elif method == 'cg':
        U, info = cg(A, F, tol=1e-8, maxiter=1000)
        if info > 0:
            logger.warning(f"CG: convergence non atteinte après {info} itérations")
        elif info < 0:
            raise SolverError(f"CG: erreur d'entrée invalide (info={info})")

    elif method == 'gmres':
        U, info = gmres(A, F, tol=1e-8, maxiter=1000)
        if info > 0:
            logger.warning(f"GMRES: convergence non atteinte après {info} itérations")
        elif info < 0:
            raise SolverError(f"GMRES: erreur d'entrée invalide (info={info})")

    t_end = time.time()

    logger.info(f"Résolution terminée en {t_end - t_start:.3f} s")
    logger.debug(f"  - min(U) = {np.min(U):.2f}")
    logger.debug(f"  - max(U) = {np.max(U):.2f}")
    logger.debug(f"  - mean(U) = {np.mean(U):.2f}")

    norm_F = np.linalg.norm(F)
    if norm_F > 1e-14:
        residual = np.linalg.norm(A @ U - F) / norm_F
        logger.debug(f"  - Résidu relatif: {residual:.2e}")

        if residual > 1e-4:
            logger.warning(f"Résidu élevé: {residual:.2e}. La solution peut être imprécise.")

    return U


def extract_solution_stats(U: NDArray[np.float64]) -> Dict[str, float]:
    """
    @brief Extract statistical measures from the solution vector.

    @details
    Computes summary statistics of the temperature field:
    - Minimum temperature (coldest point)
    - Maximum temperature (hottest point, critical for thermal design)
    - Mean temperature (average thermal state)
    - Standard deviation (temperature variation across domain)

    These statistics are useful for:
    - Validating solution reasonableness
    - Identifying thermal hotspots
    - Comparing different simulation configurations

    @param U: NumPy array containing the solution vector (temperatures at nodes)

    @return Dictionary with keys:
        - 'min': Minimum value in U
        - 'max': Maximum value in U
        - 'mean': Arithmetic mean of U
        - 'std': Standard deviation of U

    @example
    >>> stats = extract_solution_stats(U)
    >>> print(f"Temperature range: {stats['min']:.1f} K to {stats['max']:.1f} K")
    """
    stats: Dict[str, float] = {
        'min': float(np.min(U)),
        'max': float(np.max(U)),
        'mean': float(np.mean(U)),
        'std': float(np.std(U))
    }

    logger.debug(f"Statistiques solution: min={stats['min']:.2f}, max={stats['max']:.2f}, "
                f"mean={stats['mean']:.2f}, std={stats['std']:.2f}")

    return stats


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

    logger.info("Module de résolution de systèmes linéaires")
    logger.info("Test simple avec matrice de Laplace 1D")

    from scipy.sparse import diags

    n = 10
    h = 1.0 / (n + 1)
    A = diags([1, -2, 1], [-1, 0, 1], shape=(n, n)) / h**2
    A = A.tocsr()
    F = np.ones(n)

    U = solve_linear_system(A, F, method='direct')
    logger.info(f"Solution test 1D: {U}")

    stats = extract_solution_stats(U)
    logger.info(f"Statistiques: {stats}")
