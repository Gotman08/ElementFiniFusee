"""
Module de résolution du système linéaire
Utilise scipy.sparse.linalg pour systèmes creux
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, cg, gmres
import time

def solve_linear_system(A: csr_matrix,
                        F: np.ndarray,
                        method: str = 'direct') -> np.ndarray:
    """
    Résout le système linéaire A * U = F

    Args:
        A: Matrice sparse CSR (N, N)
        F: Second membre (N,)
        method: Méthode de résolution
            - 'direct': Factorisation directe (LU)
            - 'cg': Gradient conjugué (pour matrices symétriques définies positives)
            - 'gmres': GMRES (pour matrices non-symétriques)

    Returns:
        U: Solution (N,)
    """
    print(f"Résolution du système ({A.shape[0]} DOFs) - Méthode: {method}")

    t_start = time.time()

    if method == 'direct':
        # Factorisation LU directe (meilleure pour petits systèmes)
        U = spsolve(A, F)

    elif method == 'cg':
        # Gradient conjugué (requiert A symétrique définie positive)
        U, info = cg(A, F, tol=1e-8, maxiter=1000)
        if info != 0:
            print(f"Attention: CG n'a pas convergé (info={info})")

    elif method == 'gmres':
        # GMRES (cas général)
        U, info = gmres(A, F, tol=1e-8, maxiter=1000)
        if info != 0:
            print(f"Attention: GMRES n'a pas convergé (info={info})")

    else:
        raise ValueError(f"Méthode inconnue: {method}")

    t_end = time.time()

    print(f"Résolution terminée en {t_end - t_start:.3f} s")
    print(f"  - min(U) = {np.min(U):.2f}")
    print(f"  - max(U) = {np.max(U):.2f}")
    print(f"  - mean(U) = {np.mean(U):.2f}")

    # Vérification du résidu
    residual = np.linalg.norm(A @ U - F) / np.linalg.norm(F)
    print(f"  - Résidu relatif: {residual:.2e}")

    return U


def extract_solution_stats(U: np.ndarray) -> dict:
    """
    Extrait les statistiques de la solution

    Args:
        U: Vecteur solution

    Returns:
        Dictionnaire avec statistiques
    """
    stats = {
        'min': np.min(U),
        'max': np.max(U),
        'mean': np.mean(U),
        'std': np.std(U)
    }
    return stats


if __name__ == '__main__':
    print("Module de résolution de systèmes linéaires")
    print("Test simple avec matrice de Laplace 1D")

    from scipy.sparse import diags

    # Système test: -u'' = 1 sur [0,1] avec u(0)=0, u(1)=0
    n = 10
    h = 1.0 / (n + 1)
    A = diags([1, -2, 1], [-1, 0, 1], shape=(n, n)) / h**2
    A = A.tocsr()
    F = np.ones(n)

    U = solve_linear_system(A, F, method='direct')
    print(f"Solution test 1D: {U}")
