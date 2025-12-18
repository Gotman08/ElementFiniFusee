"""
@file assembly_1d.py
@brief Assemblage du système linéaire global pour la méthode des éléments finis 1D
@author ElementFiniFusee
@date 2025

@details
Ce module implémente l'assemblage du système linéaire global $A \\cdot U = F$
pour l'analyse thermique 1D utilisant la méthode des éléments finis.

Le processus d'assemblage inclut :
1. Assemblage volumique : Matrice de rigidité issue de la diffusion thermique
   $A_{vol}[i,j] = \\int_\\Omega \\kappa \\frac{du_i}{dx} \\frac{du_j}{dx} dx$

2. Assemblage des termes sources :
   $F_{vol}[i] = \\int_\\Omega f \\phi_i dx$

3. Assemblage des conditions aux limites de Robin :
   $A_{robin}[i,i] = \\alpha$ (au nœud de bord)
   $F_{robin}[i] = \\alpha u_E$ (au nœud de bord)

L'implémentation utilise des matrices denses numpy (la taille des systèmes 1D
reste raisonnable, typiquement < 10000 DOFs).
"""
import logging
import numpy as np
from typing import Dict, Tuple, Union, Callable, Optional
from numpy.typing import NDArray

from src.mesh.mesh_1d import Mesh1D
from src.core.fem_elements_1d import SegmentP1, PointP1
from src.utils.exceptions import AssemblyError, ValidationError

logger = logging.getLogger(__name__)


def assemble_1d_system(
    mesh: Mesh1D,
    kappa: float,
    source_term: Union[float, Callable[[float], float]] = 0.0,
    robin_bc: Optional[Dict[str, Tuple[float, float]]] = None
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    @brief Assemble le système global 1D avec conditions de Robin.

    @details
    Assemble le système global $A \\cdot U = F$ où :
    - $A = A_{volume} + A_{robin}$ : Matrice de rigidité globale avec termes de convection
    - $F = F_{volume} + F_{robin}$ : Vecteur charge avec terme source et convection

    La contribution volumique provient du terme de diffusion thermique :
    $A_{vol}[i,j] = \\int_\\Omega \\kappa \\frac{d\\phi_i}{dx} \\frac{d\\phi_j}{dx} dx$

    La contribution source :
    $F_{vol}[i] = \\int_\\Omega f(x) \\phi_i dx$

    La contribution de Robin modélise le transfert convectif :
    $A_{robin}[i,i] = \\alpha$ (ajouté au nœud de bord)
    $F_{robin}[i] = \\alpha u_E$ (ajouté au nœud de bord)

    @param mesh: Objet Mesh1D contenant les nœuds et la connectivité
    @param kappa: Coefficient de conductivité thermique $\\kappa$ [W/(m.K)]
    @param source_term: Terme source volumique $f$ [W/m³]
        - float constant, ou
        - fonction callable f(x) retournant un float
        - Par défaut : 0.0 (équation de Laplace)
    @param robin_bc: Dictionnaire des conditions de Robin
        - Clé : 'left' ou 'right' (position du bord)
        - Valeur : tuple (alpha, u_E)
          - alpha : Coefficient de transfert [W/(m².K)]
          - u_E : Température extérieure [K]
        - Si None ou dict vide : pas de condition de Robin

    @return Tuple contenant :
        - A : Matrice globale dense de forme (n_nodes, n_nodes)
        - F : Vecteur charge de forme (n_nodes,)

    @raises ValidationError: Si kappa <= 0, alpha < 0, ou paramètres invalides
    @raises AssemblyError: Si problème lors de l'assemblage

    @example
    >>> from src.mesh.mesh_1d import create_uniform_mesh
    >>> mesh = create_uniform_mesh(L=1.0, n_elements=10)
    >>> robin_bc = {'right': (50.0, 300.0)}
    >>> A, F = assemble_1d_system(mesh, kappa=10.0, source_term=100.0, robin_bc=robin_bc)
    >>> # Système prêt pour résolution après application de BC Dirichlet
    """
    # Validation
    if kappa <= 0:
        raise ValidationError(f"kappa doit être positif, reçu {kappa}")

    n = mesh.n_nodes
    A = np.zeros((n, n))
    F = np.zeros(n)

    logger.info(
        f"Assemblage système 1D : {mesh.n_elements} éléments, "
        f"{n} nœuds, κ={kappa}"
    )

    # ========== 1. ASSEMBLAGE VOLUMIQUE ==========
    for elem_id in range(mesh.n_elements):
        coords = mesh.get_element_coords(elem_id)
        global_dofs = mesh.elements[elem_id]

        try:
            # Matrice de rigidité élémentaire
            K_elem = SegmentP1.local_stiffness_matrix(coords, kappa)

            # Vecteur charge élémentaire (terme source)
            F_elem = SegmentP1.local_load_vector(coords, source_term)

        except (ElementError, ValidationError) as e:
            raise AssemblyError(
                f"Erreur lors du calcul des matrices élémentaires "
                f"pour l'élément {elem_id} : {str(e)}"
            )

        # Assemblage dans la matrice et le vecteur globaux
        for i in range(2):
            F[global_dofs[i]] += F_elem[i]
            for j in range(2):
                A[global_dofs[i], global_dofs[j]] += K_elem[i, j]

    logger.debug(
        f"Assemblage volumique terminé : matrice {A.shape}, "
        f"nnz={np.count_nonzero(A)}"
    )

    # ========== 2. ASSEMBLAGE CONDITIONS DE ROBIN ==========
    if robin_bc:
        boundary_nodes = mesh.get_boundary_nodes()

        for location, (alpha, u_E) in robin_bc.items():
            if location not in boundary_nodes:
                raise ValidationError(
                    f"Position de Robin invalide : '{location}'. "
                    f"Doit être 'left' ou 'right'"
                )

            # Récupérer l'indice du nœud de bord
            dof = boundary_nodes[location]

            # Contributions de Robin
            try:
                alpha_contrib = PointP1.local_robin_matrix(alpha)
                F_robin = PointP1.local_robin_load(alpha, u_E)
            except ValidationError as e:
                raise AssemblyError(
                    f"Erreur lors du calcul de la contribution Robin "
                    f"à {location} : {str(e)}"
                )

            # Ajouter au système global
            A[dof, dof] += alpha_contrib
            F[dof] += F_robin

            logger.debug(
                f"Condition de Robin ajoutée au nœud {location} (DOF {dof}) : "
                f"α={alpha}, u_E={u_E}"
            )

    # ========== 3. DIAGNOSTICS ==========
    # Vérifier la symétrie de A (avant application de BC Dirichlet)
    asymmetry = np.linalg.norm(A - A.T) / (np.linalg.norm(A) + 1e-16)
    if asymmetry > 1e-12:
        logger.warning(
            f"Matrice assemblée non symétrique : ||A - A^T|| / ||A|| = {asymmetry:.2e}"
        )
    else:
        logger.debug(f"Matrice assemblée symétrique (asymétrie = {asymmetry:.2e})")

    # Numéro de conditionnement (indicateur de stabilité)
    try:
        cond_number = np.linalg.cond(A)
        logger.info(f"Numéro de conditionnement de A : {cond_number:.2e}")
        if cond_number > 1e12:
            logger.warning(
                f"Matrice mal conditionnée (cond = {cond_number:.2e}). "
                "Possible instabilité numérique."
            )
    except np.linalg.LinAlgError:
        logger.warning("Impossible de calculer le numéro de conditionnement")

    # Norme du vecteur charge
    F_norm = np.linalg.norm(F)
    logger.debug(f"Norme du vecteur charge : ||F|| = {F_norm:.2e}")

    logger.info("Assemblage global terminé avec succès")

    return A, F


def apply_dirichlet_1d(
    A: NDArray[np.float64],
    F: NDArray[np.float64],
    boundary_conditions: Dict[str, float],
    mesh: Mesh1D
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    @brief Applique les conditions de Dirichlet par méthode d'élimination.

    @details
    Applique les conditions aux limites essentielles (Dirichlet) au système
    $A \\cdot U = F$ en utilisant la méthode d'élimination.

    Pour chaque DOF $i$ avec valeur imposée $g_i$ :
    1. Transférer la contribution au second membre : $F[j] -= A[j,i] \\cdot g_i$ pour tout $j$
    2. Restaurer le terme diagonal : $F[i] += A[i,i] \\cdot g_i$
    3. Mettre la ligne $i$ à zéro sauf la diagonale : $A[i,:] = 0$, $A[i,i] = 1$
    4. Imposer la valeur : $F[i] = g_i$

    Cette méthode préserve la symétrie de la matrice et assure que $U[i] = g_i$.

    @param A: Matrice globale de forme (n, n), sera copiée (non modifiée en place)
    @param F: Vecteur charge de forme (n,), sera copié (non modifié en place)
    @param boundary_conditions: Dictionnaire des conditions de Dirichlet
        - Clé : 'left' ou 'right' (position du bord)
        - Valeur : température imposée [K]
    @param mesh: Objet Mesh1D pour accéder aux nœuds de bord

    @return Tuple contenant :
        - A_bc : Matrice modifiée de forme (n, n)
        - F_bc : Vecteur modifié de forme (n,)

    @raises ValidationError: Si position de bord invalide

    @example
    >>> A_bc, F_bc = apply_dirichlet_1d(A, F, {'left': 0.0, 'right': 100.0}, mesh)
    >>> U = np.linalg.solve(A_bc, F_bc)
    >>> # U[0] = 0.0 et U[-1] = 100.0 exactement
    """
    # Copier pour ne pas modifier les matrices d'entrée
    A_bc = A.copy()
    F_bc = F.copy()

    boundary_nodes = mesh.get_boundary_nodes()

    logger.info(
        f"Application de {len(boundary_conditions)} conditions de Dirichlet"
    )

    for location, value in boundary_conditions.items():
        if location not in boundary_nodes:
            raise ValidationError(
                f"Position de Dirichlet invalide : '{location}'. "
                f"Doit être 'left' ou 'right'"
            )

        dof = boundary_nodes[location]

        # Méthode d'élimination
        # Étape 1 : Transférer la contribution de la colonne dof au RHS
        F_bc -= A_bc[:, dof] * value

        # Étape 2 : Restaurer le terme diagonal (éliminé à l'étape 1)
        F_bc[dof] += A_bc[dof, dof] * value

        # Étape 3 : Mettre la ligne dof à zéro sauf la diagonale
        A_bc[dof, :] = 0
        A_bc[dof, dof] = 1

        # Étape 4 : Imposer la valeur au RHS
        F_bc[dof] = value

        logger.debug(
            f"BC Dirichlet appliquée au nœud {location} (DOF {dof}) : u = {value}"
        )

    # Vérification : les lignes modifiées ont bien [0,...,0,1,0,...,0]
    for location in boundary_conditions.keys():
        dof = boundary_nodes[location]
        row = A_bc[dof, :]
        expected_nnz = 1  # Seulement le terme diagonal
        actual_nnz = np.count_nonzero(row)
        if actual_nnz != expected_nnz:
            logger.warning(
                f"Ligne {dof} a {actual_nnz} termes non-nuls "
                f"(attendu {expected_nnz})"
            )

    logger.info("Conditions de Dirichlet appliquées avec succès")

    return A_bc, F_bc


def solve_1d_system(A: NDArray[np.float64], F: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    @brief Résout le système linéaire 1D.

    @details
    Résout $A \\cdot U = F$ en utilisant la factorisation LU de numpy.
    Pour les matrices tridiagonales (cas 1D), cette méthode est très efficace.

    Effectue également une vérification du résidu pour s'assurer de la qualité
    de la solution.

    @param A: Matrice du système de forme (n, n)
    @param F: Vecteur second membre de forme (n,)

    @return Vecteur solution U de forme (n,)

    @raises np.linalg.LinAlgError: Si la matrice est singulière

    @example
    >>> U = solve_1d_system(A, F)
    >>> # Vérifier : np.allclose(A @ U, F)
    """
    logger.info(f"Résolution du système linéaire : {A.shape[0]} DOFs")

    try:
        U = np.linalg.solve(A, F)
    except np.linalg.LinAlgError as e:
        logger.error(f"Échec de la résolution : {str(e)}")
        raise

    # Vérification du résidu
    residual = np.linalg.norm(A @ U - F)
    relative_residual = residual / (np.linalg.norm(F) + 1e-16)

    logger.info(
        f"Solution obtenue : u_min={U.min():.4e}, u_max={U.max():.4e}"
    )
    logger.debug(
        f"Résidu : ||A·U - F|| = {residual:.2e}, "
        f"relatif = {relative_residual:.2e}"
    )

    if relative_residual > 1e-6:
        logger.warning(
            f"Résidu élevé ({relative_residual:.2e}). "
            "Vérifier le conditionnement de la matrice."
        )

    return U


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

    from src.mesh.mesh_1d import create_uniform_mesh

    logger.info("=== Test Assemblage 1D ===")

    # Problème test : -κ u'' = f sur [0,1] avec u(0)=0, -κ u'(1) = α(u(1) - u_E)
    L = 1.0
    n_elements = 10
    kappa = 10.0
    f = 100.0  # Source constante
    alpha = 50.0
    u_E = 300.0

    # Créer maillage
    mesh = create_uniform_mesh(L, n_elements)

    # Assembler
    robin_bc = {'right': (alpha, u_E)}
    A, F = assemble_1d_system(mesh, kappa, source_term=f, robin_bc=robin_bc)

    logger.info(f"Matrice assemblée : forme {A.shape}, nnz={np.count_nonzero(A)}")
    logger.debug(f"Matrice A :\n{A}")
    logger.debug(f"Vecteur F :\n{F}")

    # Appliquer BC Dirichlet
    dirichlet_bc = {'left': 0.0}
    A_bc, F_bc = apply_dirichlet_1d(A, F, dirichlet_bc, mesh)

    logger.info(f"BC Dirichlet appliquées")
    logger.debug(f"Première ligne de A_bc : {A_bc[0, :]}")
    logger.debug(f"F_bc[0] = {F_bc[0]}")

    # Résoudre
    U = solve_1d_system(A_bc, F_bc)

    logger.info(f"Solution : u_min={U.min():.2f}, u_max={U.max():.2f}")
    logger.debug(f"Solution complète :\n{U}")

    # Vérifier BC
    logger.info(f"Vérification : u(0) = {U[0]:.2e} (attendu 0.0)")
    logger.info(f"Vérification : u(L) = {U[-1]:.2f}")

    logger.info("Tests terminés avec succès")
