"""
Module d'assemblage du système linéaire global
Implémente l'assemblage de la matrice de rigidité et du second membre
AVEC LES TERMES DE BORD (conditions de Robin)
"""
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from mesh_reader import Mesh
from fem_elements import TriangleP1, EdgeP1
from typing import Dict, Tuple

def assemble_global_system(mesh: Mesh,
                          node_to_dof: Dict[int, int],
                          kappa: float,
                          robin_boundaries: Dict[int, Tuple[float, float]]) -> Tuple[csr_matrix, np.ndarray]:
    """
    Assemble le système linéaire global A*U = F
    AVEC termes de bord Robin

    Args:
        mesh: Objet Mesh
        node_to_dof: Mapping node_id -> DOF index
        kappa: Conductivité thermique (W/m·K)
        robin_boundaries: Dict {physical_id: (alpha, u_E)}
            - physical_id: ID du groupe physique de bord
            - alpha: Coefficient de convection (W/m²·K)
            - u_E: Température extérieure (K)

    Returns:
        (A, F): Matrice sparse (N, N) et vecteur (N,)
                A = A_volume + A_robin
                F = F_robin
    """
    num_dofs = len(node_to_dof)

    # Matrices sparse (format LIL pour assemblage efficace)
    A = lil_matrix((num_dofs, num_dofs))
    F = np.zeros(num_dofs)

    # ========================================================================
    # ÉTAPE 1: ASSEMBLAGE VOLUMIQUE (Matrice de rigidité)
    # A_vol[i,j] = ∫_Ω κ ∇φ_i · ∇φ_j dΩ
    # ========================================================================
    triangles = mesh.get_triangles()
    print(f"Assemblage volumique: {len(triangles)} triangles...")

    for elem_id, elem in triangles.items():
        # Récupérer les IDs globaux des noeuds
        global_nodes = elem['nodes']

        # Récupérer les coordonnées physiques
        coords = mesh.get_node_coords(global_nodes)

        # Calculer la matrice de rigidité élémentaire
        K_elem = TriangleP1.local_stiffness_matrix(coords, kappa)

        # Mapper vers les DOFs globaux
        local_dofs = [node_to_dof[nid] for nid in global_nodes]

        # Assembler dans la matrice globale
        for i in range(3):
            for j in range(3):
                A[local_dofs[i], local_dofs[j]] += K_elem[i, j]

    # ========================================================================
    # ÉTAPE 2: ASSEMBLAGE DES TERMES DE BORD ROBIN
    # A_robin[i,j] += ∫_{∂K ∩ Γ_F} α φ_i φ_j dσ   (MATRICE DE MASSE SURFACIQUE)
    # F_robin[i]   += ∫_{∂K ∩ Γ_F} α u_E φ_i dσ   (VECTEUR DE CHARGE SURFACIQUE)
    # ========================================================================
    for physical_id, (alpha, u_E) in robin_boundaries.items():
        # Récupérer les arêtes de bord appartenant à ce groupe physique
        boundary_edges = mesh.get_boundary_edges_by_physical(physical_id)

        print(f"Assemblage Robin (Physical ID {physical_id}): "
              f"{len(boundary_edges)} aretes, alpha={alpha}, u_E={u_E}")

        for edge_id, edge_elem in boundary_edges:
            # Récupérer les IDs globaux des noeuds de l'arête
            global_nodes = edge_elem['nodes']

            # Récupérer les coordonnées physiques
            coords = mesh.get_node_coords(global_nodes)

            # Calculer la matrice de masse surfacique élémentaire
            M_elem = EdgeP1.local_mass_matrix(coords, alpha)

            # Calculer le vecteur de charge surfacique élémentaire
            F_elem = EdgeP1.local_load_vector(coords, alpha, u_E)

            # Mapper vers les DOFs globaux
            local_dofs = [node_to_dof[nid] for nid in global_nodes]

            # Assembler dans A et F
            for i in range(2):
                # Vecteur de charge
                F[local_dofs[i]] += F_elem[i]

                # Matrice de masse surfacique
                for j in range(2):
                    A[local_dofs[i], local_dofs[j]] += M_elem[i, j]

    # Conversion en format CSR (efficace pour résolution)
    A = A.tocsr()

    print(f"Assemblage terminé: Système {num_dofs} × {num_dofs}")
    print(f"  - Nombre d'éléments non-nuls: {A.nnz}")
    print(f"  - Norme du second membre: {np.linalg.norm(F):.2e}")

    return A, F


def assemble_volumetric_load(mesh: Mesh,
                             node_to_dof: Dict[int, int],
                             source_term) -> np.ndarray:
    """
    Assemble le vecteur de charge volumique (terme source)
    F_vol[i] = ∫_Ω f φ_i dΩ

    Args:
        mesh: Objet Mesh
        node_to_dof: Mapping node_id -> DOF index
        source_term: Fonction f(x, y) ou constante

    Returns:
        F_vol: Vecteur de charge volumique
    """
    num_dofs = len(node_to_dof)
    F_vol = np.zeros(num_dofs)

    triangles = mesh.get_triangles()

    for elem_id, elem in triangles.items():
        global_nodes = elem['nodes']
        coords = mesh.get_node_coords(global_nodes)

        # Calcul de l'aire
        _, area = TriangleP1.physical_gradients(coords)

        # Points de quadrature
        points, weights = TriangleP1.quadrature_volume()

        # Calculer les coordonnées physiques du point de quadrature
        # x_phys = N * coords (transformation isoparamétrique)
        xi_q, eta_q = points[0]
        phi_q = TriangleP1.shape_functions(xi_q, eta_q)
        x_phys = phi_q @ coords

        # Évaluer le terme source
        if callable(source_term):
            f_value = source_term(x_phys[0], x_phys[1])
        else:
            f_value = source_term

        # Intégration
        F_elem = f_value * weights[0] * 2 * area * phi_q  # 2*area = det(J)

        # Assembler
        local_dofs = [node_to_dof[nid] for nid in global_nodes]
        for i in range(3):
            F_vol[local_dofs[i]] += F_elem[i]

    return F_vol


if __name__ == '__main__':
    # Test sur un maillage simple
    print("Test du module d'assemblage")
    print("=" * 60)
    print("Créer un maillage .msh pour tester l'assemblage complet")
    print("Exemple: maillage rectangulaire avec bords Robin")
    print("=" * 60)
