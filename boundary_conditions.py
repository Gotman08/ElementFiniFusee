"""
Module d'application des conditions aux limites de Dirichlet
Méthode d'élimination (imposer directement la valeur aux noeuds)
"""
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from mesh_reader import Mesh
from typing import Dict, List

def apply_dirichlet_conditions(A: csr_matrix,
                               F: np.ndarray,
                               mesh: Mesh,
                               node_to_dof: Dict[int, int],
                               dirichlet_boundaries: Dict[int, float]) -> tuple:
    """
    Applique les conditions de Dirichlet par élimination

    Méthode: Pour chaque DOF i avec condition u_i = g_i:
        1. Modifier la ligne i: A[i, :] = 0, A[i, i] = 1
        2. Modifier le second membre: F[i] = g_i
        3. Modifier les autres lignes: F[j] -= A[j, i] * g_i

    Args:
        A: Matrice globale (CSR format)
        F: Second membre global
        mesh: Objet Mesh
        node_to_dof: Mapping node_id -> DOF index
        dirichlet_boundaries: Dict {physical_id: imposed_value}

    Returns:
        (A_bc, F_bc): Système modifié avec CL appliquées
    """
    # Convertir en LIL pour modification efficace
    A_bc = A.tolil()
    F_bc = F.copy()

    # Collecter tous les DOFs Dirichlet
    dirichlet_dofs = []
    dirichlet_values = []

    for physical_id, imposed_value in dirichlet_boundaries.items():
        # Récupérer les arêtes de bord
        boundary_edges = mesh.get_boundary_edges_by_physical(physical_id)

        for edge_id, edge_elem in boundary_edges:
            global_nodes = edge_elem['nodes']

            for node_id in global_nodes:
                if node_id in node_to_dof:
                    dof = node_to_dof[node_id]
                    if dof not in dirichlet_dofs:
                        dirichlet_dofs.append(dof)
                        dirichlet_values.append(imposed_value)

    print(f"Application de {len(dirichlet_dofs)} conditions de Dirichlet...")

    # Appliquer les conditions
    for dof, value in zip(dirichlet_dofs, dirichlet_values):
        # Modifier les autres lignes (avant d'effacer la colonne)
        for i in range(A_bc.shape[0]):
            if i != dof:
                F_bc[i] -= A_bc[i, dof] * value

        # Modifier la ligne dof
        A_bc[dof, :] = 0
        A_bc[dof, dof] = 1
        F_bc[dof] = value

    # Reconvertir en CSR
    A_bc = A_bc.tocsr()

    return A_bc, F_bc


def get_boundary_nodes(mesh: Mesh,
                       node_to_dof: Dict[int, int],
                       physical_ids: List[int]) -> List[int]:
    """
    Récupère les DOFs des noeuds appartenant à certains bords

    Args:
        mesh: Objet Mesh
        node_to_dof: Mapping node_id -> DOF index
        physical_ids: Liste des IDs de groupes physiques

    Returns:
        Liste des DOFs
    """
    boundary_dofs = []

    for physical_id in physical_ids:
        boundary_edges = mesh.get_boundary_edges_by_physical(physical_id)

        for edge_id, edge_elem in boundary_edges:
            global_nodes = edge_elem['nodes']

            for node_id in global_nodes:
                if node_id in node_to_dof:
                    dof = node_to_dof[node_id]
                    if dof not in boundary_dofs:
                        boundary_dofs.append(dof)

    return boundary_dofs


if __name__ == '__main__':
    print("Module de conditions aux limites de Dirichlet")
    print("Utiliser avec un maillage et un système assemblé")
