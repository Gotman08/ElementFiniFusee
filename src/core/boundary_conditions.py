"""
@file boundary_conditions.py
@brief Dirichlet boundary condition application for FEM systems
@author HPC-Code-Documenter
@date 2025

@details
This module implements the application of Dirichlet (essential) boundary conditions
to the assembled finite element system using the elimination method.

The elimination method modifies the system $A \\cdot U = F$ to enforce prescribed
values at boundary nodes:
1. For each Dirichlet DOF $i$ with prescribed value $g_i$:
   - Row modification: $A[i, :] = 0$, $A[i, i] = 1$
   - RHS modification: $F[i] = g_i$
   - Column elimination: $F[j] -= A[j, i] \\cdot g_i$ for $j \\neq i$

This approach maintains symmetry of the matrix (important for CG solver)
and produces the correct solution at all nodes including boundaries.

Boundary conditions in thermal analysis:
- Dirichlet: Fixed temperature (e.g., cooled surfaces)
- Robin: Convective heat transfer (handled in assembly module)
- Neumann: Prescribed heat flux (natural boundary condition)
"""
import logging
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from typing import Dict, List, Tuple
from numpy.typing import NDArray

from src.mesh.mesh_reader import Mesh
from src.utils.exceptions import ValidationError

logger = logging.getLogger(__name__)


def apply_dirichlet_conditions(A: csr_matrix,
                               F: NDArray[np.float64],
                               mesh: Mesh,
                               node_to_dof: Dict[int, int],
                               dirichlet_boundaries: Dict[int, float]) -> Tuple[csr_matrix, NDArray[np.float64]]:
    """
    @brief Apply Dirichlet boundary conditions using the elimination method.

    @details
    Modifies the linear system $A \\cdot U = F$ to enforce prescribed values
    at boundary nodes identified by physical group IDs.

    The elimination procedure for each Dirichlet DOF $i$ with value $g_i$:
    1. Transfer contribution to RHS: $F[j] -= A[j,i] \\cdot g_i$ for all $j \\neq i$
    2. Zero out row $i$: $A[i, :] = 0$
    3. Set diagonal: $A[i, i] = 1$
    4. Set RHS: $F[i] = g_i$

    This ensures the solution satisfies $U[i] = g_i$ exactly.

    @param A: Global stiffness matrix in CSR sparse format, shape (N, N)
    @param F: Global load vector, shape (N,)
    @param mesh: Mesh object containing boundary edge information
    @param node_to_dof: Dictionary mapping node IDs to DOF indices
    @param dirichlet_boundaries: Dictionary mapping physical boundary IDs to
        prescribed temperature values {physical_id: temperature_value}

    @return Tuple containing:
        - A_bc: Modified stiffness matrix with Dirichlet conditions applied (CSR format)
        - F_bc: Modified load vector with Dirichlet conditions applied

    @raises ValidationError: If matrix A is not square or dimensions mismatch with F

    @example
    >>> # Apply T=300K on boundary with physical ID 1
    >>> dirichlet_bc = {1: 300.0}
    >>> A_bc, F_bc = apply_dirichlet_conditions(A, F, mesh, node_to_dof, dirichlet_bc)
    """
    if A.shape[0] != A.shape[1]:
        raise ValidationError(f"La matrice A doit être carrée, reçu {A.shape}")

    if A.shape[0] != len(F):
        raise ValidationError(f"Dimensions incompatibles: A={A.shape}, F={len(F)}")

    A_bc = A.tolil()
    F_bc = F.copy()

    dirichlet_dofs: List[int] = []
    dirichlet_values: List[float] = []

    for physical_id, imposed_value in dirichlet_boundaries.items():
        boundary_edges = mesh.get_boundary_edges_by_physical(physical_id)

        if len(boundary_edges) == 0:
            logger.warning(f"Aucune arête trouvée pour physical_id={physical_id}")

        for edge_id, edge_elem in boundary_edges:
            global_nodes = edge_elem['nodes']

            for node_id in global_nodes:
                if node_id in node_to_dof:
                    dof = node_to_dof[node_id]
                    if dof not in dirichlet_dofs:
                        dirichlet_dofs.append(dof)
                        dirichlet_values.append(imposed_value)

    logger.info(f"Application de {len(dirichlet_dofs)} conditions de Dirichlet...")

    for dof, value in zip(dirichlet_dofs, dirichlet_values):
        for i in range(A_bc.shape[0]):
            if i != dof:
                F_bc[i] -= A_bc[i, dof] * value

        A_bc[dof, :] = 0
        A_bc[dof, dof] = 1
        F_bc[dof] = value

    A_bc = A_bc.tocsr()

    logger.debug(f"Conditions de Dirichlet appliquées sur {len(dirichlet_dofs)} DOFs")

    return A_bc, F_bc


def get_boundary_nodes(mesh: Mesh,
                       node_to_dof: Dict[int, int],
                       physical_ids: List[int]) -> List[int]:
    """
    @brief Retrieve DOF indices for nodes belonging to specified boundary groups.

    @details
    Iterates through all boundary edges belonging to the specified physical
    groups and collects the unique DOF indices of their nodes.

    This function is useful for:
    - Identifying nodes for post-processing on specific boundaries
    - Extracting boundary temperatures for validation
    - Setting up boundary-specific analysis

    @param mesh: Mesh object containing boundary edge information
    @param node_to_dof: Dictionary mapping node IDs to DOF indices
    @param physical_ids: List of physical group IDs to query

    @return List of unique DOF indices belonging to the specified boundaries

    @example
    >>> # Get all DOFs on inlet (ID=1) and outlet (ID=2) boundaries
    >>> boundary_dofs = get_boundary_nodes(mesh, node_to_dof, [1, 2])
    >>> inlet_temps = U[boundary_dofs]
    """
    boundary_dofs: List[int] = []

    for physical_id in physical_ids:
        boundary_edges = mesh.get_boundary_edges_by_physical(physical_id)

        for edge_id, edge_elem in boundary_edges:
            global_nodes = edge_elem['nodes']

            for node_id in global_nodes:
                if node_id in node_to_dof:
                    dof = node_to_dof[node_id]
                    if dof not in boundary_dofs:
                        boundary_dofs.append(dof)

    logger.debug(f"Trouvé {len(boundary_dofs)} DOFs de bord pour physical_ids={physical_ids}")

    return boundary_dofs


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

    logger.info("Module de conditions aux limites de Dirichlet")
    logger.info("Utiliser avec un maillage et un système assemblé")
