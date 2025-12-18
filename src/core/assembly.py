"""
@file assembly.py
@brief Global linear system assembly for finite element method
@author HPC-Code-Documenter
@date 2025

@details
This module implements the assembly of the global linear system $A \\cdot U = F$
for thermal analysis using the Finite Element Method.

The assembly process includes:
1. Volume assembly: Stiffness matrix from thermal diffusion
   $A_{vol}[i,j] = \\int_\\Omega \\kappa \\nabla\\phi_i \\cdot \\nabla\\phi_j \\, d\\Omega$

2. Boundary assembly: Robin (convective) boundary conditions
   $A_{robin}[i,j] = \\int_{\\Gamma} \\alpha \\phi_i \\phi_j \\, d\\sigma$
   $F_{robin}[i] = \\int_{\\Gamma} \\alpha u_E \\phi_i \\, d\\sigma$

3. Volumetric source assembly:
   $F_{vol}[i] = \\int_\\Omega f \\phi_i \\, d\\Omega$

The implementation uses sparse matrices (LIL format for assembly,
CSR format for solving) for memory efficiency with large meshes.
"""
import logging
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from typing import Dict, Tuple, Union, Callable
from numpy.typing import NDArray

from src.mesh.mesh_reader import Mesh
from src.core.fem_elements import TriangleP1, EdgeP1
from src.utils.exceptions import AssemblyError, ValidationError

logger = logging.getLogger(__name__)


def assemble_global_system(
    mesh: Mesh,
    node_to_dof: Dict[int, int],
    kappa: float,
    robin_boundaries: Dict[int, Tuple[float, float]],
    radiation_boundaries: Dict[int, Tuple[float, float, float]] = None,
    U_current: NDArray[np.float64] = None
) -> Tuple[csr_matrix, NDArray[np.float64]]:
    """
    @brief Assemble the complete global linear system with Robin and optional radiation boundary conditions.

    @details
    Assembles the global system $A \\cdot U = F$ where:
    - $A = A_{volume} + A_{robin}$: Global stiffness matrix with convection (and optional radiation) terms
    - $F = F_{robin}$: Load vector from boundary convection (and optional radiation)

    The volume contribution comes from the thermal diffusion term:
    $A_{vol}[i,j] = \\int_\\Omega \\kappa \\nabla\\phi_i \\cdot \\nabla\\phi_j \\, d\\Omega$

    The Robin boundary contribution models convective heat transfer:
    $A_{robin}[i,j] = \\int_{\\Gamma_R} \\alpha \\phi_i \\phi_j \\, d\\sigma$
    $F_{robin}[i] = \\int_{\\Gamma_R} \\alpha u_E \\phi_i \\, d\\sigma$

    When radiation is included (radiation_boundaries provided and U_current is not None):
    - The radiation term is linearized using Picard iteration: α_rad = ε·σ·T³
    - Effective coefficient becomes: α_eff = α_conv + α_rad
    - Radiation load term added: F_rad = ∫ ε·σ·T_∞⁴ φ_i dσ

    @param mesh: Mesh object containing nodes, elements, and boundary information
    @param node_to_dof: Dictionary mapping node IDs to degree of freedom indices
    @param kappa: Thermal conductivity coefficient $\\kappa$ [W/(m.K)]
    @param robin_boundaries: Dictionary mapping physical boundary IDs to (alpha, u_E) tuples
        - alpha: Convection heat transfer coefficient [W/(m^2.K)]
        - u_E: External/recovery temperature [K]
    @param radiation_boundaries: Optional dict of {physical_id: (epsilon, T_inf, sigma)}
        - epsilon: Surface emissivity (0-1)
        - T_inf: Ambient temperature [K]
        - sigma: Stefan-Boltzmann constant [W/(m²·K⁴)]
        If None, no radiation is included.
    @param U_current: Current temperature field for Picard linearization [K]
        Required if radiation_boundaries is provided. If None, radiation is ignored.

    @return Tuple containing:
        - A: Sparse CSR matrix of shape (N, N) representing the global stiffness matrix
        - F: NumPy array of shape (N,) representing the global load vector

    @raises ValidationError: If kappa <= 0, alpha < 0, or invalid radiation parameters
    @raises AssemblyError: If a node in an element is not found in node_to_dof mapping

    @example
    >>> # Without radiation
    >>> robin_bc = {1: (10.0, 300.0)}
    >>> A, F = assemble_global_system(mesh, node_to_dof, kappa=1.0, robin_boundaries=robin_bc)
    >>>
    >>> # With radiation (for Picard iteration)
    >>> rad_bc = {1: (0.85, 230.0, 5.67e-8)}
    >>> A, F = assemble_global_system(mesh, node_to_dof, 160.0, robin_bc, rad_bc, U_current=U0)
    """
    # Validation
    if kappa <= 0:
        raise ValidationError(f"kappa doit être positif, reçu {kappa}")

    for physical_id, (alpha, u_E) in robin_boundaries.items():
        if alpha < 0:
            raise ValidationError(f"alpha doit être >= 0 pour physical_id={physical_id}, reçu {alpha}")

    if radiation_boundaries is not None:
        for physical_id, (epsilon, T_inf, sigma) in radiation_boundaries.items():
            if not (0.0 <= epsilon <= 1.0):
                raise ValidationError(f"epsilon doit être dans [0,1], reçu {epsilon} pour physical_id={physical_id}")
            if T_inf < 0:
                raise ValidationError(f"T_inf doit être >= 0, reçu {T_inf} pour physical_id={physical_id}")
            if sigma <= 0:
                raise ValidationError(f"sigma doit être > 0, reçu {sigma} pour physical_id={physical_id}")

    num_dofs = len(node_to_dof)

    if num_dofs == 0:
        raise ValidationError("node_to_dof est vide")

    A = lil_matrix((num_dofs, num_dofs))
    F = np.zeros(num_dofs)

    # 1. Volume assembly (thermal diffusion)
    triangles = mesh.get_triangles()
    logger.info(f"Assemblage volumique: {len(triangles)} triangles...")

    for elem_id, elem in triangles.items():
        global_nodes = elem['nodes']
        coords = mesh.get_node_coords(global_nodes)
        K_elem = TriangleP1.local_stiffness_matrix(coords, kappa)

        try:
            local_dofs = [node_to_dof[nid] for nid in global_nodes]
        except KeyError as e:
            raise AssemblyError(f"Noeud {e} non trouvé dans node_to_dof (élément {elem_id})")

        for i in range(3):
            for j in range(3):
                A[local_dofs[i], local_dofs[j]] += K_elem[i, j]

    # 2. Boundary assembly (Robin + optional Radiation)
    for physical_id, (alpha, u_E) in robin_boundaries.items():
        boundary_edges = mesh.get_boundary_edges_by_physical(physical_id)

        # Check if this boundary also has radiation
        has_radiation = radiation_boundaries is not None and physical_id in radiation_boundaries
        include_radiation = has_radiation and (U_current is not None)

        if include_radiation:
            epsilon, T_inf, sigma = radiation_boundaries[physical_id]
            logger.info(f"Assemblage Robin+Radiation (Physical ID {physical_id}): "
                       f"{len(boundary_edges)} aretes, alpha={alpha}, u_E={u_E}, "
                       f"epsilon={epsilon}, T_inf={T_inf}")
        else:
            logger.info(f"Assemblage Robin (Physical ID {physical_id}): "
                       f"{len(boundary_edges)} aretes, alpha={alpha}, u_E={u_E}")

        for edge_id, edge_elem in boundary_edges:
            global_nodes = edge_elem['nodes']
            coords = mesh.get_node_coords(global_nodes)

            # Get local DOFs
            try:
                local_dofs = [node_to_dof[nid] for nid in global_nodes]
            except KeyError as e:
                raise AssemblyError(f"Noeud {e} non trouvé dans node_to_dof (arête {edge_id})")

            # Effective convection coefficient (includes radiation if active)
            alpha_eff = alpha

            if include_radiation:
                # Average temperature on this edge for Picard linearization
                T_nodes = [U_current[dof] for dof in local_dofs]
                T_avg = sum(T_nodes) / len(T_nodes)

                # Clip temperature to reasonable range
                if T_avg < 200.0 or T_avg > 5000.0:
                    logger.warning(
                        f"Température hors bornes détectée: T_avg = {T_avg:.1f} K. "
                        f"Clipping à [200, 5000] K."
                    )
                T_avg = np.clip(T_avg, 200.0, 5000.0)

                # Linearized radiation coefficient: α_rad = ε·σ·T³
                alpha_rad = epsilon * sigma * T_avg**3
                alpha_eff = alpha + alpha_rad

                logger.debug(f"  Edge {edge_id}: T_avg={T_avg:.1f} K, "
                            f"alpha_conv={alpha:.1f}, alpha_rad={alpha_rad:.1f}")

            # Boundary mass matrix with effective coefficient
            M_elem = EdgeP1.local_mass_matrix(coords, alpha_eff)

            # Load vector: convection term
            F_elem = EdgeP1.local_load_vector(coords, alpha, u_E)

            # Load vector: radiation ambient term (if active)
            if include_radiation:
                # F_rad = ∫ ε·σ·T_∞⁴ φ_i dσ
                T_inf_power4 = T_inf**4
                F_rad = EdgeP1.local_load_vector(coords, epsilon * sigma, T_inf_power4)
                F_elem += F_rad

            # Assemble into global system
            for i in range(2):
                F[local_dofs[i]] += F_elem[i]
                for j in range(2):
                    A[local_dofs[i], local_dofs[j]] += M_elem[i, j]

    A = A.tocsr()

    logger.info(f"Assemblage terminé: Système {num_dofs} × {num_dofs}")
    logger.debug(f"  - Nombre d'éléments non-nuls: {A.nnz}")
    logger.debug(f"  - Norme du second membre: {np.linalg.norm(F):.2e}")

    return A, F


# Backward compatibility alias
# The old function name is now deprecated, use assemble_global_system() with radiation_boundaries parameter
assemble_global_system_with_radiation = assemble_global_system


def assemble_volumetric_load(mesh: Mesh,
                             node_to_dof: Dict[int, int],
                             source_term: Union[float, Callable[[float, float], float]]) -> NDArray[np.float64]:
    """
    @brief Assemble the volumetric load vector from a source term.

    @details
    Computes the volumetric contribution to the load vector:
    $F_{vol}[i] = \\int_\\Omega f(x,y) \\phi_i \\, d\\Omega$

    The integration uses the centroid quadrature rule which is exact
    for constant source terms and provides first-order accuracy for
    variable source terms.

    The source term $f(x,y)$ can represent:
    - Internal heat generation [W/m^3]
    - Radiative heating absorbed in volume
    - Any distributed heat source

    @param mesh: Mesh object containing nodes and triangular elements
    @param node_to_dof: Dictionary mapping node IDs to degree of freedom indices
    @param source_term: Either a constant float value or a callable f(x, y) -> float
        representing the volumetric heat source [W/m^3]

    @return NumPy array of shape (N,) representing the volumetric load vector

    @raises AssemblyError: If a node in an element is not found in node_to_dof mapping

    @example
    >>> # Constant source term
    >>> F_vol = assemble_volumetric_load(mesh, node_to_dof, source_term=1000.0)
    >>> # Variable source term (Gaussian heat source)
    >>> def gaussian_source(x, y):
    ...     return 1000.0 * np.exp(-(x**2 + y**2) / 0.1)
    >>> F_vol = assemble_volumetric_load(mesh, node_to_dof, source_term=gaussian_source)
    """
    num_dofs = len(node_to_dof)
    F_vol = np.zeros(num_dofs)

    triangles = mesh.get_triangles()
    logger.debug(f"Assemblage charge volumique: {len(triangles)} triangles")

    for elem_id, elem in triangles.items():
        global_nodes = elem['nodes']
        coords = mesh.get_node_coords(global_nodes)

        _, area = TriangleP1.physical_gradients(coords)

        points, weights = TriangleP1.quadrature_volume()

        xi_q, eta_q = points[0]
        phi_q = TriangleP1.shape_functions(xi_q, eta_q)
        x_phys = phi_q @ coords

        if callable(source_term):
            f_value = source_term(x_phys[0], x_phys[1])
        else:
            f_value = source_term

        F_elem = f_value * weights[0] * 2 * area * phi_q

        try:
            local_dofs = [node_to_dof[nid] for nid in global_nodes]
        except KeyError as e:
            raise AssemblyError(f"Noeud {e} non trouvé dans node_to_dof (élément {elem_id})")

        for i in range(3):
            F_vol[local_dofs[i]] += F_elem[i]

    return F_vol


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

    logger.info("Test du module d'assemblage")
    logger.info("=" * 60)
    logger.info("Créer un maillage .msh pour tester l'assemblage complet")
    logger.info("Exemple: maillage rectangulaire avec bords Robin")
    logger.info("=" * 60)
