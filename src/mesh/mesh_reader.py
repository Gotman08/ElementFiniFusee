"""
@file mesh_reader.py
@brief GMSH mesh file reader for finite element analysis
@author HPC-Code-Documenter
@date 2025

@details
This module provides functionality to read GMSH mesh files in MSH format
version 2.2 (ASCII) and convert them into a usable mesh data structure
for finite element analysis.

Supported GMSH element types:
- Type 1: 2-node line (edge) - used for boundary conditions
- Type 2: 3-node triangle - used for 2D domain discretization

The mesh file format sections parsed:
- $Nodes: Node coordinates (x, y, z)
- $Elements: Element connectivity and physical group tags
- $PhysicalNames: Named boundary and domain groups

Physical groups are used to identify:
- Domain regions (e.g., different materials)
- Boundary segments (e.g., inlet, outlet, walls)
"""
import logging
import os
import numpy as np
from typing import Dict, List, Tuple, Any
from numpy.typing import NDArray

from src.utils.exceptions import MeshError, ValidationError

logger = logging.getLogger(__name__)


class Mesh:
    """
    @brief Container class for finite element mesh data.

    @details
    Stores mesh topology and geometry information including:
    - Node coordinates in 2D or 3D space
    - Element connectivity (triangles for domain, edges for boundaries)
    - Physical group associations for boundary condition application

    The class provides accessor methods for retrieving specific element
    types and coordinate data needed by the FEM assembly routines.
    """

    def __init__(self) -> None:
        """
        @brief Initialize an empty mesh structure.

        @details
        Creates empty dictionaries for nodes, elements, and physical groups.
        Default dimension is set to 2D.
        """
        self.nodes: Dict[int, List[float]] = {}
        self.elements: Dict[int, Dict[str, Any]] = {}
        self.physical_groups: Dict[int, str] = {}
        self.dimension: int = 2

    def get_triangles(self) -> Dict[int, Dict[str, Any]]:
        """
        @brief Retrieve all triangular elements from the mesh.

        @details
        Filters elements by GMSH type 2 (3-node triangle).
        These elements form the computational domain for 2D FEM analysis.

        @return Dictionary mapping element IDs to element data dictionaries
            containing 'type', 'tags', and 'nodes' keys
        """
        return {eid: elem for eid, elem in self.elements.items()
                if elem['type'] == 2}

    def get_edges(self) -> Dict[int, Dict[str, Any]]:
        """
        @brief Retrieve all edge elements from the mesh.

        @details
        Filters elements by GMSH type 1 (2-node line).
        Edge elements define boundaries where conditions are applied.

        @return Dictionary mapping element IDs to element data dictionaries
            containing 'type', 'tags', and 'nodes' keys
        """
        return {eid: elem for eid, elem in self.elements.items()
                if elem['type'] == 1}

    def get_boundary_edges_by_physical(self, physical_id: int) -> List[Tuple[int, Dict[str, Any]]]:
        """
        @brief Retrieve edges belonging to a specific physical group.

        @details
        Filters edge elements by their physical group tag (first tag in GMSH format).
        Used to identify boundary segments for applying boundary conditions.

        @param physical_id: Integer ID of the physical group to query

        @return List of tuples (element_id, element_data) for matching edges

        @example
        >>> # Get all edges on the external boundary (physical ID 1)
        >>> external_edges = mesh.get_boundary_edges_by_physical(1)
        """
        edges = []
        for eid, elem in self.get_edges().items():
            if len(elem['tags']) > 0 and elem['tags'][0] == physical_id:
                edges.append((eid, elem))
        return edges

    def get_node_coords(self, node_ids: List[int]) -> NDArray[np.float64]:
        """
        @brief Retrieve coordinates for a list of nodes.

        @details
        Extracts the spatial coordinates for specified nodes, truncated
        to the mesh dimension (2D or 3D). Coordinates are returned in
        the same order as the input node IDs.

        @param node_ids: List of node IDs to retrieve coordinates for

        @return NumPy array of shape (N, dimension) containing coordinates

        @raises ValidationError: If any node ID does not exist in the mesh

        @example
        >>> coords = mesh.get_node_coords([1, 2, 3])
        >>> # Returns array of shape (3, 2) for 2D mesh
        """
        for nid in node_ids:
            if nid not in self.nodes:
                raise ValidationError(f"Noeud {nid} n'existe pas dans le maillage")

        coords = np.array([self.nodes[nid][:self.dimension] for nid in node_ids],
                         dtype=np.float64)
        return coords


def read_gmsh_mesh(filename: str) -> Mesh:
    """
    @brief Read a GMSH mesh file in MSH 2.2 ASCII format.

    @details
    Parses a GMSH mesh file and extracts:
    - Node coordinates from $Nodes section
    - Element connectivity from $Elements section
    - Physical group names from $PhysicalNames section (optional)

    The parser handles the standard MSH 2.2 format with:
    - Node format: node_id x y z
    - Element format: elem_id type num_tags tag1 ... tagN node1 ... nodeM

    @param filename: Path to the .msh file to read

    @return Mesh object populated with nodes, elements, and physical groups

    @raises MeshError: If file does not exist, has invalid format, or contains no data

    @example
    >>> mesh = read_gmsh_mesh("rocket.msh")
    >>> print(f"Loaded {len(mesh.nodes)} nodes")
    """
    if not os.path.exists(filename):
        raise MeshError(f"Fichier maillage introuvable: {filename}")

    if not filename.endswith('.msh'):
        logger.warning(f"Extension de fichier non standard: {filename} (attendu: .msh)")

    mesh = Mesh()
    logger.info(f"Lecture du maillage: {filename}")

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            line = f.readline()

            while line:
                line = line.strip()

                if line == '$Nodes':
                    try:
                        num_nodes = int(f.readline().strip())
                        logger.debug(f"Lecture de {num_nodes} noeuds...")
                    except ValueError as e:
                        raise MeshError(f"Format invalide pour le nombre de noeuds: {e}")

                    for i in range(num_nodes):
                        parts = f.readline().strip().split()
                        try:
                            node_id = int(parts[0])
                            coords = [float(parts[1]), float(parts[2]), float(parts[3])]
                            mesh.nodes[node_id] = coords
                        except (IndexError, ValueError) as e:
                            raise MeshError(f"Format invalide pour le noeud {i+1}: {e}")

                    line = f.readline()

                elif line == '$Elements':
                    try:
                        num_elements = int(f.readline().strip())
                        logger.debug(f"Lecture de {num_elements} éléments...")
                    except ValueError as e:
                        raise MeshError(f"Format invalide pour le nombre d'éléments: {e}")

                    for i in range(num_elements):
                        parts = f.readline().strip().split()
                        try:
                            elem_id = int(parts[0])
                            elem_type = int(parts[1])
                            num_tags = int(parts[2])
                            tags = [int(parts[3 + j]) for j in range(num_tags)]
                            nodes = [int(parts[3 + num_tags + j])
                                    for j in range(len(parts) - 3 - num_tags)]

                            mesh.elements[elem_id] = {
                                'type': elem_type,
                                'tags': tags,
                                'nodes': nodes
                            }
                        except (IndexError, ValueError) as e:
                            raise MeshError(f"Format invalide pour l'élément {i+1}: {e}")

                    line = f.readline()

                elif line == '$PhysicalNames':
                    try:
                        num_groups = int(f.readline().strip())
                        logger.debug(f"Lecture de {num_groups} groupes physiques...")
                    except ValueError as e:
                        raise MeshError(f"Format invalide pour le nombre de groupes: {e}")

                    for i in range(num_groups):
                        parts = f.readline().strip().split()
                        try:
                            dimension = int(parts[0])
                            physical_id = int(parts[1])
                            name = parts[2].strip('"')
                            mesh.physical_groups[physical_id] = name
                            if dimension > mesh.dimension:
                                mesh.dimension = dimension
                        except (IndexError, ValueError) as e:
                            raise MeshError(f"Format invalide pour le groupe physique {i+1}: {e}")

                    line = f.readline()

                else:
                    line = f.readline()

    except IOError as e:
        raise MeshError(f"Erreur de lecture du fichier {filename}: {e}")

    if len(mesh.nodes) == 0:
        raise MeshError("Maillage vide: aucun noeud lu")

    if len(mesh.elements) == 0:
        raise MeshError("Maillage vide: aucun élément lu")

    num_triangles = len(mesh.get_triangles())
    num_edges = len(mesh.get_edges())

    logger.info(f"Maillage lu: {len(mesh.nodes)} noeuds, {num_triangles} triangles, {num_edges} arêtes")

    return mesh


def create_node_mapping(mesh: Mesh) -> Tuple[Dict[int, int], int]:
    """
    @brief Create a mapping from GMSH node IDs to sequential DOF indices.

    @details
    GMSH node IDs may not be sequential or start from 0. This function
    creates a contiguous mapping from node IDs to degree of freedom (DOF)
    indices suitable for matrix assembly.

    Only nodes that belong to triangular elements are included in the
    mapping, ensuring the DOF count matches the problem size.

    @param mesh: Mesh object containing elements and nodes

    @return Tuple containing:
        - node_to_dof: Dictionary mapping node IDs to DOF indices (0-based)
        - num_dofs: Total number of degrees of freedom

    @raises ValidationError: If the mesh contains no triangular elements

    @example
    >>> node_to_dof, num_dofs = create_node_mapping(mesh)
    >>> print(f"System size: {num_dofs} DOFs")
    >>> dof_index = node_to_dof[node_id]  # Get DOF for a specific node
    """
    triangles = mesh.get_triangles()

    if len(triangles) == 0:
        raise ValidationError("Le maillage ne contient aucun triangle")

    all_nodes = set()
    for elem in triangles.values():
        all_nodes.update(elem['nodes'])

    node_to_dof = {node_id: idx for idx, node_id in enumerate(sorted(all_nodes))}

    logger.debug(f"Mapping créé: {len(node_to_dof)} DOFs")

    return node_to_dof, len(node_to_dof)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

    import sys
    if len(sys.argv) > 1:
        try:
            mesh = read_gmsh_mesh(sys.argv[1])
            logger.info(f"Maillage lu avec succès:")
            logger.info(f"  - {len(mesh.nodes)} noeuds")
            logger.info(f"  - {len(mesh.get_triangles())} triangles")
            logger.info(f"  - {len(mesh.get_edges())} arêtes")
            logger.info(f"  - Dimension: {mesh.dimension}D")
            if mesh.physical_groups:
                logger.info(f"  - Groupes physiques: {mesh.physical_groups}")
        except MeshError as e:
            logger.error(f"Erreur de lecture: {e}")
    else:
        logger.warning("Usage: python mesh_reader.py <fichier.msh>")
