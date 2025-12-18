"""
Mesh utility functions for common operations.

This module provides helper functions for mesh quality assessment
and standard mesh loading patterns to reduce code duplication.
"""
import numpy as np
from typing import Tuple, Dict
from src.mesh.mesh_reader import Mesh, read_gmsh_mesh, create_node_mapping


def compute_triangle_quality(coords: np.ndarray) -> float:
    """
    @brief Compute triangle quality using the course-defined criterion.

    @details
    Calculates the quality factor $Q_T$ for a triangle:
    $Q_T = \\frac{\\sqrt{3}}{6} \\times \\frac{h_T}{r_T}$

    where:
    - $h_T$ is the longest edge length
    - $r_T$ is the inscribed circle radius: $r_T = A/s$
    - $A$ is the triangle area (Heron's formula)
    - $s$ is the semi-perimeter

    @param coords: NumPy array of shape (3, 2) with vertex coordinates

    @return Quality factor $Q_T$ (1.0 for equilateral, >1 for degraded triangles)
        Returns infinity for degenerate triangles
    """
    a = np.linalg.norm(coords[1] - coords[0])
    b = np.linalg.norm(coords[2] - coords[1])
    c = np.linalg.norm(coords[0] - coords[2])

    h_T = max(a, b, c)

    s = (a + b + c) / 2

    area_sq = s * (s - a) * (s - b) * (s - c)
    if area_sq <= 0:
        return float('inf')
    area = np.sqrt(area_sq)

    r_T = area / s

    Q_T = (np.sqrt(3) / 6) * (h_T / r_T)

    return Q_T


def load_mesh_with_mapping(mesh_file: str) -> Tuple[Mesh, Dict[int, int], int]:
    """
    @brief Standard pattern for loading mesh with DOF mapping.

    @details
    This helper encapsulates the common pattern of:
    1. Reading a GMSH mesh file
    2. Creating the node-to-DOF mapping

    This reduces code duplication across scripts and improves consistency.

    @param mesh_file: Path to the GMSH mesh file (.msh)

    @return Tuple containing:
        - mesh: Loaded Mesh object
        - node_to_dof: Dictionary mapping node IDs to degree of freedom indices
        - num_dofs: Total number of degrees of freedom

    @example
    >>> mesh, node_to_dof, num_dofs = load_mesh_with_mapping("data/meshes/rocket_mesh.msh")
    >>> print(f"Loaded mesh with {num_dofs} DOFs")
    """
    mesh = read_gmsh_mesh(mesh_file)
    node_to_dof, num_dofs = create_node_mapping(mesh)
    return mesh, node_to_dof, num_dofs
