"""
@file __init__.py
@brief Mesh module initialization
@author HPC-Code-Documenter
@date 2025

@details
The mesh module handles mesh input/output and mesh generation:

Mesh reading (mesh_reader.py):
- read_gmsh_mesh: Parse GMSH MSH 2.2 format files
- create_node_mapping: Build node-to-DOF index mapping
- Mesh: Container class for mesh data

Mesh generation (generators/):
- generate_rocket_mesh: Create structured rocket geometry meshes

Supported mesh elements:
- Type 1: 2-node line (boundary edges)
- Type 2: 3-node triangle (domain elements)

Physical groups are used to identify:
- Domain regions (different materials)
- Boundary segments (BCs application)

@example
>>> from src.mesh import read_gmsh_mesh, create_node_mapping
>>> mesh = read_gmsh_mesh("rocket.msh")
>>> node_to_dof, n_dofs = create_node_mapping(mesh)
>>> print(f"Mesh has {n_dofs} degrees of freedom")
"""

from src.mesh.mesh_reader import read_gmsh_mesh, create_node_mapping, Mesh

__all__ = [
    "read_gmsh_mesh",
    "create_node_mapping",
    "Mesh",
]
