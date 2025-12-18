"""
@file __init__.py
@brief Main package initialization for FEM Thermique
@author HPC-Code-Documenter
@date 2025

@details
FEM Thermique is a finite element analysis package for thermal simulation
of rockets during atmospheric reentry.

Package structure:
- src.core: FEM element formulations, assembly, and solvers
- src.mesh: Mesh reading and generation utilities
- src.physics: Aerothermal models and parametric studies
- src.visualization: Plotting and animation tools
- src.utils: Custom exceptions and utilities

Key classes:
- TriangleP1: P1 triangular finite element
- EdgeP1: P1 edge element for boundary conditions
- Mesh: Container for mesh data

Key functions:
- assemble_global_system: Build global FEM matrices
- solve_linear_system: Solve the assembled system
- apply_dirichlet_conditions: Apply essential BCs
- parametric_velocity_study: Run velocity sweep analysis

@example
>>> from src import (
...     read_gmsh_mesh, create_node_mapping,
...     assemble_global_system, solve_linear_system,
...     parametric_velocity_study
... )
>>> mesh = read_gmsh_mesh("rocket.msh")
>>> node_to_dof, n_dofs = create_node_mapping(mesh)
"""

from src.core import (
    TriangleP1,
    EdgeP1,
    assemble_global_system,
    solve_linear_system,
    apply_dirichlet_conditions,
)
from src.mesh import read_gmsh_mesh, create_node_mapping, Mesh
from src.physics import parametric_velocity_study

__version__ = "1.0.0"
__all__ = [
    "TriangleP1",
    "EdgeP1",
    "assemble_global_system",
    "solve_linear_system",
    "apply_dirichlet_conditions",
    "read_gmsh_mesh",
    "create_node_mapping",
    "Mesh",
    "parametric_velocity_study",
]
