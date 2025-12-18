"""
@file __init__.py
@brief Core FEM module initialization
@author HPC-Code-Documenter
@date 2025

@details
The core module contains the fundamental finite element method components:

Element formulations (fem_elements.py):
- TriangleP1: P1 triangular element with linear shape functions
- EdgeP1: P1 edge element for boundary integrals

System assembly (assembly.py):
- assemble_global_system: Build global stiffness matrix and load vector
- assemble_volumetric_load: Add volumetric source contributions

Linear solvers (solver.py):
- solve_linear_system: Direct and iterative solvers
- extract_solution_stats: Solution statistics extraction

Boundary conditions (boundary_conditions.py):
- apply_dirichlet_conditions: Essential BC application
- get_boundary_nodes: Retrieve boundary DOF indices

Mathematical formulation:
The core module implements the discrete thermal problem:
$A \\cdot U = F$

where:
- $A = K + M_R$ (stiffness + Robin mass)
- $F = F_R + F_V$ (Robin + volumetric loads)
"""

from src.core.fem_elements import TriangleP1, EdgeP1
from src.core.assembly import assemble_global_system, assemble_volumetric_load
from src.core.solver import solve_linear_system, extract_solution_stats
from src.core.boundary_conditions import apply_dirichlet_conditions, get_boundary_nodes

__all__ = [
    "TriangleP1",
    "EdgeP1",
    "assemble_global_system",
    "assemble_volumetric_load",
    "solve_linear_system",
    "extract_solution_stats",
    "apply_dirichlet_conditions",
    "get_boundary_nodes",
]
