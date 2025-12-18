"""
@file __init__.py
@brief Mesh generators module initialization
@author HPC-Code-Documenter
@date 2025

@details
The generators submodule provides mesh generation utilities that create
meshes programmatically without requiring external meshing software.

Available generators:
- generate_rocket_mesh: Structured triangular mesh for rocket geometry
  - Supports nose-first and tail-first reentry configurations
  - Automatic quality optimization for quasi-equilateral triangles
  - GMSH MSH 2.2 format output

Mesh quality metrics:
- $Q_T$ criterion from course material
- Aspect ratio monitoring
- Automatic refinement for target quality

Physical groups generated:
- ID 1 (Gamma_F): Robin boundary (convective flux)
- ID 2 (Gamma_D): Dirichlet boundary (fixed temperature)
- ID 3 (Gamma_N): Neumann boundary (insulated)
- ID 4 (Gamma_axis): Axis/tip boundary
- ID 10 (Omega): Domain interior

@example
>>> from src.mesh.generators import generate_rocket_mesh
>>> stats = generate_rocket_mesh(
...     output_file="rocket.msh",
...     mode="tail-first"
... )
"""

from src.mesh.generators.rocket_mesh import generate_rocket_mesh

__all__ = [
    "generate_rocket_mesh",
]
