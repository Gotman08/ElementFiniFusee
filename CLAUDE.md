# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Finite Element Method (FEM) solver for thermal analysis of rocket structures during atmospheric reentry. Implements P1 (linear) triangular elements with Robin boundary conditions for convective heat transfer.

## Build & Run Commands

```bash
# Install dependencies
pip install -e .                    # Editable install with all dependencies
pip install -e ".[dev]"             # Include dev dependencies (pytest)

# Run simulations
python scripts/demo_reentry.py      # Live reentry animation demo
python scripts/run_parametric_study.py  # Parametric velocity study

# Generate mesh (requires GMSH installed)
gmsh -2 data/meshes/rocket.geo -o data/meshes/rocket_mesh.msh

# Run tests
pytest                              # All tests
pytest tests/test_foo.py            # Single test file
pytest -v --tb=short                # Verbose with short traceback
```

## Architecture

### Core FEM Pipeline (`src/core/`)
- **fem_elements.py**: P1 shape functions for triangles (TriangleP1) and edges (EdgeP1). Computes local stiffness matrices and boundary mass matrices.
- **assembly.py**: Assembles global system `A·U = F` with Robin boundary terms. Critical: `assemble_global_system()` handles both volume stiffness and surface convection terms.
- **boundary_conditions.py**: Applies Dirichlet BC via elimination method. Modifies assembled system in-place.
- **solver.py**: Sparse linear solver with direct (LU), CG, and GMRES methods.

### Mesh Handling (`src/mesh/`)
- **mesh_reader.py**: Parses GMSH MSH 2.2 ASCII format. The `Mesh` class stores nodes, elements, and physical groups.
- **generators/**: Python scripts to generate meshes programmatically (ariane5, rocket shapes).

### Physics Models (`src/physics/`)
- **parametric_study.py**: Aerothermal correlations (Reynolds, Nusselt, recovery temperature). Key function: `compute_aerothermal_parameters(V)` returns (alpha, u_E) for given velocity.
- **reentry_profile.py**: Generates velocity/altitude profiles for reentry trajectories.

### Visualization (`src/visualization/`)
- **animation.py**: Live matplotlib animation of temperature evolution.
- **plotting.py**: Static plots for temperature fields and parametric results.

## Key Mathematical Formulation

The thermal problem with Robin BC:
```
-div(κ ∇u) = 0           in Ω
-κ ∂u/∂n = α(u - u_E)    on Γ_Robin (convection)
u = u_D                  on Γ_Dirichlet (fixed temp)
```

Robin terms add to both matrix and RHS:
- `A[i,j] += ∫ α φ_i φ_j dσ` (boundary mass matrix)
- `F[i] += ∫ α u_E φ_i dσ` (boundary load vector)

The velocity dependence enters through aerothermal parameters:
- `α(V)` via Nusselt correlation (turbulent: Nu = 0.037 Re^0.8 Pr^1/3)
- `u_E(V)` via recovery temperature (T_aw = T_∞ + r·V²/(2·c_p))

## Data Directories

- `data/meshes/`: GMSH mesh files (.msh, .geo)
- `data/output/figures/`: Generated plots and animations
- `data/output/results/`: CSV exports of parametric studies

## Physical Constants

Default values in `src/physics/parametric_study.py`:
- `KAPPA_material = 160.0` W/(m·K) - thermal conductivity (aluminum-like)
- `RHO_inf = 0.02` kg/m³ - air density at ~30 km altitude
- `T_inf = 230.0` K - ambient temperature
- `r_recovery = 0.89` - turbulent recovery factor
