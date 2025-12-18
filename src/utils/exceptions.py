"""
@file exceptions.py
@brief Custom exception hierarchy for FEM thermal analysis
@author HPC-Code-Documenter
@date 2025

@details
This module defines a hierarchical set of custom exceptions for explicit
and informative error handling throughout the FEM thermal analysis project.

Exception Hierarchy:
```
FEMError (base)
    |-- MeshError (mesh reading, format, validity)
    |-- SolverError (convergence, singularity)
    |-- AssemblyError (matrix assembly issues)
    |-- ValidationError (invalid input parameters)
    |-- ElementError (degenerate elements, zero Jacobian)
```

Using specific exception types allows for:
- Precise error catching at appropriate levels
- Informative error messages for debugging
- Clean separation of error sources

@example
>>> from src.utils.exceptions import ValidationError, MeshError
>>> if kappa <= 0:
...     raise ValidationError(f"kappa must be positive, got {kappa}")
"""


class FEMError(Exception):
    """
    @brief Base exception class for all FEM-related errors.

    @details
    All custom exceptions in the FEM project inherit from this class,
    allowing for catch-all exception handling when appropriate.

    @example
    >>> try:
    ...     run_fem_analysis()
    ... except FEMError as e:
    ...     logger.error(f"FEM analysis failed: {e}")
    """
    pass


class MeshError(FEMError):
    """
    @brief Exception for mesh-related errors.

    @details
    Raised when issues occur during mesh operations:
    - File not found or unreadable
    - Invalid mesh format
    - Corrupted mesh data
    - Empty mesh (no nodes or elements)

    @example
    >>> if not os.path.exists(filename):
    ...     raise MeshError(f"Mesh file not found: {filename}")
    """
    pass


class SolverError(FEMError):
    """
    @brief Exception for linear solver errors.

    @details
    Raised when the linear system solution encounters issues:
    - Iterative solver fails to converge
    - Matrix singularity detected
    - Invalid solver configuration
    - Numerical instability

    @example
    >>> if info > 0:
    ...     raise SolverError(f"CG failed to converge after {info} iterations")
    """
    pass


class AssemblyError(FEMError):
    """
    @brief Exception for matrix assembly errors.

    @details
    Raised during global system assembly when:
    - Node-to-DOF mapping is inconsistent
    - Element connectivity references non-existent nodes
    - Boundary condition references invalid physical groups

    @example
    >>> if node_id not in node_to_dof:
    ...     raise AssemblyError(f"Node {node_id} not found in DOF mapping")
    """
    pass


class ValidationError(FEMError):
    """
    @brief Exception for input validation errors.

    @details
    Raised when function inputs fail validation:
    - Negative physical parameters (kappa, alpha)
    - Array shape mismatches
    - Invalid method selection
    - Out-of-range values

    @example
    >>> if kappa <= 0:
    ...     raise ValidationError(f"kappa must be positive, got {kappa}")
    """
    pass


class ElementError(FEMError):
    """
    @brief Exception for finite element computation errors.

    @details
    Raised when element-level computations fail:
    - Degenerate element (zero or negative area)
    - Zero Jacobian determinant
    - Invalid element node ordering
    - Collapsed element geometry

    @example
    >>> if abs(det_J) < 1e-14:
    ...     raise ElementError(f"Degenerate element: det(J) = {det_J}")
    """
    pass
