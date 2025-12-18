"""
@file __init__.py
@brief Utilities module initialization
@author HPC-Code-Documenter
@date 2025

@details
The utils module provides support utilities for the FEM thermal project.

Custom exceptions (exceptions.py):
- FEMError: Base exception for all FEM errors
- MeshError: Mesh reading and validation errors
- SolverError: Linear system solution errors
- AssemblyError: Matrix assembly errors
- ValidationError: Input parameter validation errors
- ElementError: Element-level computation errors

Exception hierarchy allows for precise error handling:
```python
try:
    run_analysis()
except ValidationError as e:
    print(f"Invalid input: {e}")
except SolverError as e:
    print(f"Solver failed: {e}")
except FEMError as e:
    print(f"General FEM error: {e}")
```

@example
>>> from src.utils import ValidationError, MeshError
>>> if kappa <= 0:
...     raise ValidationError("kappa must be positive")
"""

from src.utils.exceptions import (
    FEMError,
    MeshError,
    SolverError,
    AssemblyError,
    ValidationError,
    ElementError,
)

__all__ = [
    "FEMError",
    "MeshError",
    "SolverError",
    "AssemblyError",
    "ValidationError",
    "ElementError",
]
