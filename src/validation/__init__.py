"""
Thermal validation package for rocket reentry FEM analysis.

This package provides comprehensive validation tools for thermal analysis results,
identifying physical correctness issues, missing physics effects, and providing
engineering recommendations.

Main Components:
    - ThermalValidator: Core validation engine
    - MaterialProperties: TPS material database
    - ValidationThresholds: Configurable validation criteria
    - ValidationResult: Individual validation check result

Example Usage:
    >>> from src.validation import ThermalValidator, get_material
    >>> validator = ThermalValidator(material=get_material("PICA-X"))
    >>> results = validator.validate_complete_study("data/output/csv/results.csv")
    >>> print(validator.generate_report(results))
"""

from .thermal_validator import ThermalValidator
from .validation_config import (
    MaterialProperties,
    ValidationThresholds,
    ValidationResult,
    ReferenceData,
    MATERIALS_DATABASE,
    REFERENCE_MISSIONS,
    get_material,
    get_default_material,
    list_available_materials,
)

__all__ = [
    "ThermalValidator",
    "MaterialProperties",
    "ValidationThresholds",
    "ValidationResult",
    "ReferenceData",
    "MATERIALS_DATABASE",
    "REFERENCE_MISSIONS",
    "get_material",
    "get_default_material",
    "list_available_materials",
]

__version__ = "1.0.0"
__author__ = "ElementFiniFusee Thermal Analysis Team"
