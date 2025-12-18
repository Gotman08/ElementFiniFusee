"""
Configuration module for thermal validation system.

Contains material properties database, validation thresholds,
and physical constants for aerospace thermal protection systems.
"""

from dataclasses import dataclass
from typing import Optional
from src.physics.constants import STEFAN_BOLTZMANN


# Physical constants
GRAVITY = 9.81  # m/s²


@dataclass
class MaterialProperties:
    """
    Thermal protection system material properties.

    Attributes:
        name: Material designation
        T_melting: Melting point [K]
        T_max_service: Maximum service temperature [K]
        T_ablation_start: Temperature at which ablation begins [K]
        emissivity: Surface emissivity for radiation (0-1)
        density: Material density [kg/m³]
        specific_heat: Specific heat capacity [J/(kg·K)]
        thermal_conductivity: Thermal conductivity [W/(m·K)]
    """
    name: str
    T_melting: float
    T_max_service: float
    T_ablation_start: float
    emissivity: float
    density: float
    specific_heat: float
    thermal_conductivity: float


@dataclass
class ValidationThresholds:
    """
    Configurable thresholds for validation checks.

    Attributes:
        T_max_realistic: Maximum realistic temperature with best TPS [K]
        T_warning: Temperature threshold for warnings [K]
        T_gas_dissociation: Temperature where gas dissociation begins [K]
        q_flux_max: Maximum practical heat flux [W/m²]
        q_flux_warning: Heat flux warning threshold [W/m²]
        T_rad_threshold: Temperature where radiation becomes significant [K]
        rad_conv_ratio_warning: q_rad/q_conv ratio for warning
        rad_conv_ratio_error: q_rad/q_conv ratio for error
        Mach_hypersonic: Mach number for hypersonic regime
        Re_turbulent: Reynolds number for turbulent transition
        Re_min: Minimum Reynolds for correlation validity
        Re_max: Maximum Reynolds for correlation validity
        efficiency_warning: T_material/T_recovery warning threshold
        efficiency_critical: T_material/T_recovery critical threshold
    """
    T_max_realistic: float = 3000.0  # K
    T_warning: float = 2000.0  # K
    T_gas_dissociation: float = 6000.0  # K
    q_flux_max: float = 1.0e7  # W/m² (10 MW/m²)
    q_flux_warning: float = 5.0e6  # W/m² (5 MW/m²)
    T_rad_threshold: float = 1000.0  # K
    rad_conv_ratio_warning: float = 0.1
    rad_conv_ratio_error: float = 0.5
    Mach_hypersonic: float = 5.0
    Re_turbulent: float = 5.0e5
    Re_min: float = 1.0e4
    Re_max: float = 1.0e8
    efficiency_warning: float = 0.8
    efficiency_critical: float = 0.95


@dataclass
class ValidationResult:
    """
    Result of a single validation check.

    Attributes:
        check_name: Name of the validation check
        severity: 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'
        message: Human-readable description
        value: Actual measured value (optional)
        threshold: Threshold value that was exceeded (optional)
        recommendation: Suggested fix or action (optional)
        velocity: Velocity at which issue occurs [m/s] (optional)
    """
    check_name: str
    severity: str
    message: str
    value: Optional[float] = None
    threshold: Optional[float] = None
    recommendation: Optional[str] = None
    velocity: Optional[float] = None


@dataclass
class ReferenceData:
    """
    Historical reentry reference data for comparison.

    Attributes:
        mission: Mission name
        velocity: Reentry velocity [m/s]
        temperature: Stagnation point temperature [K]
        heat_flux_max: Maximum heat flux [W/m²]
        description: Additional context
    """
    mission: str
    velocity: float
    temperature: float
    heat_flux_max: float
    description: str


# Material Database
# Based on aerospace industry standards and published data

MATERIALS_DATABASE = {
    "Aluminum": MaterialProperties(
        name="Aluminum (structural)",
        T_melting=933.0,  # K
        T_max_service=700.0,  # K (conservative for structural integrity)
        T_ablation_start=float('inf'),  # No ablation
        emissivity=0.10,  # Polished aluminum
        density=2700.0,  # kg/m³
        specific_heat=900.0,  # J/(kg·K)
        thermal_conductivity=237.0  # W/(m·K)
    ),

    "Carbon-Carbon": MaterialProperties(
        name="Carbon-Carbon (Space Shuttle nose)",
        T_melting=3800.0,  # K (sublimation)
        T_max_service=3000.0,  # K
        T_ablation_start=3200.0,  # K
        emissivity=0.85,  # High emissivity
        density=1900.0,  # kg/m³
        specific_heat=710.0,  # J/(kg·K)
        thermal_conductivity=100.0  # W/(m·K) (highly anisotropic)
    ),

    "PICA-X": MaterialProperties(
        name="PICA-X (SpaceX Dragon)",
        T_melting=float('inf'),  # Ablates, doesn't melt
        T_max_service=2600.0,  # K
        T_ablation_start=2000.0,  # K (begins pyrolysis)
        emissivity=0.80,
        density=240.0,  # kg/m³ (very low density)
        specific_heat=1200.0,  # J/(kg·K)
        thermal_conductivity=0.15  # W/(m·K) (excellent insulator)
    ),

    "RCC": MaterialProperties(
        name="Reinforced Carbon-Carbon (Shuttle leading edges)",
        T_melting=3800.0,  # K
        T_max_service=3300.0,  # K
        T_ablation_start=3500.0,  # K
        emissivity=0.87,
        density=2000.0,  # kg/m³
        specific_heat=710.0,  # J/(kg·K)
        thermal_conductivity=95.0  # W/(m·K)
    ),

    "Phenolic": MaterialProperties(
        name="Phenolic (Apollo)",
        T_melting=float('inf'),  # Ablative
        T_max_service=2000.0,  # K
        T_ablation_start=1800.0,  # K
        emissivity=0.75,
        density=1300.0,  # kg/m³
        specific_heat=1100.0,  # J/(kg·K)
        thermal_conductivity=0.20  # W/(m·K)
    ),

    "Stainless-Steel": MaterialProperties(
        name="Stainless Steel 304",
        T_melting=1673.0,  # K
        T_max_service=1200.0,  # K
        T_ablation_start=float('inf'),
        emissivity=0.60,  # Oxidized
        density=8000.0,  # kg/m³
        specific_heat=500.0,  # J/(kg·K)
        thermal_conductivity=16.0  # W/(m·K)
    ),
}


# Historical Reference Data
REFERENCE_MISSIONS = [
    ReferenceData(
        mission="Space Shuttle (STS)",
        velocity=7800.0,  # m/s
        temperature=1923.0,  # K (1650°C)
        heat_flux_max=1.5e6,  # W/m² (1.5 MW/m²)
        description="Nose cone and wing leading edges"
    ),

    ReferenceData(
        mission="Apollo (Lunar return)",
        velocity=11000.0,  # m/s
        temperature=3033.0,  # K (2760°C)
        heat_flux_max=6.0e6,  # W/m² (6 MW/m²)
        description="Command module heat shield"
    ),

    ReferenceData(
        mission="SpaceX Dragon 2",
        velocity=7500.0,  # m/s
        temperature=1800.0,  # K (estimated)
        heat_flux_max=1.2e6,  # W/m²
        description="PICA-X heat shield"
    ),

    ReferenceData(
        mission="Orion (LEO return)",
        velocity=8000.0,  # m/s
        temperature=2200.0,  # K (estimated)
        heat_flux_max=2.0e6,  # W/m²
        description="Avcoat heat shield"
    ),
]


def get_material(material_name: str) -> MaterialProperties:
    """
    Retrieve material properties from database.

    Args:
        material_name: Name of the material (case-insensitive)

    Returns:
        MaterialProperties object

    Raises:
        ValueError: If material not found in database
    """
    # Case-insensitive lookup
    for key, mat in MATERIALS_DATABASE.items():
        if key.lower() == material_name.lower():
            return mat

    # If not found, list available materials
    available = ", ".join(MATERIALS_DATABASE.keys())
    raise ValueError(
        f"Material '{material_name}' not found in database. "
        f"Available materials: {available}"
    )


def get_default_material() -> MaterialProperties:
    """Return default material (Carbon-Carbon TPS)."""
    return MATERIALS_DATABASE["Carbon-Carbon"]


def list_available_materials() -> list:
    """Return list of all available material names."""
    return list(MATERIALS_DATABASE.keys())
