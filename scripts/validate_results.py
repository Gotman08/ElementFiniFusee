#!/usr/bin/env python3
"""
Command-line interface for thermal validation system.

Usage:
    python scripts/validate_results.py
    python scripts/validate_results.py --csv path/to/results.csv
    python scripts/validate_results.py --material PICA-X --verbose
    python scripts/validate_results.py --export data/output/validation/flagged_results.csv
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validation.thermal_validator import ThermalValidator
from src.validation.validation_config import (
    ValidationThresholds,
    get_material,
    list_available_materials,
)


def main():
    """Main entry point for validation CLI."""
    parser = argparse.ArgumentParser(
        description="Validate thermal analysis results for rocket reentry simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate default CSV with default material (Carbon-Carbon)
  python scripts/validate_results.py

  # Specify CSV file and material
  python scripts/validate_results.py --csv data/output/csv/my_results.csv --material PICA-X

  # Export results with validation flags
  python scripts/validate_results.py --export data/output/validation/flagged_results.csv

  # List available materials
  python scripts/validate_results.py --list-materials

  # Verbose output with detailed recommendations
  python scripts/validate_results.py --verbose
        """
    )

    parser.add_argument(
        "--csv",
        type=str,
        default="data/output/csv/results_parametric_study.csv",
        help="Path to CSV file with parametric study results (default: data/output/csv/results_parametric_study.csv)"
    )

    parser.add_argument(
        "--material",
        type=str,
        default="Carbon-Carbon",
        help="Material for validation (default: Carbon-Carbon)"
    )

    parser.add_argument(
        "--list-materials",
        action="store_true",
        help="List available materials and exit"
    )

    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export results with validation flags to CSV file"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output with detailed recommendations"
    )

    parser.add_argument(
        "--T-max-realistic",
        type=float,
        default=3000.0,
        help="Maximum realistic temperature threshold [K] (default: 3000)"
    )

    parser.add_argument(
        "--T-warning",
        type=float,
        default=2000.0,
        help="Temperature warning threshold [K] (default: 2000)"
    )

    args = parser.parse_args()

    # List materials and exit
    if args.list_materials:
        print("Available materials:")
        for mat_name in list_available_materials():
            print(f"  - {mat_name}")
        return 0

    # Validate inputs
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {args.csv}")
        print(f"Current working directory: {Path.cwd()}")
        return 1

    # Get material
    try:
        material = get_material(args.material)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Create custom thresholds if provided
    thresholds = ValidationThresholds(
        T_max_realistic=args.T_max_realistic,
        T_warning=args.T_warning
    )

    print(f"Loading results from: {csv_path}")
    print(f"Material: {material.name}")
    print(f"Service temperature limit: {material.T_max_service:.1f} K\n")

    # Create validator
    validator = ThermalValidator(thresholds=thresholds, material=material)

    # Run validation
    try:
        velocities, temperatures = validator.load_csv_results(str(csv_path))
        print(f"Analyzing {len(velocities)} cases (V: {velocities[0]:.0f}-{velocities[-1]:.0f} m/s)...\n")

        results = validator.validate_complete_study(str(csv_path))

    except Exception as e:
        print(f"Error during validation: {e}")
        return 1

    # Generate and print report
    report = validator.generate_report(results)
    print(report)

    # Export if requested
    if args.export:
        export_path = Path(args.export)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        validator.export_validation_csv(velocities, temperatures, results, str(export_path))
        print(f"\nValidation results exported to: {export_path}")

    # Summary for quick assessment
    critical_count = sum(1 for r in results if r.severity == "CRITICAL")
    error_count = sum(1 for r in results if r.severity == "ERROR")

    print("\n" + "="*80)
    if critical_count > 0:
        print(f"CRITICAL: {critical_count} critical issues found. Immediate attention required.")
        return 2
    elif error_count > 0:
        print(f"ERROR: {error_count} errors found. Review recommendations.")
        return 1
    else:
        print("Validation complete. Review warnings if any.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
