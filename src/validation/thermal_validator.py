"""
Thermal validation engine for rocket reentry FEM analysis.

Validates physical correctness of thermal analysis results,
identifies missing physics, and provides engineering recommendations.
"""

import csv
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

from .validation_config import (
    MaterialProperties,
    ValidationThresholds,
    ValidationResult,
    ReferenceData,
    STEFAN_BOLTZMANN,
    REFERENCE_MISSIONS,
    get_default_material,
)

# Import physics calculations from existing code
from src.physics.aerothermal import (
    compute_reynolds_number,
    compute_recovery_temperature,
    compute_heat_transfer_coefficient,
    compute_stagnation_temperature,
)
from src.physics.constants import (
    RHO_inf,
    T_inf,
    MU_inf,
    k_air,
    c_p,
    gamma,
    r_recovery,
    L_ref,
    Pr,
)


class ThermalValidator:
    """
    Comprehensive validation engine for thermal analysis results.

    Validates physical correctness, identifies missing physics effects,
    and provides actionable engineering recommendations.
    """

    def __init__(
        self,
        thresholds: Optional[ValidationThresholds] = None,
        material: Optional[MaterialProperties] = None,
    ):
        """
        Initialize thermal validator.

        Args:
            thresholds: Validation thresholds (uses defaults if None)
            material: Material properties (uses Carbon-Carbon if None)
        """
        self.thresholds = thresholds if thresholds else ValidationThresholds()
        self.material = material if material else get_default_material()
        self.results: List[ValidationResult] = []

    def load_csv_results(self, csv_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load parametric study results from CSV file.

        Args:
            csv_file: Path to CSV file with columns (Velocity_m_s, T_max_K)

        Returns:
            Tuple of (velocities, temperatures) as numpy arrays

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If CSV format is invalid
        """
        csv_path = Path(csv_file)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

        velocities = []
        temperatures = []

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)

            # Check for required columns
            if 'Velocity_m_s' not in reader.fieldnames or 'T_max_K' not in reader.fieldnames:
                raise ValueError(
                    f"CSV must contain 'Velocity_m_s' and 'T_max_K' columns. "
                    f"Found: {reader.fieldnames}"
                )

            for row in reader:
                try:
                    v = float(row['Velocity_m_s'])
                    t = float(row['T_max_K'])
                    velocities.append(v)
                    temperatures.append(t)
                except (ValueError, KeyError) as e:
                    raise ValueError(f"Invalid CSV data: {e}")

        if len(velocities) == 0:
            raise ValueError("CSV file contains no data")

        return np.array(velocities), np.array(temperatures)

    def validate_complete_study(
        self, csv_file: str
    ) -> List[ValidationResult]:
        """
        Run all validation checks on a parametric study.

        Args:
            csv_file: Path to parametric study CSV

        Returns:
            List of ValidationResult objects
        """
        self.results = []

        # Load data
        velocities, temperatures = self.load_csv_results(csv_file)

        # Run validation checks for each case
        for V, T_max in zip(velocities, temperatures):
            self.results.extend(self.check_temperature_bounds(T_max, V))
            self.results.extend(self.check_recovery_temperature(V, T_max))
            self.results.extend(self.check_radiative_cooling(T_max, V))
            self.results.extend(self.check_heat_flux(V))
            self.results.extend(self.check_ablation_requirement(T_max, V))
            self.results.extend(self.check_reynolds_mach(V))

        # Global checks
        self.results.extend(self.compare_to_reference_data(velocities, temperatures))

        return self.results

    def check_temperature_bounds(
        self, T_max: float, V: float
    ) -> List[ValidationResult]:
        """
        Check if temperatures exceed material service limits.

        Args:
            T_max: Maximum temperature [K]
            V: Velocity [m/s]

        Returns:
            List of validation results
        """
        results = []

        # Check against material service temperature
        if T_max > self.material.T_max_service:
            ratio = T_max / self.material.T_max_service
            results.append(ValidationResult(
                check_name="Material Temperature Limit",
                severity="CRITICAL",
                message=f"Temperature exceeds {self.material.name} service limit",
                value=T_max,
                threshold=self.material.T_max_service,
                recommendation=f"T_max = {T_max:.1f} K > T_service = {self.material.T_max_service:.1f} K "
                              f"(ratio: {ratio:.2f}x). Material structural failure imminent. "
                              f"Consider advanced TPS or active cooling.",
                velocity=V
            ))

        # Check against melting point
        if T_max > self.material.T_melting and self.material.T_melting < 1e6:
            results.append(ValidationResult(
                check_name="Melting Point",
                severity="CRITICAL",
                message=f"Temperature exceeds melting point of {self.material.name}",
                value=T_max,
                threshold=self.material.T_melting,
                recommendation=f"Material will melt at {self.material.T_melting:.1f} K. "
                              f"Current T = {T_max:.1f} K. Use refractory materials or ablative TPS.",
                velocity=V
            ))

        # Warning threshold
        elif T_max > self.thresholds.T_warning:
            results.append(ValidationResult(
                check_name="Temperature Warning",
                severity="WARNING",
                message=f"Temperature {T_max:.1f} K exceeds warning threshold",
                value=T_max,
                threshold=self.thresholds.T_warning,
                recommendation="High temperature requires advanced thermal protection system.",
                velocity=V
            ))

        return results

    def check_recovery_temperature(
        self, V: float, T_max: float
    ) -> List[ValidationResult]:
        """
        Compare material temperature to theoretical recovery temperature.

        Args:
            V: Velocity [m/s]
            T_max: Maximum material temperature [K]

        Returns:
            List of validation results
        """
        results = []

        # Calculate recovery temperature
        T_recovery = self._compute_recovery_temp(V)

        # Calculate thermal efficiency (how much of recovery temp is reached)
        efficiency = T_max / T_recovery if T_recovery > 0 else 0

        # Check if efficiency is unrealistically high
        if efficiency > self.thresholds.efficiency_critical:
            results.append(ValidationResult(
                check_name="Recovery Temperature Efficiency",
                severity="CRITICAL",
                message=f"Material temperature too close to recovery temperature",
                value=efficiency,
                threshold=self.thresholds.efficiency_critical,
                recommendation=f"Efficiency = {efficiency:.2f} (T_mat={T_max:.1f}K / T_rec={T_recovery:.1f}K). "
                              f"Physical model missing major cooling mechanisms. "
                              f"Expected efficiency: 0.2-0.6 for passive TPS. "
                              f"Add: (1) Radiative cooling, (2) Conduction into structure, (3) Ablation if T>2000K.",
                velocity=V
            ))
        elif efficiency > self.thresholds.efficiency_warning:
            results.append(ValidationResult(
                check_name="Recovery Temperature Efficiency",
                severity="WARNING",
                message=f"Material temperature approaching recovery temperature",
                value=efficiency,
                threshold=self.thresholds.efficiency_warning,
                recommendation=f"Efficiency = {efficiency:.2f}. Consider adding heat dissipation mechanisms.",
                velocity=V
            ))

        # Check if recovery temperature itself is in gas dissociation regime
        if T_recovery > self.thresholds.T_gas_dissociation:
            results.append(ValidationResult(
                check_name="Gas Dissociation",
                severity="ERROR",
                message=f"Recovery temperature in dissociation regime",
                value=T_recovery,
                threshold=self.thresholds.T_gas_dissociation,
                recommendation=f"T_recovery = {T_recovery:.1f} K > {self.thresholds.T_gas_dissociation:.1f} K. "
                              f"Ideal gas assumption invalid. Molecular dissociation (O2->2O, N2->2N) absorbs energy. "
                              f"Current model overestimates temperature. Consider real gas effects and chemistry.",
                velocity=V
            ))

        return results

    def check_radiative_cooling(
        self, T_max: float, V: float
    ) -> List[ValidationResult]:
        """
        Determine if radiative cooling is significant.

        Args:
            T_max: Maximum temperature [K]
            V: Velocity [m/s]

        Returns:
            List of validation results
        """
        results = []

        if T_max < self.thresholds.T_rad_threshold:
            return results  # Radiation negligible below threshold

        # Calculate radiative heat flux
        epsilon = self.material.emissivity
        q_rad = epsilon * STEFAN_BOLTZMANN * (T_max**4 - T_inf**4)

        # Calculate convective heat transfer coefficient and flux
        alpha = self._compute_heat_transfer_coeff(V)
        T_recovery = self._compute_recovery_temp(V)
        q_conv = alpha * abs(T_max - T_recovery)

        # Ratio of radiation to convection
        # Use numerical threshold to avoid division by near-zero values
        if q_conv > 1e-6:  # 1 µW/m² threshold
            ratio = q_rad / q_conv
        else:
            ratio = float('inf')
            logger.warning(
                f"q_conv ≈ 0 détecté (q_conv = {q_conv:.2e} W/m²). "
                f"Cas physiquement improbable: T_max ≈ T_recovery. "
                f"Ratio radiation/convection = inf."
            )

        # Check significance
        if ratio > self.thresholds.rad_conv_ratio_error:
            results.append(ValidationResult(
                check_name="Radiative Cooling Missing",
                severity="ERROR",
                message=f"Radiative cooling is dominant but not modeled",
                value=ratio,
                threshold=self.thresholds.rad_conv_ratio_error,
                recommendation=f"q_rad/q_conv = {ratio:.2f} at V={V:.0f} m/s. "
                              f"Radiation flux: {q_rad/1e6:.2f} MW/m². "
                              f"Model severely overestimates temperature. "
                              f"FIX: Add radiation term to Robin BC: "
                              f"-k*dT/dn = alpha*(T-T_E) - epsilon*sigma*T^4 "
                              f"(modify src/core/assembly.py line ~117)",
                velocity=V
            ))
        elif ratio > self.thresholds.rad_conv_ratio_warning:
            results.append(ValidationResult(
                check_name="Radiative Cooling Significant",
                severity="WARNING",
                message=f"Radiative cooling contributes significantly",
                value=ratio,
                threshold=self.thresholds.rad_conv_ratio_warning,
                recommendation=f"q_rad/q_conv = {ratio:.2f}. Radiation flux: {q_rad/1e6:.2f} MW/m². "
                              f"Consider adding Stefan-Boltzmann radiation term.",
                velocity=V
            ))

        return results

    def check_heat_flux(self, V: float) -> List[ValidationResult]:
        """
        Validate stagnation point heat flux using Sutton-Graves correlation.

        Args:
            V: Velocity [m/s]

        Returns:
            List of validation results
        """
        results = []

        # Sutton-Graves correlation: q = C * sqrt(rho/R_nose) * V^3
        # For R_nose = 0.5 m, C ≈ 1.83e-4 SI units
        R_nose = L_ref  # Use reference length as nose radius
        C_sutton_graves = 1.83e-4

        q_stag = C_sutton_graves * np.sqrt(RHO_inf / R_nose) * V**3

        if q_stag > self.thresholds.q_flux_max:
            results.append(ValidationResult(
                check_name="Heat Flux Limit",
                severity="ERROR",
                message=f"Stagnation heat flux exceeds practical TPS limits",
                value=q_stag,
                threshold=self.thresholds.q_flux_max,
                recommendation=f"q_stag = {q_stag/1e6:.2f} MW/m² at V={V:.0f} m/s. "
                              f"Exceeds practical limit of {self.thresholds.q_flux_max/1e6:.1f} MW/m². "
                              f"Apollo peak: 6 MW/m². Design requires extraordinary TPS beyond current capabilities.",
                velocity=V
            ))
        elif q_stag > self.thresholds.q_flux_warning:
            results.append(ValidationResult(
                check_name="Heat Flux Warning",
                severity="WARNING",
                message=f"Heat flux approaching design limits",
                value=q_stag,
                threshold=self.thresholds.q_flux_warning,
                recommendation=f"q_stag = {q_stag/1e6:.2f} MW/m². "
                              f"Space Shuttle peak: 1.5 MW/m². Advanced TPS required.",
                velocity=V
            ))

        return results

    def check_ablation_requirement(
        self, T_max: float, V: float
    ) -> List[ValidationResult]:
        """
        Determine if ablative TPS is required.

        Args:
            T_max: Maximum temperature [K]
            V: Velocity [m/s]

        Returns:
            List of validation results
        """
        results = []

        if T_max > self.material.T_ablation_start and self.material.T_ablation_start < 1e6:
            results.append(ValidationResult(
                check_name="Ablation Required",
                severity="ERROR",
                message=f"Temperature exceeds ablation threshold",
                value=T_max,
                threshold=self.material.T_ablation_start,
                recommendation=f"T = {T_max:.1f} K > T_ablation = {self.material.T_ablation_start:.1f} K. "
                              f"Ablative TPS required (PICA-X, phenolic, cork). "
                              f"Ablation provides massive cooling through: "
                              f"(1) Pyrolysis endothermic reactions, "
                              f"(2) Mass loss carrying away heat, "
                              f"(3) Transpiration cooling from ablation gases. "
                              f"Current model does not include ablation physics.",
                velocity=V
            ))

        return results

    def check_reynolds_mach(self, V: float) -> List[ValidationResult]:
        """
        Validate flow regime and correlation applicability.

        Args:
            V: Velocity [m/s]

        Returns:
            List of validation results
        """
        results = []

        # Calculate Reynolds number
        Re = self._compute_reynolds(V)

        if Re < self.thresholds.Re_min or Re > self.thresholds.Re_max:
            results.append(ValidationResult(
                check_name="Reynolds Number Range",
                severity="WARNING",
                message=f"Reynolds number outside correlation validity range",
                value=Re,
                threshold=self.thresholds.Re_min if Re < self.thresholds.Re_min else self.thresholds.Re_max,
                recommendation=f"Re = {Re:.2e} outside valid range [{self.thresholds.Re_min:.0e}, {self.thresholds.Re_max:.0e}]. "
                              f"Heat transfer correlations may be inaccurate.",
                velocity=V
            ))

        # Calculate Mach number
        M = self._compute_mach(V)

        if M > self.thresholds.Mach_hypersonic:
            results.append(ValidationResult(
                check_name="Hypersonic Regime",
                severity="WARNING",
                message=f"Flow in hypersonic regime (M > {self.thresholds.Mach_hypersonic})",
                value=M,
                threshold=self.thresholds.Mach_hypersonic,
                recommendation=f"Mach = {M:.1f} (hypersonic). At M>5, significant effects: "
                              f"(1) Chemical dissociation (O2->2O, N2->2N), "
                              f"(2) Variable c_p and gamma with temperature, "
                              f"(3) Real gas effects, "
                              f"(4) Viscous dissipation in boundary layer. "
                              f"Current ideal gas model may overestimate heating.",
                velocity=V
            ))

        return results

    def compare_to_reference_data(
        self, velocities: np.ndarray, temperatures: np.ndarray
    ) -> List[ValidationResult]:
        """
        Compare results to historical reentry mission data.

        Args:
            velocities: Array of velocities [m/s]
            temperatures: Array of temperatures [K]

        Returns:
            List of validation results
        """
        results = []

        for ref in REFERENCE_MISSIONS:
            # Find closest velocity in dataset
            idx = np.argmin(np.abs(velocities - ref.velocity))
            V_closest = velocities[idx]
            T_predicted = temperatures[idx]

            # Calculate deviation
            if abs(V_closest - ref.velocity) / ref.velocity < 0.2:  # Within 20% velocity match
                deviation = abs(T_predicted - ref.temperature) / ref.temperature

                if deviation > 0.5:  # More than 50% deviation
                    results.append(ValidationResult(
                        check_name="Reference Data Comparison",
                        severity="WARNING",
                        message=f"Significant deviation from {ref.mission} data",
                        value=T_predicted,
                        threshold=ref.temperature,
                        recommendation=f"At V≈{V_closest:.0f} m/s, predicted T={T_predicted:.0f}K "
                                      f"vs {ref.mission} T={ref.temperature:.0f}K "
                                      f"(deviation: {deviation*100:.0f}%). "
                                      f"{ref.description}. "
                                      f"Large deviation suggests missing physics (likely radiation cooling).",
                        velocity=V_closest
                    ))

        return results

    def generate_report(self, results: Optional[List[ValidationResult]] = None) -> str:
        """
        Generate human-readable validation report.

        Args:
            results: List of ValidationResult objects (uses self.results if None)

        Returns:
            Formatted report string
        """
        if results is None:
            results = self.results

        # Count by severity
        counts = {"INFO": 0, "WARNING": 0, "ERROR": 0, "CRITICAL": 0}
        for r in results:
            if r.severity in counts:
                counts[r.severity] += 1

        # Build report
        lines = []
        lines.append("=" * 80)
        lines.append("THERMAL ANALYSIS VALIDATION REPORT")
        lines.append("=" * 80)
        lines.append(f"Material: {self.material.name}")
        lines.append(f"T_max_service: {self.material.T_max_service:.1f} K")
        lines.append("")

        lines.append("SUMMARY")
        lines.append("-" * 80)
        lines.append(f"Total checks: {len(results)}")
        total_passed = len(results) - sum(counts.values())
        lines.append(f"  PASSED:   {total_passed}")
        lines.append(f"  WARNING:  {counts['WARNING']}")
        lines.append(f"  ERROR:    {counts['ERROR']}")
        lines.append(f"  CRITICAL: {counts['CRITICAL']}")
        lines.append("")

        # Group by severity
        for severity in ["CRITICAL", "ERROR", "WARNING", "INFO"]:
            severity_results = [r for r in results if r.severity == severity]
            if not severity_results:
                continue

            lines.append(f"{severity} ISSUES")
            lines.append("-" * 80)

            # Group by check name to avoid repetition
            by_check = {}
            for r in severity_results:
                if r.check_name not in by_check:
                    by_check[r.check_name] = []
                by_check[r.check_name].append(r)

            for check_name, check_results in by_check.items():
                # Show first occurrence in detail
                r = check_results[0]
                lines.append(f"[{severity}] {r.message}")
                if r.velocity is not None:
                    lines.append(f"  Velocity: {r.velocity:.1f} m/s")
                if r.value is not None and r.threshold is not None:
                    lines.append(f"  Value: {r.value:.2f}  |  Threshold: {r.threshold:.2f}")
                if r.recommendation:
                    # Wrap long recommendations
                    rec_lines = r.recommendation.split('. ')
                    for rec_line in rec_lines:
                        if rec_line:
                            lines.append(f"  -> {rec_line}")

                # If multiple occurrences, summarize
                if len(check_results) > 1:
                    lines.append(f"  (Occurs in {len(check_results)} cases)")
                lines.append("")

        lines.append("=" * 80)
        lines.append("END REPORT")
        lines.append("=" * 80)

        return "\n".join(lines)

    def export_validation_csv(
        self,
        velocities: np.ndarray,
        temperatures: np.ndarray,
        results: List[ValidationResult],
        output_file: str
    ):
        """
        Export results with validation flags to CSV.

        Args:
            velocities: Array of velocities [m/s]
            temperatures: Array of temperatures [K]
            results: List of validation results
            output_file: Output CSV file path
        """
        # Group results by velocity
        results_by_velocity = {}
        for r in results:
            if r.velocity is not None:
                v_key = f"{r.velocity:.1f}"
                if v_key not in results_by_velocity:
                    results_by_velocity[v_key] = []
                results_by_velocity[v_key].append(r)

        # Write CSV
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Velocity_m_s',
                'T_max_K',
                'T_recovery_K',
                'Efficiency',
                'Severity_Max',
                'Issue_Count',
                'Issues'
            ])

            for V, T_max in zip(velocities, temperatures):
                T_rec = self._compute_recovery_temp(V)
                efficiency = T_max / T_rec if T_rec > 0 else 0

                v_key = f"{V:.1f}"
                case_results = results_by_velocity.get(v_key, [])

                # Find max severity
                severity_order = {"INFO": 0, "WARNING": 1, "ERROR": 2, "CRITICAL": 3}
                max_severity = "PASSED"
                if case_results:
                    max_severity = max(case_results, key=lambda r: severity_order.get(r.severity, 0)).severity

                # Concatenate issue messages
                issues = " | ".join([r.check_name for r in case_results])

                writer.writerow([
                    f"{V:.1f}",
                    f"{T_max:.1f}",
                    f"{T_rec:.1f}",
                    f"{efficiency:.3f}",
                    max_severity,
                    len(case_results),
                    issues
                ])

    # Helper methods for physics calculations
    def _compute_recovery_temp(self, V: float) -> float:
        """Calculate recovery temperature [K]."""
        try:
            return compute_recovery_temperature(V, T_inf, r_recovery, gamma, c_p)
        except NameError:
            # Fallback implementation
            return T_inf + r_recovery * V**2 / (2 * c_p)

    def _compute_reynolds(self, V: float) -> float:
        """Calculate Reynolds number."""
        try:
            return compute_reynolds_number(V, RHO_inf, MU_inf, L_ref)
        except NameError:
            # Fallback implementation
            return (RHO_inf * V * L_ref) / MU_inf

    def _compute_mach(self, V: float) -> float:
        """Calculate Mach number."""
        # Speed of sound: a = sqrt(gamma * R * T)
        R_specific = c_p * (gamma - 1) / gamma
        a_inf = np.sqrt(gamma * R_specific * T_inf)
        return V / a_inf

    def _compute_heat_transfer_coeff(self, V: float) -> float:
        """Calculate heat transfer coefficient [W/(m²·K)]."""
        try:
            return compute_heat_transfer_coefficient(V, RHO_inf, MU_inf, k_air, L_ref, Pr)
        except NameError:
            # Fallback: simple correlation
            Re = self._compute_reynolds(V)
            if Re > 5e5:
                Nu = 0.037 * Re**0.8 * Pr**(1.0/3.0)
            else:
                Nu = 0.664 * Re**0.5 * Pr**(1.0/3.0)
            return Nu * k_air / L_ref
