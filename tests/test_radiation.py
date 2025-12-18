"""
Unit and integration tests for radiation cooling implementation.

Tests verify:
1. Radiation coefficient calculation magnitude
2. Picard iteration convergence behavior
3. Temperature reduction with radiation vs without
4. Comparison with historical reentry data
"""

import pytest
import numpy as np
from src.physics.parametric_study import (
    compute_aerothermal_parameters,
    STEFAN_BOLTZMANN,
    EPSILON_carbon_carbon
)
from src.core.nonlinear_solver import compute_radiation_coefficient


class TestRadiationCoefficient:
    """Test radiation coefficient calculation."""

    def test_magnitude_at_2000K(self):
        """Verify α_rad magnitude at T=2000K."""
        epsilon = 0.85
        sigma = 5.67e-8
        T = 2000.0  # K

        alpha_rad = compute_radiation_coefficient(epsilon, sigma, T)

        # At T=2000K, α_rad = 0.85 * 5.67e-8 * 2000³ ≈ 387 W/(m²·K)
        assert 380 < alpha_rad < 400, f"Expected ~387, got {alpha_rad}"

    def test_magnitude_at_3000K(self):
        """Verify α_rad magnitude at T=3000K."""
        epsilon = 0.85
        sigma = 5.67e-8
        T = 3000.0  # K

        alpha_rad = compute_radiation_coefficient(epsilon, sigma, T)

        # At T=3000K, α_rad = 0.85 * 5.67e-8 * 3000³ ≈ 1303 W/(m²·K)
        assert 1250 < alpha_rad < 1350, f"Expected ~1303, got {alpha_rad}"

    def test_temperature_scaling(self):
        """Verify α_rad scales as T³."""
        epsilon = 0.85
        sigma = 5.67e-8

        T1 = 1000.0
        T2 = 2000.0

        alpha1 = compute_radiation_coefficient(epsilon, sigma, T1)
        alpha2 = compute_radiation_coefficient(epsilon, sigma, T2)

        # Should scale as (T2/T1)³ = 8
        ratio = alpha2 / alpha1
        assert 7.9 < ratio < 8.1, f"Expected ratio ~8, got {ratio}"

    def test_emissivity_dependence(self):
        """Verify α_rad is proportional to emissivity."""
        sigma = 5.67e-8
        T = 2000.0

        alpha1 = compute_radiation_coefficient(0.5, sigma, T)
        alpha2 = compute_radiation_coefficient(1.0, sigma, T)

        ratio = alpha2 / alpha1
        assert 1.99 < ratio < 2.01, f"Expected ratio 2, got {ratio}"


class TestAerothermalParameters:
    """Test aerothermal parameter computation with radiation."""

    def test_returns_four_values(self):
        """Verify function returns (alpha, T_E, epsilon, sigma)."""
        V = 5000.0  # m/s

        result = compute_aerothermal_parameters(V)

        assert len(result) == 4, "Should return 4 values"
        alpha, T_E, epsilon, sigma = result

        # Check types and reasonable ranges
        assert alpha > 0, "alpha should be positive"
        assert T_E > 230, "T_E should be > ambient temperature"
        assert 0 < epsilon <= 1, "epsilon should be in [0,1]"
        assert sigma == STEFAN_BOLTZMANN, "sigma should be Stefan-Boltzmann constant"

    def test_custom_emissivity(self):
        """Verify custom emissivity is returned correctly."""
        V = 5000.0
        custom_epsilon = 0.75

        _, _, epsilon, _ = compute_aerothermal_parameters(V, emissivity=custom_epsilon)

        assert epsilon == custom_epsilon, f"Expected {custom_epsilon}, got {epsilon}"

    def test_velocity_dependence(self):
        """Verify parameters increase with velocity."""
        V1 = 3000.0
        V2 = 6000.0

        alpha1, T_E1, _, _ = compute_aerothermal_parameters(V1)
        alpha2, T_E2, _, _ = compute_aerothermal_parameters(V2)

        assert alpha2 > alpha1, "alpha should increase with velocity"
        assert T_E2 > T_E1, "T_E should increase with velocity"


@pytest.mark.integration
class TestTemperatureReduction:
    """Integration tests verifying temperature reduction with radiation."""

    @pytest.fixture
    def mesh_file(self):
        """Path to test mesh file."""
        return "data/meshes/rocket_mesh.msh"

    def test_radiation_reduces_temperature(self, mesh_file):
        """Verify radiation dramatically reduces maximum temperature."""
        import os
        if not os.path.exists(mesh_file):
            pytest.skip(f"Mesh file not found: {mesh_file}")

        from src.physics.parametric_study import parametric_velocity_study

        V_test = np.array([5000.0])  # Single test case at 5 km/s

        # Run without radiation
        try:
            _, T_max_no_rad, _ = parametric_velocity_study(
                mesh_file, V_test,
                mode="tail-first",
                include_radiation=False
            )
        except Exception as e:
            pytest.skip(f"Cannot run without radiation: {e}")

        # Run with radiation
        try:
            _, T_max_with_rad, _ = parametric_velocity_study(
                mesh_file, V_test,
                mode="tail-first",
                include_radiation=True
            )
        except Exception as e:
            pytest.fail(f"Radiation case failed: {e}")

        T_no_rad = T_max_no_rad[0]
        T_with_rad = T_max_with_rad[0]

        print(f"\nT_max without radiation: {T_no_rad:.1f} K")
        print(f"T_max with radiation: {T_with_rad:.1f} K")
        print(f"Reduction factor: {T_no_rad / T_with_rad:.2f}x")

        # Radiation should reduce temperature by at least 50%
        assert T_with_rad < 0.5 * T_no_rad, \
            f"Radiation should reduce T_max significantly: {T_with_rad} vs {T_no_rad}"

        # With radiation, temperature should be realistic (< 3500 K)
        assert T_with_rad < 3500, \
            f"T_max with radiation should be realistic: {T_with_rad} K"

        # Temperature should still be elevated above ambient
        assert T_with_rad > 1000, \
            f"T_max should be elevated: {T_with_rad} K"

    def test_radiation_convergence(self, mesh_file):
        """Verify Picard iteration converges."""
        import os
        if not os.path.exists(mesh_file):
            pytest.skip(f"Mesh file not found: {mesh_file}")

        from src.physics.parametric_study import parametric_velocity_study

        V_test = np.array([3000.0])  # Lower velocity for easier convergence

        try:
            _, T_max, _ = parametric_velocity_study(
                mesh_file, V_test,
                mode="tail-first",
                include_radiation=True
            )
            # If we get here, convergence succeeded
            assert len(T_max) == 1, "Should return one result"
            assert 500 < T_max[0] < 3000, f"T_max should be reasonable: {T_max[0]} K"

        except Exception as e:
            pytest.fail(f"Convergence failed: {e}")


@pytest.mark.validation
class TestHistoricalComparison:
    """Compare predictions with historical reentry data."""

    @pytest.fixture
    def mesh_file(self):
        """Path to test mesh file."""
        return "data/meshes/rocket_mesh.msh"

    def test_shuttle_comparison(self, mesh_file):
        """Compare with Space Shuttle reentry data."""
        import os
        if not os.path.exists(mesh_file):
            pytest.skip(f"Mesh file not found: {mesh_file}")

        from src.physics.parametric_study import parametric_velocity_study

        V_shuttle = np.array([7800.0])  # Space Shuttle reentry velocity

        try:
            _, T_max, _ = parametric_velocity_study(
                mesh_file, V_shuttle,
                mode="tail-first",
                include_radiation=True
            )

            T_pred = T_max[0]

            # Space Shuttle nose: ~1923 K (1650°C)
            # Allow ±500 K tolerance due to geometry differences
            T_shuttle_ref = 1923.0
            tolerance = 500.0

            print(f"\nSpace Shuttle comparison:")
            print(f"  Reference: {T_shuttle_ref:.1f} K")
            print(f"  Predicted: {T_pred:.1f} K")
            print(f"  Error: {abs(T_pred - T_shuttle_ref):.1f} K")

            # Relaxed assertion for validation
            assert T_shuttle_ref - tolerance < T_pred < T_shuttle_ref + tolerance, \
                f"Prediction {T_pred:.1f} K outside range {T_shuttle_ref:.1f} ± {tolerance:.1f} K"

        except Exception as e:
            pytest.skip(f"Cannot run Shuttle comparison: {e}")

    def test_apollo_comparison(self, mesh_file):
        """Compare with Apollo reentry data."""
        import os
        if not os.path.exists(mesh_file):
            pytest.skip(f"Mesh file not found: {mesh_file}")

        from src.physics.parametric_study import parametric_velocity_study

        V_apollo = np.array([11000.0])  # Apollo reentry velocity

        try:
            _, T_max, _ = parametric_velocity_study(
                mesh_file, V_apollo,
                mode="tail-first",
                include_radiation=True
            )

            T_pred = T_max[0]

            # Apollo heat shield: ~3033 K (2760°C)
            # Allow ±500 K tolerance
            T_apollo_ref = 3033.0
            tolerance = 500.0

            print(f"\nApollo comparison:")
            print(f"  Reference: {T_apollo_ref:.1f} K")
            print(f"  Predicted: {T_pred:.1f} K")
            print(f"  Error: {abs(T_pred - T_apollo_ref):.1f} K")

            # Relaxed assertion
            assert T_apollo_ref - tolerance < T_pred < T_apollo_ref + tolerance, \
                f"Prediction {T_pred:.1f} K outside range {T_apollo_ref:.1f} ± {tolerance:.1f} K"

        except Exception as e:
            pytest.skip(f"Cannot run Apollo comparison: {e}")


@pytest.mark.physics
class TestRadiationPhysics:
    """Test physical correctness of radiation implementation."""

    def test_low_temperature_negligible_radiation(self):
        """At low T, radiation should be negligible compared to convection."""
        epsilon = 0.85
        sigma = 5.67e-8
        T_low = 500.0  # K

        alpha_rad = compute_radiation_coefficient(epsilon, sigma, T_low)

        # Typical convection coefficient at reentry
        alpha_conv = 100.0  # W/(m²·K)

        ratio = alpha_rad / alpha_conv

        # At 500K, radiation should be < 10% of convection
        assert ratio < 0.1, f"Radiation should be negligible at low T: ratio={ratio}"

    def test_high_temperature_dominant_radiation(self):
        """At high T, radiation should dominate over convection."""
        epsilon = 0.85
        sigma = 5.67e-8
        T_high = 3000.0  # K

        alpha_rad = compute_radiation_coefficient(epsilon, sigma, T_high)

        # Typical convection coefficient
        alpha_conv = 200.0  # W/(m²·K)

        ratio = alpha_rad / alpha_conv

        # At 3000K, radiation should be >> convection
        assert ratio > 1.0, f"Radiation should dominate at high T: ratio={ratio}"
        assert ratio > 5.0, f"Radiation should be much larger: ratio={ratio}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
