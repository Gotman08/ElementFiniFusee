"""
Tests de non-régression pour l'analyse thermique FEM.

Vérifie que les corrections de bugs et l'intégration du refroidissement
radiatif produisent des résultats physiquement corrects.
"""
import pytest
import numpy as np
import os
from src.mesh.mesh_reader import read_gmsh_mesh, create_node_mapping
from src.physics.parametric_study import parametric_velocity_study


class TestThermalRegression:
    """
    Tests de régression pour garantir la correction des bugs
    et la validation physique des résultats.
    """

    @pytest.fixture
    def mesh_file(self):
        """Fixture pour le fichier mesh."""
        return "data/meshes/rocket_mesh.msh"

    def test_radiation_reduces_temperature_high_velocity(self, mesh_file):
        """
        Test: Le refroidissement radiatif doit réduire significativement
        la température à haute vitesse (V >= 5000 m/s).

        Régression fixée: Avant correction, pas de radiation → T_max irréaliste.
        """
        if not os.path.exists(mesh_file):
            pytest.skip(f"Mesh file not found: {mesh_file}")

        V_test = np.array([5000])  # m/s

        # Sans radiation (linéaire)
        _, T_max_no_rad, _ = parametric_velocity_study(
            mesh_file=mesh_file,
            velocity_range=V_test,
            mode="tail-first",
            include_radiation=False
        )

        # Avec radiation (non-linéaire)
        _, T_max_with_rad, _ = parametric_velocity_study(
            mesh_file=mesh_file,
            velocity_range=V_test,
            mode="tail-first",
            include_radiation=True
        )

        # Validation: radiation doit réduire température de > 10%
        reduction_percent = (T_max_no_rad[0] - T_max_with_rad[0]) / T_max_no_rad[0] * 100

        assert reduction_percent > 10.0, (
            f"Radiation devrait réduire T de > 10% @ V=5000 m/s, "
            f"obtenu: {reduction_percent:.1f}%"
        )

    def test_temperature_realistic_hypersonic(self, mesh_file):
        """
        Test: Température doit rester < 3500 K @ V=7000 m/s avec radiation.

        Régression fixée: Avant, T atteignait ~11,300 K (non physique).
        """
        if not os.path.exists(mesh_file):
            pytest.skip(f"Mesh file not found: {mesh_file}")

        V_test = np.array([7000])  # m/s - vitesse hypersonique

        _, T_max_with_rad, _ = parametric_velocity_study(
            mesh_file=mesh_file,
            velocity_range=V_test,
            mode="tail-first",
            include_radiation=True
        )

        T_max = T_max_with_rad[0]

        # Température doit être réaliste pour matériaux haute température
        assert T_max < 3500.0, (
            f"Température trop élevée @ V=7000 m/s: {T_max:.1f} K. "
            f"Radiation inefficace ou bug non corrigé?"
        )

        # Température doit rester au-dessus de l'ambiante
        assert T_max > 2000.0, (
            f"Température trop basse @ V=7000 m/s: {T_max:.1f} K. "
            f"Physique incorrecte?"
        )

    def test_temperature_increases_with_velocity(self, mesh_file):
        """
        Test: Température doit augmenter monotoniquement avec la vitesse.

        Régression fixée: Vérifie la cohérence physique globale.
        """
        if not os.path.exists(mesh_file):
            pytest.skip(f"Mesh file not found: {mesh_file}")

        V_range = np.array([1000, 3000, 5000, 7000])

        _, T_max_list, _ = parametric_velocity_study(
            mesh_file=mesh_file,
            velocity_range=V_range,
            mode="tail-first",
            include_radiation=True
        )

        # Vérifier croissance monotone
        for i in range(len(T_max_list) - 1):
            assert T_max_list[i] < T_max_list[i+1], (
                f"Température non croissante: "
                f"T({V_range[i]} m/s) = {T_max_list[i]:.1f} K >= "
                f"T({V_range[i+1]} m/s) = {T_max_list[i+1]:.1f} K"
            )

    def test_low_velocity_stable_temperature(self, mesh_file):
        """
        Test: À basse vitesse (V < 1000 m/s), température doit rester proche de l'ambiante.

        Régression fixée: Vérifie que le modèle reste stable à basse vitesse.
        """
        if not os.path.exists(mesh_file):
            pytest.skip(f"Mesh file not found: {mesh_file}")

        V_test = np.array([500])  # m/s - subsonique

        _, T_max_with_rad, _ = parametric_velocity_study(
            mesh_file=mesh_file,
            velocity_range=V_test,
            mode="tail-first",
            include_radiation=True
        )

        T_max = T_max_with_rad[0]

        # À basse vitesse, chauffage aérodynamique négligeable
        assert T_max < 600.0, (
            f"Température trop élevée @ V=500 m/s: {T_max:.1f} K. "
            f"Chauffage incorrect à basse vitesse?"
        )

        assert T_max > 250.0, (
            f"Température anormalement basse @ V=500 m/s: {T_max:.1f} K"
        )


@pytest.mark.skipif(
    not os.path.exists("data/meshes/rocket_mesh.msh"),
    reason="Mesh file required for integration tests"
)
class TestBugFixes:
    """
    Tests spécifiques pour vérifier que les bugs critiques sont corrigés.
    """

    def test_sparse_matrix_copy_fix(self):
        """
        Test: Vérifie que la copie de matrice sparse est correctement effectuée.

        Régression fixée: Bug dans animation.py:95 où A_bc, F_bc = A, F
        créait des alias au lieu de copies.
        """
        # Ce test est indirect - vérifie que les simulations multiples donnent
        # des résultats cohérents (pas de mutation de matrice)
        from src.visualization.animation import compute_all_frames
        from src.mesh.mesh_reader import read_gmsh_mesh, create_node_mapping

        mesh_file = "data/meshes/rocket_mesh.msh"
        mesh = read_gmsh_mesh(mesh_file)
        node_to_dof, _ = create_node_mapping(mesh)

        # Profil simple
        times = np.array([0, 100, 200])
        velocities = np.array([5000, 4000, 3000])
        altitudes = np.array([80000, 70000, 60000])

        # Première exécution
        _, _, sols1, T_max1 = compute_all_frames(
            mesh, node_to_dof,
            (times, velocities),
            altitudes,
            include_radiation=False
        )

        # Deuxième exécution (devrait donner résultats identiques)
        _, _, sols2, T_max2 = compute_all_frames(
            mesh, node_to_dof,
            (times, velocities),
            altitudes,
            include_radiation=False
        )

        # Les résultats doivent être identiques (pas de mutation)
        np.testing.assert_allclose(T_max1, T_max2, rtol=1e-10,
                                   err_msg="Mutation de matrice détectée!")

    def test_solver_return_type_validation(self):
        """
        Test: Vérifie que le solver valide correctement le type retourné.

        Régression fixée: Bug dans solver.py:85 - pas de validation du type
        retourné par spsolve.
        """
        # Ce test est indirect - vérifie que le solver ne plante pas
        # et retourne bien un ndarray
        from src.core.solver import solve_linear_system
        import scipy.sparse as sp

        # Créer un système simple
        A = sp.csr_matrix([[2, 1], [1, 2]], dtype=float)
        F = np.array([1.0, 1.0])

        U = solve_linear_system(A, F, method='direct')

        # Vérifier que c'est bien un ndarray
        assert isinstance(U, np.ndarray), f"Expected ndarray, got {type(U)}"
        assert U.shape == (2,), f"Wrong shape: {U.shape}"
