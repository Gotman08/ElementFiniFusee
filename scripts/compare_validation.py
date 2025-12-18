"""
Script de comparaison des résultats de validation avant/après corrections.

Compare les températures et l'efficacité du refroidissement radiatif
avant et après les corrections des bugs et l'intégration du solveur non-linéaire.
"""
import os
import sys
import numpy as np
from src.mesh.mesh_reader import read_gmsh_mesh, create_node_mapping
from src.physics.parametric_study import parametric_velocity_study


def compare_radiation_effectiveness():
    """
    Compare les résultats avec et sans radiation pour démontrer l'amélioration.
    """
    print("\n" + "="*80)
    print("COMPARAISON AVANT/APRES - VALIDATION DU REFROIDISSEMENT RADIATIF")
    print("="*80)
    print()

    mesh_file = "data/meshes/rocket_mesh.msh"

    if not os.path.exists(mesh_file):
        print(f"[ERREUR] Mesh file not found: {mesh_file}")
        print("Veuillez générer le maillage d'abord.")
        sys.exit(1)

    # Test velocities
    velocities_test = np.array([1000, 3000, 5000, 7000])  # m/s

    print("Configuration:")
    print(f"  - Mesh: {mesh_file}")
    print(f"  - Vitesses de test: {velocities_test} m/s")
    print(f"  - Mode: tail-first (rentrée rétropropulsive)")
    print()

    # ========================================================================
    # SANS RADIATION (Modèle linéaire - AVANT corrections conceptuelles)
    # ========================================================================
    print("-" * 80)
    print("1. SIMULATION SANS RADIATION (modèle linéaire)")
    print("-" * 80)

    V_no_rad, T_max_no_rad, _ = parametric_velocity_study(
        mesh_file=mesh_file,
        velocity_range=velocities_test,
        mode="tail-first",
        include_radiation=False
    )

    print()

    # ========================================================================
    # AVEC RADIATION (Modèle non-linéaire - APRES corrections)
    # ========================================================================
    print("-" * 80)
    print("2. SIMULATION AVEC RADIATION (modèle non-linéaire)")
    print("-" * 80)

    V_with_rad, T_max_with_rad, _ = parametric_velocity_study(
        mesh_file=mesh_file,
        velocity_range=velocities_test,
        mode="tail-first",
        include_radiation=True
    )

    print()

    # ========================================================================
    # COMPARAISON ET VALIDATION
    # ========================================================================
    print("="*80)
    print("RESULTATS COMPARATIFS")
    print("="*80)
    print()

    print(f"{'Vitesse (m/s)':<15} {'Sans Rad (K)':<15} {'Avec Rad (K)':<15} {'Réduction (%)':<15} {'Statut':<20}")
    print("-" * 85)

    all_passed = True
    for i, V in enumerate(velocities_test):
        T_no = T_max_no_rad[i]
        T_yes = T_max_with_rad[i]
        reduction = (T_no - T_yes) / T_no * 100

        # Validation criteria
        if V >= 3000:
            # High velocity: radiation should reduce temperature by > 10%
            passed = reduction > 10.0
            status = "✅ OK" if passed else "❌ FAIL"
        else:
            # Low velocity: small reduction expected
            passed = True
            status = "✅ OK"

        if not passed:
            all_passed = False

        print(f"{V:<15.0f} {T_no:<15.1f} {T_yes:<15.1f} {reduction:<15.1f} {status:<20}")

    print("-" * 85)
    print()

    # ========================================================================
    # VALIDATION PHYSIQUE
    # ========================================================================
    print("VALIDATION PHYSIQUE:")
    print()

    # Test 1: High velocity temperature should be realistic
    T_max_high_v = T_max_with_rad[-1]  # @ V=7000 m/s
    test1_pass = T_max_high_v < 3500.0
    print(f"  1. Température @ V=7000 m/s: {T_max_high_v:.1f} K")
    print(f"     Attendu: < 3500 K")
    print(f"     Résultat: {'✅ PASS' if test1_pass else '❌ FAIL'}")
    print()

    # Test 2: Radiation should have significant effect at high velocity
    reduction_7000 = (T_max_no_rad[-1] - T_max_with_rad[-1]) / T_max_no_rad[-1] * 100
    test2_pass = reduction_7000 > 15.0
    print(f"  2. Réduction par radiation @ V=7000 m/s: {reduction_7000:.1f}%")
    print(f"     Attendu: > 15%")
    print(f"     Résultat: {'✅ PASS' if test2_pass else '❌ FAIL'}")
    print()

    # Test 3: Temperature should increase with velocity
    test3_pass = all(T_max_with_rad[i] < T_max_with_rad[i+1] for i in range(len(T_max_with_rad)-1))
    print(f"  3. Température croissante avec vitesse:")
    print(f"     Résultat: {'✅ PASS' if test3_pass else '❌ FAIL'}")
    print()

    # ========================================================================
    # RÉSUMÉ FINAL
    # ========================================================================
    print("="*80)
    final_status = all_passed and test1_pass and test2_pass and test3_pass
    if final_status:
        print("✅ VALIDATION RÉUSSIE - Le refroidissement radiatif fonctionne correctement")
    else:
        print("❌ VALIDATION ÉCHOUÉE - Vérifier l'intégration du solveur non-linéaire")
    print("="*80)
    print()

    return final_status


if __name__ == '__main__':
    try:
        success = compare_radiation_effectiveness()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERREUR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
