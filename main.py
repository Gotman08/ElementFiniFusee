"""
SCRIPT PRINCIPAL - ANALYSE THERMIQUE DE FUSEE PAR ELEMENTS FINIS
Etude parametrique de l'influence de la vitesse sur la temperature maximale
"""
import numpy as np
import os
import sys
from mesh_reader import read_gmsh_mesh, create_node_mapping
from parametric_study import parametric_velocity_study
from visualization import (plot_parametric_curve, plot_temperature_field,
                           plot_multiple_temperature_fields, export_results_to_csv)

def main():
    """
    Fonction principale orchestrant l'etude complete
    """
    print("\n")
    print("=" * 80)
    print("     ANALYSE THERMIQUE ELEMENTS FINIS - FUSEE")
    print("           Etude parametrique en vitesse")
    print("=" * 80)
    print()

    # ========================================================================
    # CONFIGURATION
    # ========================================================================

    # Fichier de maillage
    mesh_file = "rocket_mesh.msh"

    # Verifier si le maillage existe
    if not os.path.exists(mesh_file):
        print(f"[WARNING] Le fichier de maillage '{mesh_file}' n'existe pas.")
        print(f"   Vous devez d'abord generer le maillage avec GMSH:")
        print(f"   > gmsh -2 rocket_geometry.geo -o rocket_mesh.msh")
        print()
        print(f"   Alternative: utiliser l'interface GMSH pour ouvrir rocket_geometry.geo")
        print(f"   et generer le maillage 2D.")
        sys.exit(1)

    # Plage de vitesses a etudier (m/s)
    # Vitesses typiques de rentree atmospherique: 1000 - 7000 m/s
    V_min = 1000.0   # m/s
    V_max = 5000.0   # m/s
    n_velocities = 15  # Nombre de points

    velocity_range = np.linspace(V_min, V_max, n_velocities)

    # Temperature a la base (condition de Dirichlet)
    T_base = 300.0  # K (temperature ambiante interne)

    print("PARAMETRES DE L'ETUDE")
    print("-" * 80)
    print(f"  - Fichier de maillage:     {mesh_file}")
    print(f"  - Plage de vitesses:       {V_min:.0f} - {V_max:.0f} m/s")
    print(f"  - Nombre de cas:           {n_velocities}")
    print(f"  - Temperature de base:     {T_base:.1f} K")
    print("-" * 80)
    print()

    # ========================================================================
    # ETUDE PARAMETRIQUE
    # ========================================================================

    print("LANCEMENT DE L'ETUDE PARAMETRIQUE\n")

    velocities, T_max_list, solutions = parametric_velocity_study(
        mesh_file=mesh_file,
        velocity_range=velocity_range,
        base_temperature=T_base
    )

    print()
    print("[OK] ETUDE PARAMETRIQUE TERMINEE")
    print()

    # ========================================================================
    # RESULTATS NUMERIQUES
    # ========================================================================

    print("=" * 80)
    print("RESULTATS NUMERIQUES")
    print("=" * 80)
    print()
    print(f"{'Vitesse (m/s)':<20} {'T_max (K)':<15} {'T_max (C)':<15}")
    print("-" * 50)
    for V, T_max in zip(velocities, T_max_list):
        T_celsius = T_max - 273.15
        print(f"{V:<20.1f} {T_max:<15.2f} {T_celsius:<15.2f}")
    print()

    # Statistiques
    T_min_global = min(T_max_list)
    T_max_global = max(T_max_list)
    delta_T = T_max_global - T_min_global

    print(f"ANALYSE:")
    print(f"   - Temperature maximale atteinte:    {T_max_global:.2f} K ({T_max_global - 273.15:.2f} C)")
    print(f"   - Temperature minimale atteinte:    {T_min_global:.2f} K ({T_min_global - 273.15:.2f} C)")
    print(f"   - Variation totale:                 DeltaT = {delta_T:.2f} K")
    print(f"   - Vitesse critique (T_max):         {velocities[T_max_list.index(T_max_global)]:.0f} m/s")
    print()

    # ========================================================================
    # VISUALISATIONS
    # ========================================================================

    print("=" * 80)
    print("GENERATION DES GRAPHIQUES")
    print("=" * 80)
    print()

    # Creer un dossier pour les resultats
    output_dir = "resultats"
    os.makedirs(output_dir, exist_ok=True)

    # 1. GRAPHIQUE PRINCIPAL: T_max vs V (COURBE D'INGENIEUR)
    print("[1/3] Trace de la courbe T_max(V)...")
    plot_parametric_curve(
        velocities,
        T_max_list,
        title="Temperature Maximale en Fonction de la Vitesse",
        save_file=os.path.join(output_dir, "T_max_vs_velocity.png"),
        show=False
    )

    # 2. CHAMP DE TEMPERATURE (cas le plus critique)
    print("[2/3] Trace du champ de temperature (cas critique)...")
    idx_max = T_max_list.index(T_max_global)
    mesh = read_gmsh_mesh(mesh_file)
    node_to_dof, _ = create_node_mapping(mesh)

    plot_temperature_field(
        mesh,
        node_to_dof,
        solutions[idx_max],
        title=f"Champ de Temperature - V = {velocities[idx_max]:.0f} m/s",
        save_file=os.path.join(output_dir, "temperature_field_critical.png"),
        show=False
    )

    # 3. COMPARAISON DE PLUSIEURS CAS
    print("[3/3] Trace des champs de temperature (comparaison)...")
    plot_multiple_temperature_fields(
        mesh,
        node_to_dof,
        velocities,
        solutions,
        indices=[0, len(velocities) // 2, -1],
        save_file=os.path.join(output_dir, "temperature_fields_comparison.png"),
        show=False
    )

    # 4. EXPORT CSV
    csv_file = os.path.join(output_dir, "results_parametric_study.csv")
    export_results_to_csv(velocities, T_max_list, csv_file)

    print()
    print("[OK] Tous les graphiques ont ete generes et sauvegardes dans:", output_dir)
    print()

    # ========================================================================
    # CONCLUSION
    # ========================================================================

    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()
    print("L'etude parametrique montre que:")
    print(f"  - La temperature maximale croit avec la vitesse (echauffement cinetique)")
    print(f"  - A V = {velocities[-1]:.0f} m/s, T_max = {T_max_list[-1]:.1f} K")
    print(f"  - La zone la plus critique se situe sur l'ogive (convection forcee)")
    print()
    print("Les resultats sont disponibles dans:")
    print(f"  [Graphiques] {output_dir}/")
    print(f"  [Donnees CSV] {csv_file}")
    print()
    print("=" * 80)
    print()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[WARNING] Interruption par l'utilisateur")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n[ERROR] ERREUR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
