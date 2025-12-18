"""
@file run_parametric_study.py
@brief Main script for rocket thermal parametric study using FEM
@author HPC-Code-Documenter
@date 2025

@details
This script orchestrates a complete parametric thermal analysis of a rocket
during atmospheric reentry. It studies the influence of entry velocity on
the maximum surface temperature.

Workflow:
1. Configuration of study parameters
2. Automatic mesh generation (if needed)
3. Parametric velocity sweep with FEM solutions
4. Post-processing and visualization
5. Results export to CSV and PNG formats

Supported reentry modes:
- "nose-first": Classical reentry with nose cone leading
  - Robin BC on external surface (aerodynamic heating)
  - Dirichlet BC on base (fixed temperature)

- "tail-first": Retropropulsive landing (SpaceX-style)
  - Robin BC on base (attack zone receives heating)
  - Neumann BC on flanks and nose (in aerodynamic wake)

Output files:
- data/output/figures/T_max_vs_velocity.png: Key engineering result curve
- data/output/figures/temperature_field_critical.png: Critical case visualization
- data/output/figures/temperature_fields_comparison.png: Multi-velocity comparison
- data/output/csv/results_parametric_study.csv: Numerical results

Usage:
    python scripts/run_parametric_study.py

Configuration can be modified by editing the parameter section in main().
"""
import logging
import numpy as np
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

logging.getLogger('matplotlib').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

from src.mesh.mesh_reader import read_gmsh_mesh, create_node_mapping
from src.physics.parametric_study import parametric_velocity_study
from src.visualization.plotting import (plot_parametric_curve, plot_temperature_field,
                                        plot_multiple_temperature_fields, export_results_to_csv)


def main():
    """
    @brief Main function orchestrating the complete parametric study.

    @details
    Executes the following workflow:

    1. Configuration:
       - Set reentry mode (nose-first or tail-first)
       - Define velocity range for parametric sweep
       - Specify mesh file location

    2. Mesh Preparation:
       - Check for existing mesh file
       - Auto-generate if not found using rocket_mesh module

    3. Parametric Study:
       - Loop over velocity range
       - Compute aerothermal parameters at each velocity
       - Solve FEM thermal problem
       - Extract maximum temperatures

    4. Visualization:
       - Generate $T_{max}(V)$ curve (primary engineering result)
       - Plot critical temperature field
       - Create comparison figure for multiple velocities

    5. Export:
       - Save all figures as PNG (300 DPI)
       - Export numerical results to CSV

    The study produces key engineering insights:
    - Maximum temperature vs velocity relationship
    - Critical velocity identification
    - Thermal gradient visualization
    """
    print("\n")
    print("=" * 80)
    print("     ANALYSE THERMIQUE ELEMENTS FINIS - FUSEE")
    print("           Etude parametrique en vitesse")
    print("=" * 80)
    print()

    MODE = "tail-first"

    mesh_file = "data/meshes/rocket_mesh.msh"

    if not os.path.exists(mesh_file):
        logger.warning(f"Le fichier de maillage '{mesh_file}' n'existe pas.")
        print(f"   Generation du maillage avec le generateur Python:")
        print(f"   > python generate_mesh_python.py {MODE}")
        print()

        try:
            from src.mesh.generators.rocket_mesh import generate_rocket_mesh
            logger.info("Generation automatique du maillage...")
            os.makedirs(os.path.dirname(mesh_file), exist_ok=True)
            generate_rocket_mesh(mode=MODE, output_file=mesh_file)
            print()
        except Exception as e:
            logger.error(f"Impossible de generer le maillage: {e}")
            sys.exit(1)

    V_min = 1000.0
    V_max = 5000.0
    n_velocities = 15

    velocity_range = np.linspace(V_min, V_max, n_velocities)

    T_base = 300.0

    print("PARAMETRES DE L'ETUDE")
    print("-" * 80)
    print(f"  - Mode de rentree:         {MODE.upper()}")
    print(f"  - Fichier de maillage:     {mesh_file}")
    print(f"  - Plage de vitesses:       {V_min:.0f} - {V_max:.0f} m/s")
    print(f"  - Nombre de cas:           {n_velocities}")
    if MODE == "nose-first":
        print(f"  - Temperature de base:     {T_base:.1f} K (Dirichlet)")
    else:
        print(f"  - Condition sur base:      Robin (flux entrant)")
    print("-" * 80)
    print()

    print("LANCEMENT DE L'ETUDE PARAMETRIQUE\n")

    velocities, T_max_list, solutions = parametric_velocity_study(
        mesh_file=mesh_file,
        velocity_range=velocity_range,
        base_temperature=T_base,
        mode=MODE
    )

    print()
    print("[OK] ETUDE PARAMETRIQUE TERMINEE")
    print()

    print("=" * 80)
    print("RESULTATS NUMERIQUES")
    print("=" * 80)
    print()

    if MODE == "tail-first":
        print("Mode TAIL-FIRST: La base (zone d'attaque) est la plus chaude")
        print()

    print(f"{'Vitesse (m/s)':<20} {'T_max (K)':<15} {'T_max (C)':<15}")
    print("-" * 50)
    for V, T_max in zip(velocities, T_max_list):
        T_celsius = T_max - 273.15
        print(f"{V:<20.1f} {T_max:<15.2f} {T_celsius:<15.2f}")
    print()

    T_min_global = min(T_max_list)
    T_max_global = max(T_max_list)
    delta_T = T_max_global - T_min_global

    print(f"ANALYSE:")
    print(f"   - Temperature maximale atteinte:    {T_max_global:.2f} K ({T_max_global - 273.15:.2f} C)")
    print(f"   - Temperature minimale atteinte:    {T_min_global:.2f} K ({T_min_global - 273.15:.2f} C)")
    print(f"   - Variation totale:                 DeltaT = {delta_T:.2f} K")
    print(f"   - Vitesse critique (T_max):         {velocities[T_max_list.index(T_max_global)]:.0f} m/s")
    print()

    print("=" * 80)
    print("GENERATION DES GRAPHIQUES")
    print("=" * 80)
    print()

    output_dir = "data/output/figures"
    os.makedirs(output_dir, exist_ok=True)

    print("[1/3] Trace de la courbe T_max(V)...")

    if MODE == "tail-first":
        title = "Temperature Maximale (Base) vs Vitesse - Mode Tail-First"
    else:
        title = "Temperature Maximale en Fonction de la Vitesse"

    plot_parametric_curve(
        velocities,
        T_max_list,
        title=title,
        save_file=os.path.join(output_dir, "T_max_vs_velocity.png"),
        show=False
    )

    print("[2/3] Trace du champ de temperature (cas critique)...")
    idx_max = T_max_list.index(T_max_global)
    mesh = read_gmsh_mesh(mesh_file)
    node_to_dof, _ = create_node_mapping(mesh)

    if MODE == "tail-first":
        field_title = f"Champ de Temperature (Tail-First) - V = {velocities[idx_max]:.0f} m/s"
    else:
        field_title = f"Champ de Temperature - V = {velocities[idx_max]:.0f} m/s"

    plot_temperature_field(
        mesh,
        node_to_dof,
        solutions[idx_max],
        title=field_title,
        save_file=os.path.join(output_dir, "temperature_field_critical.png"),
        show=False
    )

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

    csv_file = "data/output/csv/results_parametric_study.csv"
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    export_results_to_csv(velocities, T_max_list, csv_file)

    print()
    print("[OK] Tous les graphiques ont ete generes et sauvegardes dans:", output_dir)
    print()

    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()

    if MODE == "tail-first":
        print("L'etude parametrique en mode TAIL-FIRST montre que:")
        print(f"  - La temperature maximale est atteinte sur la BASE (zone d'attaque)")
        print(f"  - L'ogive (dans le sillage) reste plus froide")
        print(f"  - A V = {velocities[-1]:.0f} m/s, T_max = {T_max_list[-1]:.1f} K")
    else:
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
        logger.warning("Interruption par l'utilisateur")
        sys.exit(0)
    except Exception as e:
        logger.error(f"ERREUR: {e}", exc_info=True)
        sys.exit(1)
