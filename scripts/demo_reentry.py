"""
@file demo_reentry.py
@brief Live demonstration script for atmospheric reentry thermal simulation
@author HPC-Code-Documenter
@date 2025

@details
This script provides a complete demonstration of the atmospheric reentry
thermal analysis workflow, including:

1. Configuration of simulation parameters
2. Generation of realistic reentry velocity profiles
3. Loading and validation of the rocket mesh
4. Pre-computation of temperature solutions for all time steps
5. Creation of animated visualization showing temperature evolution

The simulation models a rocket decelerating from orbital velocity (~7000 m/s)
to subsonic speeds (~500 m/s) over approximately 5 minutes, computing the
thermal response at each time step.

Output:
- Interactive animated display of temperature field evolution
- Optional export to GIF or MP4 video format
- Summary statistics of thermal response

Usage:
    python scripts/demo_reentry.py

Configuration can be modified by editing the parameter section in main().
"""
import argparse
import numpy as np
import os
import sys

from src.mesh.mesh_reader import read_gmsh_mesh, create_node_mapping
from src.physics.reentry_profile import generate_coupled_reentry, adaptive_frame_sampling
from src.visualization.animation import compute_all_frames, create_animation


def main(include_radiation=True):
    """
    @brief Main function orchestrating the live reentry demonstration.

    @details
    Executes the complete demonstration workflow:

    1. Parameter Configuration:
       - Reentry profile (initial/final velocity, duration)
       - Animation settings (FPS, export format)
       - Mesh file location

    2. Reentry Profile Generation:
       - Exponential deceleration model
       - Altitude profile from velocity integration

    3. Mesh Loading:
       - GMSH format mesh file
       - Node-to-DOF mapping creation

    4. Solution Pre-computation:
       - Solve thermal problem at each time step
       - Store all temperature fields for animation

    5. Animation Creation:
       - Multi-panel interactive display
       - Optional video export
    """
    print("\n")
    print("=" * 80)
    print("              DEMO LIVE - RENTREE ATMOSPHERIQUE")
    print("           Animation de l'evolution de la temperature")
    print("=" * 80)
    print()

    print("PARAMETRES DE LA SIMULATION")
    print("-" * 80)

    # Paramètres physiques de la rentrée
    h_initial = 250000.0   # 250 km (espace)
    V_initial = 7000.0     # Vitesse orbitale
    gamma_initial = 25.0   # Angle d'entrée (degrés)
    duration = 600.0       # Durée max simulation (10 min)

    fps = 5
    export_video = True
    video_format = 'gif'

    T_base = 300.0

    mesh_file = "data/meshes/rocket_mesh.msh"

    print(f"  - Altitude initiale:    {h_initial/1000:.0f} km")
    print(f"  - Vitesse initiale:     {V_initial:.0f} m/s")
    print(f"  - Angle d'entree:       {gamma_initial:.0f} degres")
    print(f"  - FPS:                  {fps}")
    print(f"  - Temperature base:     {T_base:.0f} K")
    print(f"  - Radiation cooling:    {'ACTIVEE' if include_radiation else 'DESACTIVEE'}")
    if export_video:
        print(f"  - Export video:         OUI (.{video_format})")
    else:
        print(f"  - Export video:         NON")
    print("-" * 80)
    print()

    if not os.path.exists(mesh_file):
        print(f"[ERROR] Fichier de maillage non trouve: {mesh_file}")
        print(f"        Executer d'abord: python generate_mesh_python.py")
        sys.exit(1)

    print("GENERATION DU PROFIL DE RENTREE (PHYSIQUE COUPLEE)")
    print("-" * 80)

    # Utilise le nouveau modèle physique couplé
    times, velocities, altitudes, heat_flux = generate_coupled_reentry(
        h_initial=h_initial,
        V_initial=V_initial,
        gamma_initial=gamma_initial,
        duration=duration,
        dt=3.0,  # Pas de temps 3 secondes
        mass=20000.0,
        Cd=1.0,
        A=10.0
    )

    # Échantillonnage adaptatif pour l'animation (max 150 frames)
    # Beaucoup plus de frames pendant la phase de chauffage critique
    times, velocities, altitudes, heat_flux = adaptive_frame_sampling(
        times, velocities, altitudes, heat_flux, max_frames=150
    )

    print(f"[OK] Profil physique genere:")
    print(f"     - Altitude: {altitudes[0]/1000:.1f} km -> {altitudes[-1]/1000:.1f} km")
    print(f"     - Vitesse:  {velocities[0]:.0f} m/s -> {velocities[-1]:.0f} m/s")
    print(f"     - Duree:    {times[-1]:.0f} s")
    print(f"     - Frames:   {len(times)}")
    print(f"     - Flux thermique max: {np.max(heat_flux)/1e6:.1f} MW/m²")
    print("-" * 80)
    print()

    print("CHARGEMENT DU MAILLAGE")
    print("-" * 80)

    mesh = read_gmsh_mesh(mesh_file)
    node_to_dof, num_dofs = create_node_mapping(mesh)

    print(f"[OK] Maillage charge:")
    print(f"     - Noeuds:    {len(mesh.nodes)}")
    print(f"     - Triangles: {len(mesh.get_triangles())}")
    print(f"     - DOFs:      {num_dofs}")
    print("-" * 80)
    print()

    times, velocities, solutions, T_max_list = compute_all_frames(
        mesh=mesh,
        node_to_dof=node_to_dof,
        velocity_profile=(times, velocities),
        altitudes=altitudes,
        base_temperature=T_base,
        include_radiation=include_radiation
    )

    print()
    print("RESUME DES RESULTATS")
    print("-" * 80)
    print(f"  - T_max initiale: {T_max_list[0]:.0f} K ({T_max_list[0] - 273.15:.0f} C)")
    print(f"  - T_max maximale: {max(T_max_list):.0f} K ({max(T_max_list) - 273.15:.0f} C)")
    print(f"  - T_max finale:   {T_max_list[-1]:.0f} K ({T_max_list[-1] - 273.15:.0f} C)")
    idx_max = T_max_list.index(max(T_max_list))
    print(f"  - Pic atteint a:  t = {times[idx_max]:.1f} s, V = {velocities[idx_max]:.0f} m/s")
    print("-" * 80)
    print()

    output_dir = "data/output/figures"
    os.makedirs(output_dir, exist_ok=True)

    if export_video:
        save_file = os.path.join(output_dir, f"reentry_animation.{video_format}")
    else:
        save_file = None

    print("LANCEMENT DE L'ANIMATION")
    print("-" * 80)
    if export_video:
        print(f"L'animation sera sauvegardee: {save_file}")
        print("Fermez la fenetre pour terminer l'export.")
    else:
        print("Animation en mode live uniquement (pas d'export)")
    print("-" * 80)
    print()

    anim = create_animation(
        mesh=mesh,
        node_to_dof=node_to_dof,
        times=times,
        velocities=velocities,
        solutions=solutions,
        T_max_list=T_max_list,
        altitudes=altitudes,
        fps=fps,
        save_file=save_file,
        show=True
    )

    print()
    print("=" * 80)
    print("DEMO TERMINEE")
    print("=" * 80)
    if export_video and save_file and os.path.exists(save_file):
        file_size = os.path.getsize(save_file) / (1024 * 1024)
        print(f"Animation sauvegardee: {save_file} ({file_size:.1f} MB)")
    print("=" * 80)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Animation de rentrée atmosphérique avec analyse thermique FEM'
    )
    parser.add_argument(
        '--no-radiation',
        action='store_true',
        help='Désactiver le refroidissement radiatif (modèle linéaire uniquement)'
    )
    args = parser.parse_args()

    include_radiation = not args.no_radiation

    try:
        main(include_radiation=include_radiation)
    except KeyboardInterrupt:
        print("\n\n[WARNING] Interruption par l'utilisateur")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n[ERROR] ERREUR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
