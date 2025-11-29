"""
DEMO LIVE - RENTREE ATMOSPHERIQUE
Animation de l'evolution de la temperature pendant la rentree
"""
import numpy as np
import os
import sys

from mesh_reader import read_gmsh_mesh, create_node_mapping
from reentry_profile import generate_reentry_profile, generate_altitude_profile
from live_animation import compute_all_frames, create_animation


def main():
    """
    Fonction principale de la demo live
    """
    print("\n")
    print("=" * 80)
    print("              DEMO LIVE - RENTREE ATMOSPHERIQUE")
    print("           Animation de l'evolution de la temperature")
    print("=" * 80)
    print()

    # ========================================================================
    # CONFIGURATION - PARAMETRES AU DEBUT
    # ========================================================================

    print("PARAMETRES DE LA SIMULATION")
    print("-" * 80)

    # Parametres du profil de rentree
    V_initial = 7000.0      # m/s - Vitesse initiale (orbitale)
    V_final = 500.0         # m/s - Vitesse finale (apres freinage)
    duration = 300.0        # s   - Duree de la rentree
    n_frames = 50           # Nombre de frames (augmenter pour plus de fluidite)

    # Parametres d'animation
    fps = 10                # Images par seconde
    export_video = True     # Exporter en video ?
    video_format = 'gif'    # 'mp4' ou 'gif'

    # Temperature de base
    T_base = 300.0          # K

    # Fichier de maillage
    mesh_file = "rocket_mesh.msh"

    print(f"  - Vitesse initiale:     {V_initial:.0f} m/s")
    print(f"  - Vitesse finale:       {V_final:.0f} m/s")
    print(f"  - Duree rentree:        {duration:.0f} s")
    print(f"  - Nombre de frames:     {n_frames}")
    print(f"  - FPS:                  {fps}")
    print(f"  - Temperature base:     {T_base:.0f} K")
    if export_video:
        print(f"  - Export video:         OUI (.{video_format})")
    else:
        print(f"  - Export video:         NON")
    print("-" * 80)
    print()

    # ========================================================================
    # VERIFICATION DU MAILLAGE
    # ========================================================================

    if not os.path.exists(mesh_file):
        print(f"[ERROR] Fichier de maillage non trouve: {mesh_file}")
        print(f"        Executer d'abord: python generate_mesh_python.py")
        sys.exit(1)

    # ========================================================================
    # GENERATION DU PROFIL DE RENTREE
    # ========================================================================

    print("GENERATION DU PROFIL DE RENTREE")
    print("-" * 80)

    times, velocities = generate_reentry_profile(
        V_initial=V_initial,
        V_final=V_final,
        duration=duration,
        n_points=n_frames,
        profile_type='exponential'  # Freinage realiste
    )

    altitudes = generate_altitude_profile(times, velocities, h_initial=80000.0)

    print(f"[OK] Profil genere:")
    print(f"     - Deceleration: {V_initial:.0f} -> {V_final:.0f} m/s")
    print(f"     - Altitude: {altitudes[0]/1000:.1f} -> {altitudes[-1]/1000:.1f} km")
    print(f"     - Deceleration max: {np.max(-np.gradient(velocities, times)):.1f} m/s^2")
    print("-" * 80)
    print()

    # ========================================================================
    # CHARGEMENT DU MAILLAGE
    # ========================================================================

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

    # ========================================================================
    # PRE-CALCUL DES SOLUTIONS
    # ========================================================================

    times, velocities, solutions, T_max_list = compute_all_frames(
        mesh=mesh,
        node_to_dof=node_to_dof,
        velocity_profile=(times, velocities),
        base_temperature=T_base
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

    # ========================================================================
    # CREATION DE L'ANIMATION
    # ========================================================================

    output_dir = "resultats"
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
