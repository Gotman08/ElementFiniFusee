"""
Module d'animation live de rentree atmospherique
Utilise matplotlib.animation pour visualiser l'evolution de la temperature
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Rectangle
import time

from mesh_reader import Mesh
from parametric_study import compute_aerothermal_parameters, KAPPA_material
from assembly import assemble_global_system
from boundary_conditions import apply_dirichlet_conditions
from solver import solve_linear_system


def compute_all_frames(mesh, node_to_dof, velocity_profile, base_temperature=300.0):
    """
    Pre-calcule toutes les solutions pour chaque vitesse du profil

    Args:
        mesh: Objet Mesh
        node_to_dof: Mapping node -> DOF
        velocity_profile: Tuple (times, velocities)
        base_temperature: Temperature a la base (K)

    Returns:
        (times, velocities, solutions, T_max_list): Donnees pour l'animation
    """
    times, velocities = velocity_profile
    n_frames = len(times)

    solutions = []
    T_max_list = []

    print("=" * 70)
    print("PRE-CALCUL DES SOLUTIONS POUR L'ANIMATION")
    print("=" * 70)
    print(f"Nombre de frames: {n_frames}")
    print()

    start_time = time.time()

    for idx, (t, V) in enumerate(zip(times, velocities)):
        # Barre de progression
        progress = (idx + 1) / n_frames * 100
        print(f"\r[{idx+1}/{n_frames}] Frame {idx+1} | V = {V:.0f} m/s | {progress:.1f}%", end='')

        # Calculer les parametres aerothermiques
        alpha, u_E = compute_aerothermal_parameters(V)

        # Assembler le systeme avec Robin
        robin_boundaries = {1: (alpha, u_E)}
        A, F = assemble_global_system(mesh, node_to_dof, KAPPA_material, robin_boundaries)

        # Appliquer Dirichlet
        dirichlet_boundaries = {2: base_temperature}
        A_bc, F_bc = apply_dirichlet_conditions(A, F, mesh, node_to_dof, dirichlet_boundaries)

        # Resoudre
        U = solve_linear_system(A_bc, F_bc, method='direct')

        # Stocker
        solutions.append(U)
        T_max_list.append(np.max(U))

    elapsed = time.time() - start_time
    print(f"\n\n[OK] Pre-calcul termine en {elapsed:.1f} s")
    print(f"     Temps moyen par frame: {elapsed/n_frames:.3f} s")
    print("=" * 70)
    print()

    return times, velocities, solutions, T_max_list


def create_animation(mesh, node_to_dof, times, velocities, solutions, T_max_list,
                     fps=20, save_file=None, show=True):
    """
    Cree l'animation de la rentree atmospherique

    Args:
        mesh: Objet Mesh
        node_to_dof: Mapping node -> DOF
        times: Array des temps (s)
        velocities: Array des vitesses (m/s)
        solutions: Liste des champs de temperature
        T_max_list: Liste des temperatures maximales
        fps: Images par seconde
        save_file: Fichier de sortie (.mp4 ou .gif)
        show: Afficher l'animation
    """
    # Preparer les donnees de visualisation
    x_coords = []
    y_coords = []
    for node_id, dof in node_to_dof.items():
        coords = mesh.nodes[node_id]
        x_coords.append(coords[0])
        y_coords.append(coords[1])

    x = np.array(x_coords)
    y = np.array(y_coords)

    # Triangulation
    triangles = []
    for elem in mesh.get_triangles().values():
        tri_nodes = [node_to_dof[nid] for nid in elem['nodes'] if nid in node_to_dof]
        if len(tri_nodes) == 3:
            triangles.append(tri_nodes)

    triang = tri.Triangulation(x, y, triangles)

    # Determiner les limites de temperature pour l'echelle de couleur
    T_min_global = min([np.min(U) for U in solutions])
    T_max_global = max([np.max(U) for U in solutions])

    print("=" * 70)
    print("CREATION DE L'ANIMATION")
    print("=" * 70)
    print(f"Nombre de frames: {len(times)}")
    print(f"FPS: {fps}")
    print(f"Duree: {times[-1]:.1f} s")
    print(f"Echelle temperature: {T_min_global:.0f} - {T_max_global:.0f} K")
    print("=" * 70)
    print()

    # Creer la figure avec subplots
    fig = plt.figure(figsize=(16, 9))

    # Layout: champ de temperature (grand) + graphiques secondaires
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    ax_temp = fig.add_subplot(gs[:, :2])  # Champ de temperature (2/3 largeur)
    ax_velocity = fig.add_subplot(gs[0, 2])  # Vitesse
    ax_tmax = fig.add_subplot(gs[1, 2])  # T_max
    ax_info = fig.add_subplot(gs[2, 2])  # Informations textuelles
    ax_info.axis('off')

    # Titre principal
    fig.suptitle('RENTREE ATMOSPHERIQUE - ANALYSE THERMIQUE', fontsize=16, fontweight='bold')

    # Initialisation du champ de temperature
    T_init = np.array([solutions[0][dof] for node_id, dof in node_to_dof.items()])
    contour = ax_temp.tricontourf(triang, T_init, levels=20, cmap='hot', vmin=T_min_global, vmax=T_max_global)

    # Ajouter les contours du maillage pour voir la forme de la fusee
    ax_temp.triplot(triang, 'w-', linewidth=0.8, alpha=0.8)

    cbar = plt.colorbar(contour, ax=ax_temp, label='Temperature (K)')

    ax_temp.set_xlabel('x (m)')
    ax_temp.set_ylabel('y (m)')
    ax_temp.set_title('Champ de Temperature - Fusee')
    ax_temp.set_aspect('equal')

    # Graphique vitesse
    line_velocity, = ax_velocity.plot(times, velocities, 'b-', linewidth=2, alpha=0.3)
    point_velocity, = ax_velocity.plot([], [], 'ro', markersize=10)
    ax_velocity.set_xlabel('Temps (s)')
    ax_velocity.set_ylabel('Vitesse (m/s)')
    ax_velocity.set_title('Profil de Vitesse')
    ax_velocity.grid(True, alpha=0.3)

    # Graphique T_max
    line_tmax, = ax_tmax.plot([], [], 'r-', linewidth=2)
    ax_tmax.set_xlabel('Temps (s)')
    ax_tmax.set_ylabel('T_max (K)')
    ax_tmax.set_title('Temperature Maximale')
    ax_tmax.set_xlim(0, times[-1])
    ax_tmax.set_ylim(T_min_global * 0.95, T_max_global * 1.05)
    ax_tmax.grid(True, alpha=0.3)

    # Texte d'informations
    info_text = ax_info.text(0.1, 0.5, '', fontsize=12, verticalalignment='center',
                             family='monospace')

    # Fonction d'initialisation
    def init():
        point_velocity.set_data([], [])
        line_tmax.set_data([], [])
        return contour, point_velocity, line_tmax, info_text

    # Fonction de mise a jour pour chaque frame
    def update(frame):
        # Mettre a jour le champ de temperature
        T_frame = np.array([solutions[frame][dof] for node_id, dof in node_to_dof.items()])

        # Supprimer l'ancien contour et en creer un nouveau
        for coll in ax_temp.collections:
            coll.remove()
        contour = ax_temp.tricontourf(triang, T_frame, levels=20, cmap='hot',
                                       vmin=T_min_global, vmax=T_max_global)

        # Ajouter les contours du maillage pour voir la forme de la fusee
        ax_temp.triplot(triang, 'w-', linewidth=0.8, alpha=0.8)

        # Mettre a jour le point sur le graphique de vitesse
        point_velocity.set_data([times[frame]], [velocities[frame]])

        # Mettre a jour la courbe T_max
        line_tmax.set_data(times[:frame+1], T_max_list[:frame+1])

        # Mettre a jour les informations textuelles
        t = times[frame]
        V = velocities[frame]
        T_max = T_max_list[frame]

        info_str = f"""
Temps:          {t:6.1f} s
Vitesse:        {V:6.0f} m/s
T_max:          {T_max:6.0f} K
                ({T_max - 273.15:6.0f} C)

Frame:  {frame+1}/{len(times)}
"""
        info_text.set_text(info_str)

        return contour, point_velocity, line_tmax, info_text

    # Creer l'animation
    anim = FuncAnimation(fig, update, frames=len(times), init_func=init,
                        blit=False, interval=1000/fps, repeat=True)

    # Sauvegarder si demande
    if save_file:
        print(f"Sauvegarde de l'animation: {save_file}")
        print("Cela peut prendre quelques minutes...")

        if save_file.endswith('.mp4'):
            writer = FFMpegWriter(fps=fps, bitrate=2000)
            anim.save(save_file, writer=writer, dpi=150)
        elif save_file.endswith('.gif'):
            writer = PillowWriter(fps=fps)
            anim.save(save_file, writer=writer, dpi=100)
        else:
            print(f"[WARNING] Format non reconnu: {save_file}")
            print("            Utilisez .mp4 ou .gif")

        print(f"[OK] Animation sauvegardee: {save_file}")

    # Afficher
    if show:
        plt.show()

    return anim


if __name__ == '__main__':
    print("Module d'animation de rentree atmospherique")
    print("Utiliser via demo_reentry.py")
