"""
Module de visualisation des résultats
- Champs de température 2D
- Courbe de température maximale en fonction de la vitesse
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mesh_reader import Mesh
from typing import Dict, List

def plot_temperature_field(mesh: Mesh,
                           node_to_dof: Dict[int, int],
                           U: np.ndarray,
                           title: str = "Champ de température",
                           save_file: str = None,
                           show: bool = True):
    """
    Trace le champ de température sur le maillage

    Args:
        mesh: Objet Mesh
        node_to_dof: Mapping node_id -> DOF index
        U: Vecteur solution (températures aux noeuds)
        title: Titre du graphique
        save_file: Chemin pour sauvegarder (optionnel)
        show: Afficher le graphique
    """
    # Extraire les coordonnées et valeurs
    x_coords = []
    y_coords = []
    temperatures = []

    for node_id, dof in node_to_dof.items():
        coords = mesh.nodes[node_id]
        x_coords.append(coords[0])
        y_coords.append(coords[1])
        temperatures.append(U[dof])

    x = np.array(x_coords)
    y = np.array(y_coords)
    T = np.array(temperatures)

    # Créer la triangulation
    triangles = []
    for elem in mesh.get_triangles().values():
        tri_nodes = [node_to_dof[nid] for nid in elem['nodes'] if nid in node_to_dof]
        if len(tri_nodes) == 3:
            triangles.append(tri_nodes)

    # Tracer
    fig, ax = plt.subplots(figsize=(10, 6))

    triang = tri.Triangulation(x, y, triangles)
    contour = ax.tricontourf(triang, T, levels=20, cmap='hot')

    # Colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Température (K)', rotation=270, labelpad=20)

    # Axes
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(title)
    ax.set_aspect('equal')

    # Annotations
    T_min, T_max = np.min(T), np.max(T)
    ax.text(0.02, 0.98, f'T_min = {T_min:.1f} K\nT_max = {T_max:.1f} K',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save_file:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Figure sauvegardée: {save_file}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_parametric_curve(velocities: List[float],
                          T_max_list: List[float],
                          title: str = "Température maximale vs Vitesse",
                          save_file: str = None,
                          show: bool = True):
    """
    Trace la courbe T_max(V) - RÉSULTAT CLÉ DE L'ÉTUDE

    Args:
        velocities: Liste des vitesses (m/s)
        T_max_list: Liste des températures maximales (K)
        title: Titre du graphique
        save_file: Chemin pour sauvegarder (optionnel)
        show: Afficher le graphique
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Tracer la courbe
    ax.plot(velocities, T_max_list, 'o-', linewidth=2, markersize=8,
            color='red', markerfacecolor='orange', markeredgecolor='darkred')

    # Axes
    ax.set_xlabel('Vitesse du lanceur V (m/s)', fontsize=12)
    ax.set_ylabel('Température maximale T_max (K)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Annotations
    V_min, V_max = velocities[0], velocities[-1]
    T_min, T_max = min(T_max_list), max(T_max_list)

    ax.text(0.02, 0.98,
            f'Plage de vitesse: {V_min:.0f} - {V_max:.0f} m/s\n'
            f'Plage de température: {T_min:.1f} - {T_max:.1f} K\n'
            f'Variation: ΔT = {T_max - T_min:.1f} K',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save_file:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Figure sauvegardée: {save_file}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_multiple_temperature_fields(mesh: Mesh,
                                     node_to_dof: Dict[int, int],
                                     velocities: List[float],
                                     solutions: List[np.ndarray],
                                     indices: List[int] = None,
                                     save_file: str = None,
                                     show: bool = True):
    """
    Trace plusieurs champs de température côte à côte

    Args:
        mesh: Objet Mesh
        node_to_dof: Mapping node_id -> DOF index
        velocities: Liste des vitesses
        solutions: Liste des solutions
        indices: Indices des cas à afficher (ex: [0, len//2, -1])
        save_file: Chemin pour sauvegarder
        show: Afficher le graphique
    """
    if indices is None:
        # Par défaut: afficher 3 cas (min, milieu, max)
        indices = [0, len(velocities) // 2, len(velocities) - 1]

    n_plots = len(indices)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

    if n_plots == 1:
        axes = [axes]

    # Extraire coordonnées et triangulation (une seule fois)
    x_coords = []
    y_coords = []
    for node_id, dof in node_to_dof.items():
        coords = mesh.nodes[node_id]
        x_coords.append(coords[0])
        y_coords.append(coords[1])

    x = np.array(x_coords)
    y = np.array(y_coords)

    triangles = []
    for elem in mesh.get_triangles().values():
        tri_nodes = [node_to_dof[nid] for nid in elem['nodes'] if nid in node_to_dof]
        if len(tri_nodes) == 3:
            triangles.append(tri_nodes)

    triang = tri.Triangulation(x, y, triangles)

    # Tracer chaque cas
    for i, idx in enumerate(indices):
        ax = axes[i]
        V = velocities[idx]
        U = solutions[idx]

        T = np.array([U[dof] for node_id, dof in node_to_dof.items()])

        contour = ax.tricontourf(triang, T, levels=20, cmap='hot')
        plt.colorbar(contour, ax=ax, label='T (K)')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(f'V = {V:.0f} m/s\nT_max = {np.max(T):.1f} K')
        ax.set_aspect('equal')

    plt.tight_layout()

    if save_file:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Figure sauvegardée: {save_file}")

    if show:
        plt.show()
    else:
        plt.close()


def export_results_to_csv(velocities: List[float],
                          T_max_list: List[float],
                          filename: str):
    """
    Exporte les résultats dans un fichier CSV

    Args:
        velocities: Liste des vitesses
        T_max_list: Liste des températures maximales
        filename: Nom du fichier de sortie
    """
    with open(filename, 'w') as f:
        f.write("Velocity_m_s,T_max_K\n")
        for V, T in zip(velocities, T_max_list):
            f.write(f"{V:.2f},{T:.4f}\n")

    print(f"Résultats exportés: {filename}")


if __name__ == '__main__':
    print("Module de visualisation")
    print("Utiliser avec les résultats de l'étude paramétrique")
