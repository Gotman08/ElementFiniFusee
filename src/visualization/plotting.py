"""
@file plotting.py
@brief Visualization module for FEM thermal analysis results
@author HPC-Code-Documenter
@date 2025

@details
This module provides visualization functions for displaying and exporting
results from rocket thermal finite element analysis.

Visualization capabilities:
- 2D temperature field plots on triangular meshes
- Parametric study curves ($T_{max}$ vs velocity)
- Multi-panel comparison figures
- CSV export for external analysis

All plots are designed for engineering reports with proper annotations,
colorbars, and units.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from src.mesh.mesh_reader import Mesh
from typing import Dict, List


def plot_temperature_field(mesh: Mesh,
                           node_to_dof: Dict[int, int],
                           U: np.ndarray,
                           title: str = "Champ de temperature",
                           save_file: str = None,
                           show: bool = True):
    """
    @brief Plot the temperature field on the finite element mesh.

    @details
    Creates a filled contour plot of the temperature distribution using
    matplotlib's tricontourf for unstructured triangular meshes.

    Features:
    - 20-level contour colormap ('hot' colorscheme)
    - Colorbar with temperature units
    - Min/max temperature annotations
    - Equal aspect ratio for geometric accuracy

    @param mesh: Mesh object containing node coordinates and element connectivity
    @param node_to_dof: Dictionary mapping node IDs to DOF indices
    @param U: Solution vector containing temperatures at each DOF
    @param title: Plot title string
    @param save_file: Path to save the figure (PNG format, 300 DPI)
    @param show: If True, display the plot interactively

    @example
    >>> plot_temperature_field(mesh, node_to_dof, U,
    ...     title="Temperature at V=5000 m/s",
    ...     save_file="temp_field.png")
    """
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

    triangles = []
    for elem in mesh.get_triangles().values():
        tri_nodes = [node_to_dof[nid] for nid in elem['nodes'] if nid in node_to_dof]
        if len(tri_nodes) == 3:
            triangles.append(tri_nodes)

    fig, ax = plt.subplots(figsize=(10, 6))

    triang = tri.Triangulation(x, y, triangles)
    contour = ax.tricontourf(triang, T, levels=20, cmap='hot')

    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Temperature (K)', rotation=270, labelpad=20)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(title)
    ax.set_aspect('equal')

    T_min, T_max = np.min(T), np.max(T)
    ax.text(0.02, 0.98, f'T_min = {T_min:.1f} K\nT_max = {T_max:.1f} K',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save_file:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Figure sauvegardee: {save_file}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_parametric_curve(velocities: List[float],
                          T_max_list: List[float],
                          title: str = "Temperature maximale vs Vitesse",
                          save_file: str = None,
                          show: bool = True):
    """
    @brief Plot the key engineering result: $T_{max}(V)$ curve.

    @details
    Creates a publication-quality plot showing how maximum temperature
    varies with rocket velocity. This is the primary output for
    thermal design decisions.

    Plot features:
    - Data points connected with lines
    - Velocity and temperature range annotations
    - Temperature variation ($\\Delta T$) displayed
    - Grid for easy reading

    @param velocities: List of velocities [m/s]
    @param T_max_list: List of maximum temperatures [K] at each velocity
    @param title: Plot title string
    @param save_file: Path to save the figure (PNG format, 300 DPI)
    @param show: If True, display the plot interactively

    @example
    >>> plot_parametric_curve(velocities, T_max_list,
    ...     title="Thermal Analysis Results",
    ...     save_file="T_max_vs_V.png")
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(velocities, T_max_list, 'o-', linewidth=2, markersize=8,
            color='red', markerfacecolor='orange', markeredgecolor='darkred')

    ax.set_xlabel('Vitesse du lanceur V (m/s)', fontsize=12)
    ax.set_ylabel('Temperature maximale T_max (K)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    V_min, V_max = velocities[0], velocities[-1]
    T_min, T_max = min(T_max_list), max(T_max_list)

    ax.text(0.02, 0.98,
            f'Plage de vitesse: {V_min:.0f} - {V_max:.0f} m/s\n'
            f'Plage de temperature: {T_min:.1f} - {T_max:.1f} K\n'
            f'Variation: DeltaT = {T_max - T_min:.1f} K',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save_file:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Figure sauvegardee: {save_file}")

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
    @brief Plot multiple temperature fields side by side for comparison.

    @details
    Creates a multi-panel figure showing temperature distributions at
    different velocities, useful for visualizing thermal evolution.

    By default, displays three cases:
    - Lowest velocity (index 0)
    - Middle velocity (index n//2)
    - Highest velocity (index -1)

    Each panel includes:
    - Temperature contour plot
    - Individual colorbar
    - Velocity and $T_{max}$ annotation

    @param mesh: Mesh object containing node coordinates and element connectivity
    @param node_to_dof: Dictionary mapping node IDs to DOF indices
    @param velocities: List of velocities corresponding to solutions
    @param solutions: List of solution vectors (temperature fields)
    @param indices: List of indices to plot (default: [0, n//2, n-1])
    @param save_file: Path to save the figure (PNG format, 300 DPI)
    @param show: If True, display the plot interactively

    @example
    >>> plot_multiple_temperature_fields(mesh, node_to_dof,
    ...     velocities, solutions,
    ...     indices=[0, 5, 10],
    ...     save_file="comparison.png")
    """
    if indices is None:
        indices = [0, len(velocities) // 2, len(velocities) - 1]

    n_plots = len(indices)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

    if n_plots == 1:
        axes = [axes]

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
        print(f"Figure sauvegardee: {save_file}")

    if show:
        plt.show()
    else:
        plt.close()


def export_results_to_csv(velocities: List[float],
                          T_max_list: List[float],
                          filename: str):
    """
    @brief Export parametric study results to CSV format.

    @details
    Creates a comma-separated values file with velocity and maximum
    temperature data, suitable for import into spreadsheet software
    or further analysis tools.

    Output format:
    ```
    Velocity_m_s,T_max_K
    1000.00,350.1234
    2000.00,450.5678
    ...
    ```

    @param velocities: List of velocities [m/s]
    @param T_max_list: List of maximum temperatures [K]
    @param filename: Output file path (typically .csv extension)

    @example
    >>> export_results_to_csv(velocities, T_max_list, "results.csv")
    """
    with open(filename, 'w') as f:
        f.write("Velocity_m_s,T_max_K\n")
        for V, T in zip(velocities, T_max_list):
            f.write(f"{V:.2f},{T:.4f}\n")

    print(f"Resultats exportes: {filename}")


if __name__ == '__main__':
    print("Module de visualisation")
    print("Utiliser avec les resultats de l'etude parametrique")
