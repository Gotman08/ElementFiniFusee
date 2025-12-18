"""
@file animation.py
@brief Live animation module for atmospheric reentry thermal simulation
@details Shows rocket descending through Re-entry Corridor with realistic angle
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Rectangle
import time

from src.mesh.mesh_reader import Mesh
from src.physics.parametric_study import compute_aerothermal_parameters, KAPPA_material, T_inf
from src.core.nonlinear_solver import picard_iteration
from src.core.boundary_conditions import apply_dirichlet_conditions
from src.core.solver import solve_linear_system


def compute_all_frames(mesh, node_to_dof, velocity_profile, altitudes, base_temperature=300.0, include_radiation=True):
    """
    Pre-compute temperature solutions for all frames.

    Parameters
    ----------
    mesh : Mesh
        Finite element mesh
    node_to_dof : Dict[int, int]
        Node to DOF mapping
    velocity_profile : Tuple[np.ndarray, np.ndarray]
        (times, velocities) arrays
    altitudes : np.ndarray
        Altitude profile [m] corresponding to velocity_profile
    base_temperature : float
        Base temperature for Dirichlet BC [K]
    include_radiation : bool
        Enable radiation cooling (default: True for realistic temperatures)

    Returns
    -------
    times, velocities, solutions, T_max_list
    """
    times, velocities = velocity_profile
    n_frames = len(times)

    solutions = []
    T_max_list = []

    print("=" * 70)
    print("PRE-CALCUL DES SOLUTIONS POUR L'ANIMATION")
    print("=" * 70)
    print(f"Nombre de frames: {n_frames}")
    print(f"Physique de radiation: {'ACTIVEE (temperatures realistes)' if include_radiation else 'DESACTIVEE'}")
    print(f"Couplage altitude-temperature: ACTIF")
    print()

    start_time = time.time()

    for idx, (t, V, h) in enumerate(zip(times, velocities, altitudes)):
        progress = (idx + 1) / n_frames * 100
        print(f"\r[{idx+1}/{n_frames}] Frame {idx+1} | h={h/1000:.0f} km, V={V:.0f} m/s | {progress:.1f}%", end='')

        # Get altitude-corrected aerothermal parameters
        from src.physics.parametric_study import compute_altitude_corrected_parameters
        alpha, u_E, epsilon, sigma = compute_altitude_corrected_parameters(V, h)

        robin_boundaries = {1: (alpha, u_E)}

        # Check if Dirichlet BC physical group exists
        dirichlet_boundaries = {}
        if 2 in mesh.physical_groups:
            dirichlet_boundaries = {2: base_temperature}

        if include_radiation:
            # Use nonlinear solver with radiation
            radiation_boundaries = {1: (epsilon, T_inf, sigma)}

            U, info = picard_iteration(
                mesh, node_to_dof, KAPPA_material,
                robin_boundaries, radiation_boundaries,
                dirichlet_boundaries=dirichlet_boundaries if dirichlet_boundaries else None,
                max_iter=100,
                tol_abs=10.0,
                tol_rel=1e-3,
                relaxation=0.3,
                verbose=False
            )
        else:
            # Use linear solver (convection only, for comparison)
            from src.core.assembly import assemble_global_system
            A, F = assemble_global_system(mesh, node_to_dof, KAPPA_material, robin_boundaries)
            if dirichlet_boundaries:
                A_bc, F_bc = apply_dirichlet_conditions(A, F, mesh, node_to_dof, dirichlet_boundaries)
            else:
                A_bc, F_bc = A.copy(), F.copy()
            U = solve_linear_system(A_bc, F_bc, method='direct')

        solutions.append(U)
        T_max_list.append(np.max(U))

    elapsed = time.time() - start_time
    print(f"\n\n[OK] Pre-calcul termine en {elapsed:.1f} s")
    print(f"     Temps moyen par frame: {elapsed/n_frames:.3f} s")
    print("=" * 70)
    print()

    return times, velocities, solutions, T_max_list


def draw_atmospheric_layers(ax, x_min, x_max):
    """Draw atmospheric layers as background."""
    layers = [
        (-5, 0, '#228B22', 'Sol'),
        (0, 12, '#87CEEB', 'Troposphere'),
        (12, 50, '#4169E1', 'Stratosphere'),
        (50, 80, '#191970', 'Mesosphere'),
        (80, 100, '#2E0854', 'Thermosphere'),
        (100, 300, '#000005', 'Espace'),  # Espace étendu jusqu'à 300 km
    ]

    for y_min, y_max, color, name in layers:
        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                         facecolor=color, alpha=0.6, edgecolor='none')
        ax.add_patch(rect)
        if y_min >= 0:
            ax.text(x_max - 2, (y_min + y_max) / 2, f'{name}\n{y_min}-{y_max} km',
                   fontsize=8, color='white', ha='right', va='center', alpha=0.9)


def rotate_points(x, y, angle_deg, cx=0, cy=0):
    """Rotate points around center (cx, cy) by angle in degrees."""
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Translate to origin
    x_shifted = x - cx
    y_shifted = y - cy

    # Rotate
    x_rot = x_shifted * cos_a - y_shifted * sin_a
    y_rot = x_shifted * sin_a + y_shifted * cos_a

    # Translate back
    return x_rot + cx, y_rot + cy


def create_animation(mesh, node_to_dof, times, velocities, solutions, T_max_list,
                     altitudes=None, fps=20, save_file=None, show=True,
                     reentry_angle=25):
    """
    Create animation showing rocket descending through Re-entry Corridor.

    Args:
        reentry_angle: Angle of reentry in degrees (from horizontal)
    """
    # Extract mesh coordinates
    x_coords = []
    y_coords = []
    for node_id, dof in node_to_dof.items():
        coords = mesh.nodes[node_id]
        x_coords.append(coords[0])
        y_coords.append(coords[1])

    x_right = np.array(x_coords)
    y_right = np.array(y_coords)
    n_nodes = len(x_right)

    # Mirror for symmetric display
    x_left = -x_right
    y_left = y_right

    x_base = np.concatenate([x_left, x_right])
    y_base = np.concatenate([y_left, y_right])

    # Scale rocket for visibility (rocket ~2.75m -> display ~8 km)
    rocket_scale = 3.0
    x_scaled = x_base * rocket_scale
    y_scaled = y_base * rocket_scale

    # Rotate rocket to match reentry angle (tilted)
    # Negative angle because rocket nose points in direction of travel
    x_rotated, y_rotated = rotate_points(x_scaled, y_scaled, -(90 - reentry_angle))

    # Triangles
    triangles_right = []
    for elem in mesh.get_triangles().values():
        tri_nodes = [node_to_dof[nid] for nid in elem['nodes'] if nid in node_to_dof]
        if len(tri_nodes) == 3:
            triangles_right.append(tri_nodes)

    triangles_left = [[tri[0] + n_nodes, tri[2] + n_nodes, tri[1] + n_nodes]
                      for tri in triangles_right]
    triangles = triangles_left + triangles_right

    # Convert altitudes to km
    if altitudes is not None:
        altitudes_km = np.array(altitudes) / 1000.0
    else:
        altitudes_km = np.linspace(80, 5, len(times))

    # Calculate horizontal positions with CURVED trajectory
    # L'angle diminue progressivement (raide en haut, plat en bas)
    alt_max = altitudes_km[0]
    alt_min = altitudes_km[-1]

    # Progress: 1.0 en haut (espace), 0.0 en bas (sol)
    progress = (altitudes_km - alt_min) / (alt_max - alt_min)

    # Angle varie: 25° en haut → ~7° en bas (trajectoire qui s'aplatit)
    angle_varying = reentry_angle * (0.3 + 0.7 * progress)

    # Calculer x par intégration (cumsum des déplacements horizontaux)
    d_altitude = -np.gradient(altitudes_km)  # Changement d'altitude (négatif = descente)
    d_horizontal = d_altitude / np.tan(np.radians(angle_varying))
    x_positions = np.cumsum(d_horizontal)

    # Centrer la trajectoire
    horizontal_range = x_positions[-1] - x_positions[0]
    x_positions = x_positions - x_positions[0] - horizontal_range / 2

    # Calculer distance totale parcourue (3D : horizontal + vertical)
    dx = np.diff(x_positions, prepend=x_positions[0])
    dh = np.diff(altitudes_km, prepend=altitudes_km[0])
    ds = np.sqrt(dx**2 + dh**2)  # Distance élémentaire [km]
    distance_traveled_km = np.cumsum(ds)  # Distance cumulée [km]

    T_min_global = min([np.min(U) for U in solutions])
    T_max_global = max([np.max(U) for U in solutions])

    print("=" * 70)
    print("CREATION DE L'ANIMATION - RE-ENTRY CORRIDOR")
    print("=" * 70)
    print(f"Nombre de frames: {len(times)}")
    print(f"FPS: {fps}")
    print(f"Duree animation: {len(times)/fps:.1f} secondes")
    print(f"Altitude: {altitudes_km[0]:.1f} -> {altitudes_km[-1]:.1f} km")
    print(f"Angle de rentree: {reentry_angle} degres")
    print(f"Distance horizontale: {horizontal_range:.1f} km")
    print(f"Echelle temperature: {T_min_global:.0f} - {T_max_global:.0f} K")
    print("=" * 70)
    print()

    # Create figure
    fig = plt.figure(figsize=(20, 14))  # Légèrement plus grand pour panneau thermal
    gs = fig.add_gridspec(4, 4,
                          height_ratios=[3, 3, 1.5, 1.5],  # Plus d'espace pour trajectoire
                          hspace=0.4, wspace=0.3)

    # Trajectoire (2 lignes supérieures, 3 colonnes gauche)
    ax_main = fig.add_subplot(gs[0:2, :3])

    # NOUVEAU: Panneau thermal mesh (2 lignes inférieures, 3 colonnes gauche)
    ax_thermal = fig.add_subplot(gs[2:4, :3])

    # Barre latérale droite (inchangée mais ajustée)
    ax_velocity = fig.add_subplot(gs[0, 3])
    ax_altitude = fig.add_subplot(gs[1, 3])
    ax_temperature = fig.add_subplot(gs[2, 3])
    ax_info = fig.add_subplot(gs[3, 3])
    ax_info.axis('off')

    fig.suptitle('RENTREE ATMOSPHERIQUE ARIANE 5 - Visualisation Thermique Detaillee',
                 fontsize=16, fontweight='bold', color='white')
    fig.patch.set_facecolor('#1a1a2e')

    # Setup main view - wider to show horizontal movement
    x_view_min = -horizontal_range - 20
    x_view_max = horizontal_range / 2 + 20
    y_view_min, y_view_max = -10, 280  # Vue très éloignée pour voir depuis l'espace (~250 km)

    ax_main.set_xlim(x_view_min, x_view_max)
    ax_main.set_ylim(y_view_min, y_view_max)
    ax_main.set_facecolor('#000011')

    draw_atmospheric_layers(ax_main, x_view_min, x_view_max)

    ax_main.set_xlabel('Distance horizontale (km)', fontsize=11, color='white')
    ax_main.set_ylabel('Altitude (km)', fontsize=11, color='white')
    ax_main.set_title(f'Trajectoire de rentree (angle: {reentry_angle}°)', fontsize=12, color='white')
    ax_main.tick_params(colors='white')

    # Add horizontal lines for layer boundaries
    for alt in [0, 12, 50, 80]:
        ax_main.axhline(y=alt, color='white', linestyle='--', linewidth=0.5, alpha=0.3)

    # Draw full planned trajectory (dotted line)
    ax_main.plot(x_positions, altitudes_km, 'w--', linewidth=1, alpha=0.3, label='Trajectoire prevue')

    # Trajectory trail (will be updated)
    trail_line, = ax_main.plot([], [], 'orange', linewidth=2, alpha=0.7, label='Trajectoire parcourue')

    # Velocity subplot
    ax_velocity.set_facecolor('#1a1a2e')
    ax_velocity.plot(times, velocities, 'cyan', linewidth=2, alpha=0.5)
    point_velocity, = ax_velocity.plot([], [], 'ro', markersize=8)
    ax_velocity.set_xlabel('Temps (s)', color='white')
    ax_velocity.set_ylabel('Vitesse (m/s)', color='white')
    ax_velocity.set_title('Vitesse', color='white')
    ax_velocity.tick_params(colors='white')
    ax_velocity.grid(True, alpha=0.3, color='gray')

    # Altitude subplot
    ax_altitude.set_facecolor('#1a1a2e')
    ax_altitude.plot(times, altitudes_km, 'lime', linewidth=2, alpha=0.5)
    point_altitude, = ax_altitude.plot([], [], 'ro', markersize=8)
    ax_altitude.set_xlabel('Temps (s)', color='white')
    ax_altitude.set_ylabel('Altitude (km)', color='white')
    ax_altitude.set_title('Altitude', color='white')
    ax_altitude.tick_params(colors='white')
    ax_altitude.grid(True, alpha=0.3, color='gray')

    # Temperature subplot
    T_max_array = np.array(T_max_list)
    ax_temperature.set_facecolor('#1a1a2e')
    ax_temperature.plot(times, T_max_array, color='#FF6B35', linewidth=2, alpha=0.5)
    point_temperature, = ax_temperature.plot([], [], 'ro', markersize=8)
    ax_temperature.set_xlabel('Temps (s)', color='white')
    ax_temperature.set_ylabel('T_max (K)', color='white')
    ax_temperature.set_title('Temperature Max', color='white')
    ax_temperature.tick_params(colors='white')
    ax_temperature.grid(True, alpha=0.3, color='gray')

    # Info text
    info_text = ax_info.text(0.05, 0.5, '', fontsize=10, verticalalignment='center',
                             family='monospace', color='white')
    ax_info.set_facecolor('#1a1a2e')

    # ============================================================================
    # THERMAL MESH PANEL SETUP
    # ============================================================================

    # Configure thermal panel appearance
    ax_thermal.set_facecolor('#0a0a0a')  # Fond sombre pour contraste
    ax_thermal.set_xlabel('Position axiale (m)', fontsize=10, color='white')
    ax_thermal.set_ylabel('Position radiale (m)', fontsize=10, color='white')
    ax_thermal.set_title('Distribution de temperature - Maillage elements finis',
                         fontsize=11, color='white', fontweight='bold')
    ax_thermal.tick_params(colors='white', labelsize=9)
    ax_thermal.set_aspect('equal')

    # Extract native mesh coordinates (right half only)
    x_thermal_right = []
    y_thermal_right = []
    for node_id, dof in node_to_dof.items():
        coords = mesh.nodes[node_id]
        x_thermal_right.append(coords[0])  # Position axiale
        y_thermal_right.append(coords[1])  # Position radiale

    x_thermal_right = np.array(x_thermal_right)
    y_thermal_right = np.array(y_thermal_right)
    n_nodes_thermal = len(x_thermal_right)

    # CREATE MIRROR SYMMETRY (like trajectory panel)
    x_thermal_left = -x_thermal_right  # Mirror along Y-axis
    y_thermal_left = y_thermal_right   # Same radial coordinate

    # Concatenate left and right halves for full rocket
    x_thermal = np.concatenate([x_thermal_left, x_thermal_right])
    y_thermal = np.concatenate([y_thermal_left, y_thermal_right])

    # Build triangulation with symmetry
    triangles_thermal_right = []
    for elem in mesh.get_triangles().values():
        tri_nodes = [node_to_dof[nid] for nid in elem['nodes'] if nid in node_to_dof]
        if len(tri_nodes) == 3:
            triangles_thermal_right.append(tri_nodes)

    # CREATE MIRROR TRIANGLES (indices offset by n_nodes_thermal, order reversed)
    triangles_thermal_left = [[tri[0] + n_nodes_thermal, tri[2] + n_nodes_thermal, tri[1] + n_nodes_thermal]
                              for tri in triangles_thermal_right]
    triangles_thermal = triangles_thermal_left + triangles_thermal_right

    triang_thermal = tri.Triangulation(x_thermal, y_thermal, triangles_thermal)

    # Set mesh bounds with padding
    x_min, x_max = x_thermal.min(), x_thermal.max()
    y_min, y_max = y_thermal.min(), y_thermal.max()
    padding_x = (x_max - x_min) * 0.05
    padding_y = (y_max - y_min) * 0.1
    ax_thermal.set_xlim(x_min - padding_x, x_max + padding_x)
    ax_thermal.set_ylim(y_min - padding_y, y_max + padding_y)

    # Initialize lists for thermal artists
    thermal_artists = []
    thermal_cbar_initialized = False

    # Store artists to remove
    artists_to_remove = []

    def update(frame):
        nonlocal artists_to_remove, thermal_artists, thermal_cbar_initialized

        # Remove previous frame's artists
        for artist in artists_to_remove:
            try:
                artist.remove()
            except:
                pass
        artists_to_remove = []

        # Get current position
        alt_km = altitudes_km[frame]
        x_pos = x_positions[frame]

        # Position rocket at current location
        x_frame = x_rotated + x_pos
        y_frame = y_rotated + alt_km

        # Create triangulation at current position
        triang = tri.Triangulation(x_frame, y_frame, triangles)

        # Get temperatures
        try:
            T_frame_right = np.array([solutions[frame][dof] for node_id, dof in node_to_dof.items()])
        except KeyError as e:
            raise ValueError(
                f"node_to_dof mapping inconsistent: clé manquante {e}. "
                f"Vérifier l'assembly du système."
            )
        T_frame = np.concatenate([T_frame_right, T_frame_right])

        # Draw rocket with temperature colors
        contour = ax_main.tricontourf(triang, T_frame, levels=20, cmap='hot',
                                       vmin=T_min_global, vmax=T_max_global)
        if hasattr(contour, 'collections'):
            artists_to_remove.extend(contour.collections)
        else:
            artists_to_remove.append(contour)

        # Draw rocket outline
        lines = ax_main.triplot(triang, 'w-', linewidth=0.4, alpha=0.6)
        artists_to_remove.extend(lines)

        # ========================================================================
        # UPDATE THERMAL MESH PANEL
        # ========================================================================

        # Clear previous thermal artists
        for artist in thermal_artists:
            try:
                artist.remove()
            except:
                pass
        thermal_artists.clear()

        # Get current temperature field (right half only from solution)
        T_thermal_right = np.array([solutions[frame][dof] for node_id, dof in node_to_dof.items()])

        # DUPLICATE for symmetric display (left half has same temperatures)
        T_thermal = np.concatenate([T_thermal_right, T_thermal_right])

        # Draw filled contours with 30 levels for smooth gradients
        thermal_contour = ax_thermal.tricontourf(
            triang_thermal, T_thermal,
            levels=30,
            cmap='RdYlBu_r',  # Rouge=chaud, Bleu=froid
            vmin=T_min_global,
            vmax=T_max_global,
            extend='both'
        )

        # Store contour collections
        if hasattr(thermal_contour, 'collections'):
            thermal_artists.extend(thermal_contour.collections)

        # Overlay mesh wireframe (thin white lines)
        thermal_wireframe = ax_thermal.triplot(
            triang_thermal,
            'w-',
            linewidth=0.3,
            alpha=0.25
        )
        thermal_artists.extend(thermal_wireframe)

        # Temperature annotation (min/max on mesh)
        T_min_current = np.min(T_thermal)
        T_max_current = np.max(T_thermal)
        temp_annotation = ax_thermal.text(
            0.02, 0.98,
            f'T_min: {T_min_current:.0f} K ({T_min_current-273.15:.0f}°C)\n'
            f'T_max: {T_max_current:.0f} K ({T_max_current-273.15:.0f}°C)',
            transform=ax_thermal.transAxes,
            fontsize=9,
            color='white',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor='white')
        )
        thermal_artists.append(temp_annotation)

        # Velocity indicator on thermal panel
        V = velocities[frame]
        vel_annotation = ax_thermal.text(
            0.98, 0.98,
            f'V = {V:.0f} m/s',
            transform=ax_thermal.transAxes,
            fontsize=10,
            color='cyan',
            verticalalignment='top',
            horizontalalignment='right',
            fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor='cyan')
        )
        thermal_artists.append(vel_annotation)

        # Initialize colorbar on first frame
        if not thermal_cbar_initialized:
            thermal_cbar = plt.colorbar(thermal_contour, ax=ax_thermal,
                                        orientation='vertical',
                                        fraction=0.046, pad=0.04)
            thermal_cbar.set_label('Temperature (K)', rotation=270, labelpad=20,
                                   color='white', fontsize=10)
            thermal_cbar.ax.tick_params(colors='white', labelsize=8)
            thermal_cbar_initialized = True

        # Update trajectory trail
        trail_line.set_data(x_positions[:frame+1], altitudes_km[:frame+1])

        # Update velocity marker
        point_velocity.set_data([times[frame]], [velocities[frame]])

        # Update altitude marker
        point_altitude.set_data([times[frame]], [alt_km])

        # Update temperature marker
        point_temperature.set_data([times[frame]], [T_max_list[frame]])

        # Update info text
        t = times[frame]
        V = velocities[frame]
        T_max = T_max_list[frame]

        info_str = f"""
  Temps:      {t:6.1f} s
  Altitude:   {alt_km:6.1f} km
  Distance:   {distance_traveled_km[frame]:6.1f} km
  Vitesse:    {V:6.0f} m/s
  T_max:      {T_max:6.0f} K
              ({T_max - 273.15:6.0f} C)

  Frame: {frame+1}/{len(times)}
"""
        info_text.set_text(info_str)

        return [trail_line, point_velocity, point_altitude, point_temperature, info_text]

    anim = FuncAnimation(fig, update, frames=len(times),
                        blit=False, interval=1000/fps, repeat=True)

    if save_file:
        print(f"Sauvegarde de l'animation: {save_file}")
        print("Cela peut prendre quelques minutes...")

        try:
            if save_file.endswith('.mp4'):
                writer = FFMpegWriter(fps=fps, bitrate=3000)
                anim.save(save_file, writer=writer, dpi=150)
            elif save_file.endswith('.gif'):
                anim.save(save_file, writer='pillow', fps=fps, dpi=80)
            else:
                print(f"[WARNING] Format non reconnu: {save_file}")

            print(f"[OK] Animation sauvegardee: {save_file}")
        except Exception as e:
            print(f"[WARNING] Erreur lors de la sauvegarde: {e}")
            print("          L'animation sera affichee mais pas sauvegardee.")

    if show:
        plt.show()

    return anim


if __name__ == '__main__':
    print("Module d'animation de rentree atmospherique")
    print("Utiliser via demo_reentry.py")
