"""
@file reentry_profile.py
@brief Atmospheric reentry trajectory and velocity profile generation
@author HPC-Code-Documenter
@date 2025

@details
This module generates realistic velocity and altitude profiles for
atmospheric reentry simulations. It models the deceleration of a
rocket during descent through the atmosphere.

Supported deceleration profiles:
- Exponential: $V(t) = V_f + (V_0 - V_f) e^{-t/\\tau}$ (most realistic)
- Linear: $V(t) = V_0 - (V_0 - V_f) \\frac{t}{T}$
- Quadratic: $V(t) = V_0 - (V_0 - V_f) \\left(\\frac{t}{T}\\right)^2$

The exponential profile best represents actual atmospheric braking
where drag force is proportional to $V^2$, causing rapid initial
deceleration followed by gradual slowdown.

Typical reentry parameters:
- Initial velocity: 7000 m/s (orbital velocity at 80 km altitude)
- Final velocity: 500 m/s (terminal velocity)
- Duration: 300 s (5 minutes)
"""
import numpy as np
import matplotlib.pyplot as plt


def generate_reentry_profile(V_initial=7000.0,
                             V_final=500.0,
                             duration=300.0,
                             n_points=100,
                             profile_type='exponential'):
    """
    @brief Generate a realistic velocity profile for atmospheric reentry.

    @details
    Creates a time-velocity trajectory representing rocket deceleration
    during atmospheric descent. Three profile types are available:

    Exponential (realistic):
    $V(t) = V_f + (V_0 - V_f) \\exp(-t/\\tau)$
    where $\\tau = T/3$ is the time constant.

    Linear (simplified):
    $V(t) = V_0 - (V_0 - V_f) \\frac{t}{T}$

    Quadratic (aggressive early braking):
    $V(t) = V_0 - (V_0 - V_f) \\left(\\frac{t}{T}\\right)^2$

    @param V_initial: Initial velocity at reentry interface [m/s]
    @param V_final: Final velocity at end of profile [m/s]
    @param duration: Total reentry duration [s]
    @param n_points: Number of time points to generate
    @param profile_type: Deceleration model - 'exponential', 'linear', or 'quadratic'

    @return Tuple (times, velocities):
        - times: NumPy array of time points [s]
        - velocities: NumPy array of corresponding velocities [m/s]

    @raises ValueError: If profile_type is not recognized

    @example
    >>> times, velocities = generate_reentry_profile(
    ...     V_initial=7000.0, V_final=500.0, duration=300.0
    ... )
    """
    times = np.linspace(0, duration, n_points)

    if profile_type == 'exponential':
        tau = duration / 3.0
        velocities = V_final + (V_initial - V_final) * np.exp(-times / tau)

    elif profile_type == 'linear':
        velocities = V_initial - (V_initial - V_final) * (times / duration)

    elif profile_type == 'quadratic':
        normalized_t = times / duration
        velocities = V_initial - (V_initial - V_final) * normalized_t**2

    else:
        raise ValueError(f"Type de profil inconnu: {profile_type}")

    return times, velocities


def generate_altitude_profile(times, velocities, h_initial=80000.0):
    """
    @brief Generate altitude profile from velocity trajectory.

    @details
    Integrates the velocity profile to compute altitude assuming
    vertical descent (simplified model):
    $h(t) = h_0 - \\int_0^t V(\\tau) \\, d\\tau$

    The altitude is clamped to zero (ground level) if integration
    would result in negative values.

    @param times: Array of time points [s]
    @param velocities: Array of velocities at each time point [m/s]
    @param h_initial: Initial altitude at start of reentry [m]

    @return NumPy array of altitudes [m]

    @note This is a simplified 1D model. Real trajectories involve
          both horizontal and vertical velocity components.
    """
    dt = times[1] - times[0]
    altitudes = h_initial - np.cumsum(velocities) * dt

    altitudes = np.maximum(altitudes, 0)

    return altitudes


def atmospheric_density(h):
    """
    Densité atmosphérique en fonction de l'altitude (modèle exponentiel).

    Args:
        h: Altitude en mètres

    Returns:
        Densité en kg/m³
    """
    rho_0 = 1.225  # kg/m³ au niveau de la mer
    H = 8500.0     # Échelle de hauteur en mètres

    if h < 0:
        return rho_0
    return rho_0 * np.exp(-h / H)


def generate_coupled_reentry(h_initial=250000.0, V_initial=7000.0, gamma_initial=25.0,
                              duration=600.0, dt=1.0,
                              mass=20000.0, Cd=1.0, A=10.0):
    """
    Génère un profil de rentrée avec physique couplée.

    Intègre les équations du mouvement avec:
    - Traînée aérodynamique fonction de ρ(h) et V²
    - Gravité
    - Angle de trajectoire variable

    Args:
        h_initial: Altitude initiale [m]
        V_initial: Vitesse initiale [m/s]
        gamma_initial: Angle d'entrée initial [degrés]
        duration: Durée max simulation [s]
        dt: Pas de temps [s]
        mass: Masse du véhicule [kg]
        Cd: Coefficient de traînée
        A: Surface frontale [m²]

    Returns:
        Tuple (times, velocities, altitudes, heat_flux)
    """
    g = 9.81  # m/s²

    # Conditions initiales
    h = h_initial
    V = V_initial
    gamma = np.radians(gamma_initial)  # Angle en radians

    # Listes pour stocker les résultats
    times = [0.0]
    altitudes = [h]
    velocities = [V]
    heat_fluxes = [0.0]

    t = 0.0

    while t < duration and h > 0 and V > 10:
        # Densité atmosphérique à l'altitude actuelle
        rho = atmospheric_density(h)

        # Force de traînée
        F_drag = 0.5 * rho * V**2 * Cd * A

        # Accélération (traînée + gravité)
        a_drag = F_drag / mass
        a_gravity = g * np.sin(gamma)

        # Mise à jour vitesse (Euler explicite)
        dV = -(a_drag + a_gravity) * dt
        V_new = max(V + dV, 10.0)  # Vitesse minimum 10 m/s

        # Mise à jour altitude
        dh = -V * np.sin(gamma) * dt
        h_new = max(h + dh, 0.0)

        # L'angle s'aplatit progressivement (effet de portance simplifiée)
        # Plus on descend, plus l'angle diminue
        if h > 50000:
            gamma_deg = gamma_initial
        elif h > 20000:
            gamma_deg = gamma_initial * (0.3 + 0.7 * (h - 20000) / 30000)
        else:
            gamma_deg = gamma_initial * 0.3 * (h / 20000)

        gamma = np.radians(max(gamma_deg, 2.0))  # Angle minimum 2°

        # Flux thermique (Sutton-Graves simplifié)
        # q = C * sqrt(rho) * V³, en W/m²
        C_heat = 1.83e-4
        q = C_heat * np.sqrt(rho) * V**3

        # Mise à jour
        V = V_new
        h = h_new
        t += dt

        times.append(t)
        altitudes.append(h)
        velocities.append(V)
        heat_fluxes.append(q)

        # Arrêt si altitude atteinte
        if h <= 0:
            break

    return (np.array(times), np.array(velocities),
            np.array(altitudes), np.array(heat_fluxes))


def compute_dynamic_pressure(velocities, rho=0.02):
    """
    @brief Compute dynamic pressure from velocity profile.

    @details
    Calculates the dynamic pressure (stagnation pressure increase):
    $q = \\frac{1}{2} \\rho V^2$

    Dynamic pressure is a key parameter for:
    - Aerodynamic loads on structure
    - Heat flux estimation
    - Maximum Q (max-q) determination

    @param velocities: Array of velocities [m/s]
    @param rho: Air density [kg/m^3] (default: 0.02 at ~30 km altitude)

    @return NumPy array of dynamic pressures [Pa]
    """
    q = 0.5 * rho * velocities**2
    return q


def plot_reentry_profile(times, velocities, altitudes=None, save_file=None):
    """
    @brief Visualize the complete reentry trajectory profile.

    @details
    Creates a 4-panel figure showing:
    1. Velocity vs time: Deceleration profile
    2. Altitude vs time: Descent trajectory
    3. Deceleration vs time: G-loads experienced
    4. Altitude vs velocity: Phase space trajectory

    @param times: Array of time points [s]
    @param velocities: Array of velocities [m/s]
    @param altitudes: Array of altitudes [m] (computed if not provided)
    @param save_file: Path to save figure (optional)

    @example
    >>> times, velocities = generate_reentry_profile()
    >>> altitudes = generate_altitude_profile(times, velocities)
    >>> plot_reentry_profile(times, velocities, altitudes, "reentry.png")
    """
    if altitudes is None:
        altitudes = generate_altitude_profile(times, velocities)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax1 = axes[0, 0]
    ax1.plot(times, velocities, 'b-', linewidth=2)
    ax1.set_xlabel('Temps (s)')
    ax1.set_ylabel('Vitesse (m/s)')
    ax1.set_title('Profil de Vitesse')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.plot(times, altitudes / 1000, 'g-', linewidth=2)
    ax2.set_xlabel('Temps (s)')
    ax2.set_ylabel('Altitude (km)')
    ax2.set_title('Profil d\'Altitude')
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    deceleration = -np.gradient(velocities, times)
    ax3.plot(times, deceleration, 'r-', linewidth=2)
    ax3.set_xlabel('Temps (s)')
    ax3.set_ylabel('Deceleration (m/s^2)')
    ax3.set_title('Deceleration')
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    ax4.plot(velocities, altitudes / 1000, 'purple', linewidth=2)
    ax4.set_xlabel('Vitesse (m/s)')
    ax4.set_ylabel('Altitude (km)')
    ax4.set_title('Trajectoire de Rentree')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_file:
        plt.savefig(save_file, dpi=300)
        print(f"Profil de rentree sauvegarde: {save_file}")

    plt.show()


if __name__ == '__main__':
    print("=" * 70)
    print("GENERATION DU PROFIL DE RENTREE ATMOSPHERIQUE")
    print("=" * 70)

    times, velocities = generate_reentry_profile(
        V_initial=7000.0,
        V_final=500.0,
        duration=300.0,
        n_points=100,
        profile_type='exponential'
    )

    altitudes = generate_altitude_profile(times, velocities, h_initial=80000.0)

    print(f"\nProfil genere:")
    print(f"  - Duree:            {times[-1]:.1f} s")
    print(f"  - Vitesse initiale: {velocities[0]:.1f} m/s")
    print(f"  - Vitesse finale:   {velocities[-1]:.1f} m/s")
    print(f"  - Altitude init:    {altitudes[0]/1000:.1f} km")
    print(f"  - Altitude finale:  {altitudes[-1]/1000:.1f} km")

    deceleration_max = np.max(-np.gradient(velocities, times))
    print(f"  - Deceleration max: {deceleration_max:.1f} m/s^2 ({deceleration_max/9.81:.1f} g)")

    plot_reentry_profile(times, velocities, altitudes, save_file='reentry_profile_test.png')
