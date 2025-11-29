"""
Module de generation de profil de rentree atmospherique
Modelise la deceleration d'une fusee lors de la rentree
"""
import numpy as np
import matplotlib.pyplot as plt

def generate_reentry_profile(V_initial=7000.0,
                             V_final=500.0,
                             duration=300.0,
                             n_points=100,
                             profile_type='exponential'):
    """
    Genere un profil de vitesse realiste pour la rentree atmospherique

    Args:
        V_initial: Vitesse initiale (m/s) - typiquement 7000 m/s
        V_final: Vitesse finale (m/s) - typiquement 500 m/s
        duration: Duree totale de la rentree (s)
        n_points: Nombre de points temporels
        profile_type: Type de deceleration
            - 'exponential': Freinage exponentiel (realiste)
            - 'linear': Deceleration lineaire (simplifie)
            - 'quadratic': Deceleration quadratique

    Returns:
        (times, velocities): Arrays numpy des temps et vitesses
    """
    times = np.linspace(0, duration, n_points)

    if profile_type == 'exponential':
        # Freinage exponentiel realiste
        # V(t) = V_f + (V_0 - V_f) * exp(-t/tau)
        tau = duration / 3.0  # Constante de temps
        velocities = V_final + (V_initial - V_final) * np.exp(-times / tau)

    elif profile_type == 'linear':
        # Deceleration lineaire
        velocities = V_initial - (V_initial - V_final) * (times / duration)

    elif profile_type == 'quadratic':
        # Deceleration quadratique (freinage plus fort au debut)
        normalized_t = times / duration
        velocities = V_initial - (V_initial - V_final) * normalized_t**2

    else:
        raise ValueError(f"Type de profil inconnu: {profile_type}")

    return times, velocities


def generate_altitude_profile(times, velocities, h_initial=80000.0):
    """
    Genere un profil d'altitude a partir du profil de vitesse

    Args:
        times: Array des temps (s)
        velocities: Array des vitesses (m/s)
        h_initial: Altitude initiale (m)

    Returns:
        altitudes: Array des altitudes (m)
    """
    # Integration de la vitesse (descente verticale simplifiee)
    # h(t) = h_0 - int_0^t V(tau) dtau
    dt = times[1] - times[0]
    altitudes = h_initial - np.cumsum(velocities) * dt

    # Limiter a l'altitude minimale (sol)
    altitudes = np.maximum(altitudes, 0)

    return altitudes


def compute_dynamic_pressure(velocities, rho=0.02):
    """
    Calcule la pression dynamique q = 0.5 * rho * V^2

    Args:
        velocities: Array des vitesses (m/s)
        rho: Densite de l'air (kg/m^3)

    Returns:
        q: Array des pressions dynamiques (Pa)
    """
    q = 0.5 * rho * velocities**2
    return q


def plot_reentry_profile(times, velocities, altitudes=None, save_file=None):
    """
    Trace le profil de rentree complet

    Args:
        times: Array des temps (s)
        velocities: Array des vitesses (m/s)
        altitudes: Array des altitudes (m) - optionnel
        save_file: Chemin pour sauvegarder la figure
    """
    if altitudes is None:
        # Creer un profil d'altitude simple
        altitudes = generate_altitude_profile(times, velocities)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Vitesse vs temps
    ax1 = axes[0, 0]
    ax1.plot(times, velocities, 'b-', linewidth=2)
    ax1.set_xlabel('Temps (s)')
    ax1.set_ylabel('Vitesse (m/s)')
    ax1.set_title('Profil de Vitesse')
    ax1.grid(True, alpha=0.3)

    # Altitude vs temps
    ax2 = axes[0, 1]
    ax2.plot(times, altitudes / 1000, 'g-', linewidth=2)
    ax2.set_xlabel('Temps (s)')
    ax2.set_ylabel('Altitude (km)')
    ax2.set_title('Profil d\'Altitude')
    ax2.grid(True, alpha=0.3)

    # Deceleration
    ax3 = axes[1, 0]
    deceleration = -np.gradient(velocities, times)
    ax3.plot(times, deceleration, 'r-', linewidth=2)
    ax3.set_xlabel('Temps (s)')
    ax3.set_ylabel('Deceleration (m/s^2)')
    ax3.set_title('Deceleration')
    ax3.grid(True, alpha=0.3)

    # Trajectoire altitude-vitesse
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
    # Test du module
    print("=" * 70)
    print("GENERATION DU PROFIL DE RENTREE ATMOSPHERIQUE")
    print("=" * 70)

    # Generer le profil
    times, velocities = generate_reentry_profile(
        V_initial=7000.0,
        V_final=500.0,
        duration=300.0,
        n_points=100,
        profile_type='exponential'
    )

    # Generer l'altitude
    altitudes = generate_altitude_profile(times, velocities, h_initial=80000.0)

    print(f"\nProfil genere:")
    print(f"  - Duree:            {times[-1]:.1f} s")
    print(f"  - Vitesse initiale: {velocities[0]:.1f} m/s")
    print(f"  - Vitesse finale:   {velocities[-1]:.1f} m/s")
    print(f"  - Altitude init:    {altitudes[0]/1000:.1f} km")
    print(f"  - Altitude finale:  {altitudes[-1]/1000:.1f} km")

    # Calculer quelques statistiques
    deceleration_max = np.max(-np.gradient(velocities, times))
    print(f"  - Deceleration max: {deceleration_max:.1f} m/s^2 ({deceleration_max/9.81:.1f} g)")

    # Tracer le profil
    plot_reentry_profile(times, velocities, altitudes, save_file='reentry_profile_test.png')
