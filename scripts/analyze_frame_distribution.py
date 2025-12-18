"""
Script d'analyse de la distribution des frames
par rapport aux phases de chauffage atmosphérique.

Ce script vérifie que l'échantillonnage adaptatif fonctionne correctement
en analysant quand le chauffage commence et où les frames sont concentrées.
"""
import numpy as np
from src.physics.reentry_profile import generate_coupled_reentry, adaptive_frame_sampling


def analyze_heating_phases(times, velocities, altitudes, heat_flux):
    """
    Analyse les phases de chauffage et identifie les moments critiques.
    """
    print("\n" + "="*80)
    print("ANALYSE DES PHASES DE CHAUFFAGE")
    print("="*80)

    # Identifier quand le chauffage commence (flux > 1% du max)
    heat_threshold = np.max(heat_flux) * 0.01
    heating_starts_idx = None
    for i, q in enumerate(heat_flux):
        if q > heat_threshold:
            heating_starts_idx = i
            break

    if heating_starts_idx is not None:
        print(f"\n1. DEBUT DU CHAUFFAGE (flux > 1% max):")
        print(f"   - Index: {heating_starts_idx}")
        print(f"   - Temps: {times[heating_starts_idx]:.1f} s")
        print(f"   - Vitesse: {velocities[heating_starts_idx]:.0f} m/s")
        print(f"   - Altitude: {altitudes[heating_starts_idx]/1000:.1f} km")
        print(f"   - Flux thermique: {heat_flux[heating_starts_idx]/1e6:.3f} MW/m²")

    # Identifier le pic de chauffage
    peak_idx = np.argmax(heat_flux)
    print(f"\n2. PIC DE CHAUFFAGE:")
    print(f"   - Index: {peak_idx}")
    print(f"   - Temps: {times[peak_idx]:.1f} s")
    print(f"   - Vitesse: {velocities[peak_idx]:.0f} m/s")
    print(f"   - Altitude: {altitudes[peak_idx]/1000:.1f} km")
    print(f"   - Flux thermique MAX: {heat_flux[peak_idx]/1e6:.3f} MW/m²")

    # Identifier quand le chauffage devient négligeable (flux < 1% du max)
    heating_ends_idx = None
    for i in range(peak_idx, len(heat_flux)):
        if heat_flux[i] < heat_threshold:
            heating_ends_idx = i
            break

    if heating_ends_idx is not None:
        print(f"\n3. FIN DU CHAUFFAGE SIGNIFICATIF (flux < 1% max):")
        print(f"   - Index: {heating_ends_idx}")
        print(f"   - Temps: {times[heating_ends_idx]:.1f} s")
        print(f"   - Vitesse: {velocities[heating_ends_idx]:.0f} m/s")
        print(f"   - Altitude: {altitudes[heating_ends_idx]/1000:.1f} km")
        print(f"   - Flux thermique: {heat_flux[heating_ends_idx]/1e6:.3f} MW/m²")

    # Déploiement du parachute (V passe sous 100 m/s)
    parachute_idx = None
    for i in range(len(velocities) - 1):
        if velocities[i] > 100 and velocities[i+1] <= 100:
            parachute_idx = i
            break

    if parachute_idx is not None:
        print(f"\n4. DEPLOIEMENT PARACHUTE (V < 100 m/s):")
        print(f"   - Index: {parachute_idx}")
        print(f"   - Temps: {times[parachute_idx]:.1f} s")
        print(f"   - Vitesse: {velocities[parachute_idx]:.0f} m/s")
        print(f"   - Altitude: {altitudes[parachute_idx]/1000:.1f} km")

    # Résumé des phases
    print(f"\n" + "="*80)
    print("RESUME DES PHASES PHYSIQUES")
    print("="*80)

    if heating_starts_idx and heating_ends_idx:
        heating_duration = times[heating_ends_idx] - times[heating_starts_idx]
        heating_points = heating_ends_idx - heating_starts_idx
        print(f"\nPhase de chauffage actif:")
        print(f"  - Durée: {heating_duration:.1f} s ({times[heating_starts_idx]:.1f}s -> {times[heating_ends_idx]:.1f}s)")
        print(f"  - Points de trajectoire: {heating_points} / {len(times)} ({100*heating_points/len(times):.1f}%)")
        print(f"  - Vitesse: {velocities[heating_starts_idx]:.0f} -> {velocities[heating_ends_idx]:.0f} m/s")

    if heating_ends_idx and parachute_idx:
        descent_duration = times[parachute_idx] - times[heating_ends_idx]
        descent_points = parachute_idx - heating_ends_idx
        print(f"\nPhase de descente (après chauffage, avant parachute):")
        print(f"  - Durée: {descent_duration:.1f} s")
        print(f"  - Points: {descent_points} / {len(times)} ({100*descent_points/len(times):.1f}%)")

    if parachute_idx:
        parachute_duration = times[-1] - times[parachute_idx]
        parachute_points = len(times) - parachute_idx
        print(f"\nPhase parachute (V constant ~15 m/s):")
        print(f"  - Durée: {parachute_duration:.1f} s")
        print(f"  - Points: {parachute_points} / {len(times)} ({100*parachute_points/len(times):.1f}%)")

    return heating_starts_idx, peak_idx, heating_ends_idx, parachute_idx


def analyze_frame_distribution(times_orig, times_sampled, velocities_orig, heating_starts_idx, peak_idx,
                               heating_ends_idx, parachute_idx):
    """
    Analyse la distribution des frames après échantillonnage adaptatif.
    """
    print("\n" + "="*80)
    print("ANALYSE DE LA DISTRIBUTION DES FRAMES")
    print("="*80)

    # Mapper les indices originaux aux indices échantillonnés
    # Pour chaque frame échantillonnée, trouver son index dans la trajectoire originale
    sampled_indices = []
    for t_sample in times_sampled:
        idx = np.argmin(np.abs(times_orig - t_sample))
        sampled_indices.append(idx)

    sampled_indices = np.array(sampled_indices)

    print(f"\nNombre total de frames: {len(times_sampled)}")
    print(f"Points de trajectoire originaux: {len(times_orig)}")
    print(f"Facteur de réduction: {len(times_orig) / len(times_sampled):.2f}x")

    # Compter les frames par phase
    if heating_starts_idx and heating_ends_idx:
        frames_heating = np.sum((sampled_indices >= heating_starts_idx) &
                               (sampled_indices < heating_ends_idx))
        print(f"\n1. PHASE DE CHAUFFAGE ACTIF:")
        print(f"   - Frames: {frames_heating} / {len(times_sampled)} ({100*frames_heating/len(times_sampled):.1f}%)")
        print(f"   - Indices: {heating_starts_idx} -> {heating_ends_idx}")

        # Densité de frames
        heating_time_range = times_orig[heating_ends_idx] - times_orig[heating_starts_idx]
        if heating_time_range > 0:
            frame_density = frames_heating / heating_time_range
            print(f"   - Densité: {frame_density:.3f} frames/s")

    if heating_ends_idx and parachute_idx:
        frames_descent = np.sum((sampled_indices >= heating_ends_idx) &
                               (sampled_indices < parachute_idx))
        print(f"\n2. PHASE DE DESCENTE (après chauffage):")
        print(f"   - Frames: {frames_descent} / {len(times_sampled)} ({100*frames_descent/len(times_sampled):.1f}%)")

        descent_time_range = times_orig[parachute_idx] - times_orig[heating_ends_idx]
        if descent_time_range > 0:
            frame_density = frames_descent / descent_time_range
            print(f"   - Densité: {frame_density:.3f} frames/s")

    if parachute_idx:
        frames_parachute = np.sum(sampled_indices >= parachute_idx)
        print(f"\n3. PHASE PARACHUTE:")
        print(f"   - Frames: {frames_parachute} / {len(times_sampled)} ({100*frames_parachute/len(times_sampled):.1f}%)")

        parachute_time_range = times_orig[-1] - times_orig[parachute_idx]
        if parachute_time_range > 0:
            frame_density = frames_parachute / parachute_time_range
            print(f"   - Densité: {frame_density:.3f} frames/s")

    # Analyser les intervalles entre frames
    print(f"\n" + "="*80)
    print("INTERVALLES ENTRE FRAMES")
    print("="*80)

    frame_intervals = np.diff(times_sampled)
    print(f"\nStatistiques des intervalles de temps:")
    print(f"  - Minimum: {np.min(frame_intervals):.2f} s")
    print(f"  - Maximum: {np.max(frame_intervals):.2f} s")
    print(f"  - Moyenne: {np.mean(frame_intervals):.2f} s")
    print(f"  - Médiane: {np.median(frame_intervals):.2f} s")

    # Identifier les régions avec intervalles les plus courts (plus de frames)
    if heating_starts_idx and heating_ends_idx:
        heating_frames_mask = (sampled_indices >= heating_starts_idx) & \
                             (sampled_indices < heating_ends_idx)
        if np.sum(heating_frames_mask) > 1:
            heating_intervals = np.diff(times_sampled[heating_frames_mask])
            print(f"\nDans la phase de chauffage:")
            print(f"  - Intervalle moyen: {np.mean(heating_intervals):.2f} s")
            print(f"  - Intervalle minimum: {np.min(heating_intervals):.2f} s")

    if parachute_idx:
        parachute_frames_mask = sampled_indices >= parachute_idx
        if np.sum(parachute_frames_mask) > 1:
            parachute_intervals = np.diff(times_sampled[parachute_frames_mask])
            print(f"\nDans la phase parachute:")
            print(f"  - Intervalle moyen: {np.mean(parachute_intervals):.2f} s")
            print(f"  - Intervalle minimum: {np.min(parachute_intervals):.2f} s")

    # Analyse détaillée par plage de vitesse
    print(f"\n" + "="*80)
    print("DISTRIBUTION PAR PLAGE DE VITESSE")
    print("="*80)

    velocity_ranges = [
        ("V > 6000 m/s (Phase spatiale)", 6000, np.inf),
        ("3000 < V < 6000 m/s (Entrée atmosphérique)", 3000, 6000),
        ("500 < V < 3000 m/s (Chauffage maximal)", 500, 3000),
        ("100 < V < 500 m/s (Descente initiale)", 100, 500),
        ("V < 100 m/s (Parachute)", 0, 100)
    ]

    for label, v_min, v_max in velocity_ranges:
        # Frames dans cette plage
        mask = np.array([v_min <= velocities_orig[idx] < v_max for idx in sampled_indices])
        frames_in_range = np.sum(mask)

        # Points originaux dans cette plage
        points_in_range = np.sum((velocities_orig >= v_min) & (velocities_orig < v_max))

        if points_in_range > 0:
            sampling_rate = frames_in_range / points_in_range
            print(f"\n{label}:")
            print(f"  - Frames: {frames_in_range} / {len(times_sampled)} ({100*frames_in_range/len(times_sampled):.1f}%)")
            print(f"  - Points originaux: {points_in_range} / {len(times_orig)} ({100*points_in_range/len(times_orig):.1f}%)")
            print(f"  - Taux d'échantillonnage: {100*sampling_rate:.1f}%")


def main():
    print("\n" + "="*80)
    print("VALIDATION DE L'ECHANTILLONNAGE ADAPTATIF")
    print("Vérification de la correspondance entre chauffage et frames")
    print("="*80)

    # Paramètres identiques à demo_reentry.py
    h_initial = 250000.0
    V_initial = 7000.0
    gamma_initial = 25.0
    duration = 600.0

    print("\nGénération du profil de rentrée...")
    times_orig, velocities_orig, altitudes_orig, heat_flux_orig = generate_coupled_reentry(
        h_initial=h_initial,
        V_initial=V_initial,
        gamma_initial=gamma_initial,
        duration=duration,
        dt=3.0,
        mass=20000.0,
        Cd=1.0,
        A=10.0
    )

    print(f"[OK] Profil généré: {len(times_orig)} points de trajectoire")

    # Analyser les phases physiques AVANT échantillonnage
    heating_starts_idx, peak_idx, heating_ends_idx, parachute_idx = \
        analyze_heating_phases(times_orig, velocities_orig, altitudes_orig, heat_flux_orig)

    # Appliquer l'échantillonnage adaptatif
    print("\n" + "="*80)
    print("APPLICATION DE L'ECHANTILLONNAGE ADAPTATIF")
    print("="*80)

    times_sampled, velocities_sampled, altitudes_sampled, heat_flux_sampled = \
        adaptive_frame_sampling(times_orig, velocities_orig, altitudes_orig,
                              heat_flux_orig, max_frames=150)

    print(f"\n[OK] Échantillonnage terminé: {len(times_sampled)} frames")

    # Analyser la distribution des frames
    analyze_frame_distribution(times_orig, times_sampled, velocities_orig, heating_starts_idx,
                              peak_idx, heating_ends_idx, parachute_idx)

    # Conclusion
    print("\n" + "="*80)
    print("CONCLUSION - VALIDATION DE LA CORRESPONDANCE")
    print("="*80)

    # Calculer le ratio attendu vs obtenu pour la phase de chauffage
    if heating_starts_idx and heating_ends_idx and parachute_idx:
        # Proportion du temps de chauffage
        heating_time = times_orig[heating_ends_idx] - times_orig[heating_starts_idx]
        total_time = times_orig[-1] - times_orig[0]
        heating_time_ratio = heating_time / total_time

        # Proportion des frames dans la phase de chauffage
        sampled_indices = []
        for t_sample in times_sampled:
            idx = np.argmin(np.abs(times_orig - t_sample))
            sampled_indices.append(idx)
        sampled_indices = np.array(sampled_indices)

        frames_heating = np.sum((sampled_indices >= heating_starts_idx) &
                               (sampled_indices < heating_ends_idx))
        heating_frame_ratio = frames_heating / len(times_sampled)

        print(f"\nPhase de chauffage:")
        print(f"  - Représente {100*heating_time_ratio:.1f}% du temps total")
        print(f"  - Contient {100*heating_frame_ratio:.1f}% des frames")
        print(f"  - Enrichissement: {heating_frame_ratio/heating_time_ratio:.2f}x")

        if heating_frame_ratio > heating_time_ratio:
            print(f"\n[OK] VALIDATION: La phase de chauffage a {heating_frame_ratio/heating_time_ratio:.2f}x plus de frames")
            print(f"     que sa proportion temporelle (echantillonnage adaptatif fonctionne)")
        else:
            print(f"\n[ATTENTION]: La phase de chauffage n'a PAS plus de frames que sa proportion temporelle")

        # Même chose pour la phase parachute
        parachute_time = times_orig[-1] - times_orig[parachute_idx]
        parachute_time_ratio = parachute_time / total_time

        frames_parachute = np.sum(sampled_indices >= parachute_idx)
        parachute_frame_ratio = frames_parachute / len(times_sampled)

        print(f"\nPhase parachute:")
        print(f"  - Represente {100*parachute_time_ratio:.1f}% du temps total")
        print(f"  - Contient {100*parachute_frame_ratio:.1f}% des frames")
        print(f"  - Reduction: {parachute_frame_ratio/parachute_time_ratio:.2f}x")

        if parachute_frame_ratio < parachute_time_ratio:
            print(f"\n[OK] VALIDATION: La phase parachute a {parachute_time_ratio/parachute_frame_ratio:.2f}x moins de frames")
            print(f"     que sa proportion temporelle (optimisation reussie)")
        else:
            print(f"\n[ATTENTION]: La phase parachute a trop de frames par rapport au temps")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
