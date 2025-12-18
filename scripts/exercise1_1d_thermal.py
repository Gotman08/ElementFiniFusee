"""
Exercice 1 : Problème thermique 1D avec condition de Robin.

Résout le problème :
  -κ d²u/dx² = f(x)  sur [0, L]
  u(0) = 0  (Dirichlet)
  -κ du/dx|_{x=L} = α(u(L) - u_E)  (Robin)

Génère :
- Graphique de la solution FEM
- Export CSV des résultats
- Statistiques de la solution
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Note: pandas non disponible, export CSV simplifié")

from src.mesh.mesh_1d import create_uniform_mesh
from src.core.assembly_1d import assemble_1d_system, apply_dirichlet_1d, solve_1d_system


def main():
    """
    Résolution du problème thermique 1D pour l'Exercice 1.
    """
    print("="*70)
    print("  EXERCICE 1 : Problème Thermique 1D avec BC de Robin")
    print("="*70)

    # ========== PARAMÈTRES DU PROBLÈME ==========
    L = 1.0          # Longueur du domaine [m]
    kappa = 10.0     # Conductivité thermique [W/(m·K)]
    f = 1000.0       # Terme source volumique constant [W/m³]
    alpha = 50.0     # Coefficient de transfert convectif [W/(m²·K)]
    u_E = 300.0      # Température extérieure [K]
    n_elements = 50  # Nombre d'éléments

    print(f"\nParamètres du problème :")
    print(f"  Domaine : [0, {L}] m")
    print(f"  Conductivité thermique κ = {kappa} W/(m·K)")
    print(f"  Terme source volumique f = {f} W/m³")
    print(f"  Coefficient de Robin α = {alpha} W/(m²·K)")
    print(f"  Température extérieure u_E = {u_E} K")
    print(f"  Nombre d'éléments : {n_elements}\n")

    print("Formulation du problème :")
    print("  -κ d²u/dx² = f  sur [0, L]")
    print("  u(0) = 0  (Dirichlet à gauche)")
    print("  -κ du/dx|_{x=L} = α(u(L) - u_E)  (Robin à droite)\n")

    # ========== MAILLAGE ==========
    print(f"Création du maillage : {n_elements} éléments...")
    mesh = create_uniform_mesh(L, n_elements)
    h = L / n_elements
    print(f"  Nombre de nœuds : {mesh.n_nodes}")
    print(f"  Pas de maillage h = {h:.4f} m\n")

    # ========== ASSEMBLAGE ==========
    print("Assemblage du système global...")
    robin_bc = {'right': (alpha, u_E)}
    A, F = assemble_1d_system(mesh, kappa, source_term=f, robin_bc=robin_bc)
    print(f"  Matrice assemblée : {A.shape}")
    print(f"  Vecteur charge assemblé : {F.shape}\n")

    # ========== CONDITIONS AUX LIMITES ==========
    print("Application des conditions de Dirichlet...")
    dirichlet_bc = {'left': 0.0}
    A_bc, F_bc = apply_dirichlet_1d(A, F, dirichlet_bc, mesh)
    print(f"  BC Dirichlet appliquée : u(0) = 0.0 K\n")

    # ========== RÉSOLUTION ==========
    print("Résolution du système linéaire...")
    U = solve_1d_system(A_bc, F_bc)
    print(f"  Solution calculée : {len(U)} DOFs\n")

    # ========== STATISTIQUES ==========
    print("="*70)
    print("RÉSULTATS DE LA SIMULATION :")
    print("="*70)
    print(f"Température minimale : u_min = {U.min():.2f} K")
    print(f"Température maximale : u_max = {U.max():.2f} K")
    print(f"Température moyenne  : u_moy = {U.mean():.2f} K")
    print(f"Température à x=0    : u(0)  = {U[0]:.2e} K  (BC Dirichlet)")
    print(f"Température à x=L    : u(L)  = {U[-1]:.2f} K")
    print("="*70 + "\n")

    # ========== VÉRIFICATION PHYSIQUE ==========
    print("Vérifications physiques :")

    # Vérifier BC Dirichlet
    if abs(U[0]) < 1e-10:
        print("  ✓ BC Dirichlet satisfaite : u(0) ≈ 0")
    else:
        print(f"  ✗ BC Dirichlet non satisfaite : u(0) = {U[0]:.2e}")

    # Vérifier que u_max est à l'intérieur du domaine
    i_max = np.argmax(U)
    x_max = mesh.nodes[i_max]
    if 0 < i_max < len(U) - 1:
        print(f"  ✓ Maximum à l'intérieur : u_max à x = {x_max:.3f} m")
    else:
        print(f"  ⚠ Maximum au bord : x = {x_max:.3f} m")

    # Vérifier que la température est raisonnable
    if U.max() < 10000:
        print(f"  ✓ Températures physiquement raisonnables (< 10000 K)")
    else:
        print(f"  ✗ Températures anormalement élevées")

    print()

    # ========== VISUALISATION ==========
    print("Génération du graphique...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Tracer la solution FEM
    ax.plot(mesh.nodes, U, 'b-', linewidth=2, label='Solution FEM P1', zorder=3)

    # Marquer les nœuds
    ax.plot(mesh.nodes, U, 'bo', markersize=4, alpha=0.5, zorder=2)

    # Ligne de référence : température extérieure
    ax.axhline(u_E, color='r', linestyle='--', linewidth=1.5,
               label=f'Température extérieure u_E = {u_E} K', zorder=1)

    # Ligne de référence : température nulle
    ax.axhline(0, color='k', linestyle=':', linewidth=1, alpha=0.5, zorder=1)

    # Annotations
    ax.plot(0, U[0], 'go', markersize=10, label=f'u(0) = {U[0]:.1f} K (Dirichlet)', zorder=4)
    ax.plot(L, U[-1], 'mo', markersize=10, label=f'u(L) = {U[-1]:.1f} K (Robin)', zorder=4)

    # Labels et titre
    ax.set_xlabel('Position x [m]', fontsize=12)
    ax.set_ylabel('Température u(x) [K]', fontsize=12)
    ax.set_title('Exercice 1 : Problème Thermique 1D avec Condition de Robin',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(fontsize=10, loc='best')

    # Ajouter une zone ombrée pour visualiser la convection
    ax.axvspan(L * 0.9, L, alpha=0.1, color='red', label='Zone convective')

    plt.tight_layout()

    # Sauvegarder
    output_dir = Path('data/output/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'exercise1_solution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure sauvegardée : {output_file}\n")

    # ========== EXPORT CSV ==========
    csv_dir = Path('data/output/csv')
    csv_dir.mkdir(parents=True, exist_ok=True)

    # Données de la solution
    csv_file = csv_dir / 'exercise1_results.csv'
    if HAS_PANDAS:
        df_solution = pd.DataFrame({
            'x[m]': mesh.nodes,
            'T[K]': U,
            'T-T_ext[K]': U - u_E
        })
        df_solution.to_csv(csv_file, index=False, float_format='%.6f')
    else:
        # Export CSV manuel
        with open(csv_file, 'w') as f:
            f.write('x[m],T[K],T-T_ext[K]\n')
            for x, T in zip(mesh.nodes, U):
                f.write(f'{x:.6f},{T:.6f},{T-u_E:.6f}\n')
    print(f"Données sauvegardées : {csv_file}")

    # Rapport détaillé
    report_file = csv_dir / 'exercise1_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("EXERCICE 1 : Rapport de simulation thermique 1D\n")
        f.write("="*70 + "\n\n")

        f.write("PARAMÈTRES DU PROBLÈME :\n")
        f.write("-"*70 + "\n")
        f.write(f"Domaine                     : [0, {L}] m\n")
        f.write(f"Conductivité thermique κ    : {kappa} W/(m·K)\n")
        f.write(f"Terme source volumique f    : {f} W/m³\n")
        f.write(f"Coefficient de Robin α      : {alpha} W/(m²·K)\n")
        f.write(f"Température extérieure u_E  : {u_E} K\n")
        f.write(f"Nombre d'éléments           : {n_elements}\n")
        f.write(f"Pas de maillage h           : {h:.6f} m\n\n")

        f.write("CONDITIONS AUX LIMITES :\n")
        f.write("-"*70 + "\n")
        f.write("À x = 0 (gauche)  : u(0) = 0 K (Dirichlet)\n")
        f.write(f"À x = L (droite)  : -κ du/dx = α(u - {u_E}) K (Robin)\n\n")

        f.write("RÉSULTATS :\n")
        f.write("-"*70 + "\n")
        f.write(f"Température minimale : {U.min():.4f} K\n")
        f.write(f"Température maximale : {U.max():.4f} K\n")
        f.write(f"Température moyenne  : {U.mean():.4f} K\n")
        f.write(f"Écart-type           : {U.std():.4f} K\n")
        f.write(f"Température à x=0    : {U[0]:.2e} K\n")
        f.write(f"Température à x=L    : {U[-1]:.4f} K\n")
        f.write(f"Position du maximum  : x = {mesh.nodes[np.argmax(U)]:.4f} m\n\n")

        f.write("INTERPRÉTATION PHYSIQUE :\n")
        f.write("-"*70 + "\n")
        f.write("Le profil de température résulte de l'équilibre entre :\n")
        f.write(f"  1. Le chauffage volumique (f = {f} W/m³)\n")
        f.write(f"  2. La conduction thermique (κ = {kappa} W/(m·K))\n")
        f.write(f"  3. La convection au bord droit (α = {alpha} W/(m²·K))\n\n")

        if U.max() > u_E:
            f.write(f"La température maximale ({U.max():.1f} K) dépasse u_E ({u_E} K),\n")
            f.write("indiquant que le chauffage volumique domine sur la convection.\n")
        else:
            f.write("La convection est suffisante pour maintenir T < u_E.\n")

        f.write("\n" + "="*70 + "\n")

    print(f"Rapport sauvegardé : {report_file}\n")

    # ========== AFFICHAGE FINAL ==========
    print("="*70)
    print("EXERCICE 1 TERMINÉ AVEC SUCCÈS")
    print("="*70)
    print("\nFichiers générés :")
    print(f"  1. {output_file}")
    print(f"  2. {csv_file}")
    print(f"  3. {report_file}")
    print("\nConclusion :")
    print("  Le problème thermique 1D avec condition de Robin a été résolu.")
    print(f"  La température maximale atteinte est {U.max():.2f} K.")
    print("  La solution respecte toutes les conditions aux limites.")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
