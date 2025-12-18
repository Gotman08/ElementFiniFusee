#!/usr/bin/env python3
"""
Post-traitement des résultats de la simulation thermique du bouclier.

Ce script analyse les fichiers CSV générés par FreeFem++ et produit:
- Des graphiques de T(0,0) vs Vitesse pour chaque configuration
- Un tableau de synthèse Vmax par matériau et épaisseur
- Une carte de température (heatmap) du champ thermique
- Un abaque graphique Vmax(Kb, e)

Auteur: Master CHPS - Université de Reims
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Configuration matplotlib pour des graphiques de qualité publication
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Chemins des fichiers
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "output" / "csv"
FIGURES_DIR = PROJECT_ROOT / "data" / "output" / "figures"

# Créer les répertoires si nécessaire
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_detailed_results():
    """Charge les résultats détaillés de la simulation."""
    filepath = DATA_DIR / "shield_results_detail.csv"
    if not filepath.exists():
        print(f"Erreur: Fichier non trouvé: {filepath}")
        print("Exécutez d'abord FreeFem++ scripts/shield_simulation.edp")
        return None
    return pd.read_csv(filepath)


def load_synthesis_results():
    """Charge le tableau de synthèse Vmax."""
    filepath = DATA_DIR / "shield_Vmax_synthesis.csv"
    if not filepath.exists():
        print(f"Erreur: Fichier non trouvé: {filepath}")
        return None
    return pd.read_csv(filepath)


def load_temperature_field():
    """Charge le champ de température pour visualisation."""
    filepath = DATA_DIR / "temperature_field_PICA_Mach5.csv"
    if not filepath.exists():
        print(f"Avertissement: Champ de température non trouvé: {filepath}")
        return None
    return pd.read_csv(filepath)


def plot_temperature_vs_velocity(df):
    """
    Trace T(0,0) en fonction de la vitesse pour chaque configuration.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    materials = df['Material'].unique()
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(df['Thickness_m'].unique())))

    # Limites de température admissibles
    T_min_C = -150
    T_max_C = 150

    for idx, mat in enumerate(materials):
        ax = axes[idx]
        df_mat = df[df['Material'] == mat]

        thicknesses = sorted(df_mat['Thickness_m'].unique())

        for i, e in enumerate(thicknesses):
            df_e = df_mat[df_mat['Thickness_m'] == e]
            ax.plot(df_e['Mach'], df_e['T_origin_C'],
                   'o-', color=colors[i], linewidth=2, markersize=4,
                   label=f'e = {e*1000:.0f} mm')

        # Zone admissible
        ax.axhspan(T_min_C, T_max_C, alpha=0.2, color='green', label='Zone admissible')
        ax.axhline(y=T_min_C, color='red', linestyle='--', linewidth=1)
        ax.axhline(y=T_max_C, color='red', linestyle='--', linewidth=1)

        ax.set_xlabel('Nombre de Mach')
        ax.set_ylabel('T(0,0) [°C]')
        ax.set_title(f'{mat}')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 26])

    plt.suptitle('Température au point critique en fonction de la vitesse', fontsize=16, y=1.02)
    plt.tight_layout()

    filepath = FIGURES_DIR / "T_vs_velocity_all_materials.png"
    plt.savefig(filepath)
    print(f"Graphique sauvegardé: {filepath}")
    plt.close()


def plot_vmax_synthesis(df_synth):
    """
    Crée un graphique en barres groupées pour Vmax par configuration.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    materials = df_synth['Material'].unique()
    thicknesses = sorted(df_synth['Thickness_m'].unique())

    x = np.arange(len(thicknesses))
    width = 0.25

    colors = ['#2ecc71', '#3498db', '#e74c3c']  # Vert, bleu, rouge

    for i, mat in enumerate(materials):
        df_mat = df_synth[df_synth['Material'] == mat]
        vmax_values = [df_mat[df_mat['Thickness_m'] == e]['Vmax_Mach'].values[0]
                       if len(df_mat[df_mat['Thickness_m'] == e]) > 0 else 0
                       for e in thicknesses]

        bars = ax.bar(x + i*width, vmax_values, width, label=mat, color=colors[i])

        # Ajouter les valeurs sur les barres
        for bar, val in zip(bars, vmax_values):
            if val > 0:
                ax.annotate(f'{val}',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Épaisseur du bouclier [mm]')
    ax.set_ylabel('Vitesse maximale admissible [Mach]')
    ax.set_title('Abaque: Vitesse maximale admissible par configuration')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'{e*1000:.0f}' for e in thicknesses])
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    filepath = FIGURES_DIR / "Vmax_abaque.png"
    plt.savefig(filepath)
    print(f"Graphique sauvegardé: {filepath}")
    plt.close()


def plot_temperature_field(df_temp):
    """
    Trace le champ de température 2D.
    """
    if df_temp is None:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Créer un scatter plot avec les triangles
    scatter = ax.tricontourf(df_temp['x'], df_temp['y'], df_temp['T_C'],
                              levels=50, cmap='hot')

    cbar = plt.colorbar(scatter, ax=ax, label='Température [°C]')

    # Marquer le point critique
    ax.plot(0, 0, 'ko', markersize=10, markerfacecolor='cyan',
            markeredgecolor='black', markeredgewidth=2, label='Point critique (0,0)')

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Champ de température - PICA, e=0.01m, Mach 5')
    ax.set_aspect('equal')
    ax.legend()

    filepath = FIGURES_DIR / "temperature_field_2D.png"
    plt.savefig(filepath)
    print(f"Graphique sauvegardé: {filepath}")
    plt.close()


def create_synthesis_table(df_synth):
    """
    Crée et affiche le tableau de synthèse formaté.
    """
    print("\n" + "="*70)
    print("TABLEAU DE SYNTHESE - VITESSE MAXIMALE ADMISSIBLE")
    print("="*70)
    print(f"Contrainte: T(0,0) ∈ [-150°C, +150°C]")
    print("-"*70)
    print(f"{'Matériau':<18} {'Kb [W/(m·K)]':>12} {'e [mm]':>10} {'Vmax [m/s]':>12} {'Vmax [Mach]':>12}")
    print("-"*70)

    for _, row in df_synth.iterrows():
        print(f"{row['Material']:<18} {row['Kb_W_mK']:>12.1f} {row['Thickness_m']*1000:>10.0f} "
              f"{row['Vmax_m_s']:>12.0f} {row['Vmax_Mach']:>12}")

    print("="*70)

    # Sauvegarder aussi en format texte
    filepath = DATA_DIR / "synthesis_table_formatted.txt"
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("TABLEAU DE SYNTHESE - VITESSE MAXIMALE ADMISSIBLE\n")
        f.write("="*70 + "\n")
        f.write("Contrainte: T(0,0) ∈ [-150°C, +150°C]\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Matériau':<18} {'Kb [W/(m·K)]':>12} {'e [mm]':>10} {'Vmax [m/s]':>12} {'Vmax [Mach]':>12}\n")
        f.write("-"*70 + "\n")
        for _, row in df_synth.iterrows():
            f.write(f"{row['Material']:<18} {row['Kb_W_mK']:>12.1f} {row['Thickness_m']*1000:>10.0f} "
                   f"{row['Vmax_m_s']:>12.0f} {row['Vmax_Mach']:>12}\n")
        f.write("="*70 + "\n")

    print(f"\nTableau sauvegardé: {filepath}")


def plot_heatmap_vmax(df_synth):
    """
    Crée une heatmap de Vmax en fonction de Kb et e.
    """
    # Pivot pour créer la matrice
    pivot = df_synth.pivot(index='Material', columns='Thickness_m', values='Vmax_Mach')

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto')

    # Labels
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([f'{e*1000:.0f}' for e in pivot.columns])
    ax.set_yticklabels(pivot.index)

    ax.set_xlabel('Épaisseur [mm]')
    ax.set_ylabel('Matériau')
    ax.set_title('Heatmap: Vitesse maximale admissible [Mach]')

    # Ajouter les valeurs dans les cellules
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = 'white' if val < pivot.values.max()/2 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center', color=color, fontsize=14, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Vmax [Mach]')

    filepath = FIGURES_DIR / "Vmax_heatmap.png"
    plt.savefig(filepath)
    print(f"Graphique sauvegardé: {filepath}")
    plt.close()


def main():
    """Fonction principale de post-traitement."""
    print("="*70)
    print("POST-TRAITEMENT DES RESULTATS - BOUCLIER THERMIQUE")
    print("="*70)

    # Charger les données
    df_detail = load_detailed_results()
    df_synth = load_synthesis_results()
    df_temp = load_temperature_field()

    if df_detail is None or df_synth is None:
        print("\nErreur: Données manquantes. Exécutez d'abord la simulation FreeFem++.")
        print("Commande: FreeFem++ scripts/shield_simulation.edp")
        return 1

    print(f"\nDonnées chargées:")
    print(f"  - Résultats détaillés: {len(df_detail)} lignes")
    print(f"  - Synthèse Vmax: {len(df_synth)} configurations")

    # Générer les graphiques
    print("\nGénération des graphiques...")

    plot_temperature_vs_velocity(df_detail)
    plot_vmax_synthesis(df_synth)
    plot_heatmap_vmax(df_synth)

    if df_temp is not None:
        plot_temperature_field(df_temp)

    # Afficher le tableau de synthèse
    create_synthesis_table(df_synth)

    print("\n" + "="*70)
    print("POST-TRAITEMENT TERMINE")
    print("="*70)
    print(f"\nGraphiques générés dans: {FIGURES_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
