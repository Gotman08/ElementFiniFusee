"""
Module d'étude paramétrique en vitesse
Calcule α(V) et u_E(V) selon des modèles aérothermiques
Résout le problème thermique pour chaque vitesse
"""
import numpy as np
from typing import Tuple, List
from mesh_reader import Mesh, read_gmsh_mesh, create_node_mapping
from assembly import assemble_global_system
from boundary_conditions import apply_dirichlet_conditions
from solver import solve_linear_system

# ============================================================================
# CONSTANTES PHYSIQUES
# ============================================================================

# Propriétés de l'air à haute altitude
RHO_inf = 0.02          # Densité (kg/m³) à ~30 km d'altitude
T_inf = 230.0           # Température ambiante (K)
MU_inf = 1.5e-5         # Viscosité dynamique (Pa·s)
k_air = 0.02            # Conductivité thermique air (W/m·K)
Pr = 0.71               # Nombre de Prandtl
c_p = 1005.0            # Chaleur spécifique (J/kg·K)
gamma = 1.4             # Rapport des chaleurs spécifiques

# Facteur de récupération (écoulement turbulent)
r_recovery = 0.89

# Propriétés du matériau (alliage aluminium)
KAPPA_material = 160.0  # Conductivité thermique (W/m·K)

# Géométrie de référence
L_ref = 0.5             # Longueur caractéristique (m)


# ============================================================================
# MODÈLES AÉROTHERMIQUES
# ============================================================================

def compute_reynolds_number(V: float, rho: float, mu: float, L: float) -> float:
    """
    Calcule le nombre de Reynolds: Re = ρ V L / μ

    Args:
        V: Vitesse (m/s)
        rho: Densité (kg/m³)
        mu: Viscosité dynamique (Pa·s)
        L: Longueur caractéristique (m)

    Returns:
        Re: Nombre de Reynolds
    """
    Re = (rho * V * L) / mu
    return Re


def compute_stagnation_temperature(V: float, T_inf: float, gamma: float, c_p: float) -> float:
    """
    Calcule la température de stagnation totale (isentropique)
    T_0 = T_inf * (1 + (γ-1)/2 * M²)
    avec M = V / a_inf (nombre de Mach)

    Args:
        V: Vitesse (m/s)
        T_inf: Température ambiante (K)
        gamma: Rapport des chaleurs spécifiques
        c_p: Chaleur spécifique (J/kg·K)

    Returns:
        T_0: Température de stagnation (K)
    """
    # Vitesse du son: a = sqrt(γ R T) = sqrt(γ (c_p - c_v) T)
    # Pour gaz parfait: R = c_p - c_v = c_p (γ-1)/γ
    a_inf = np.sqrt(gamma * (c_p * (gamma - 1) / gamma) * T_inf)
    M = V / a_inf  # Nombre de Mach

    T_0 = T_inf * (1 + (gamma - 1) / 2 * M**2)
    return T_0


def compute_recovery_temperature(V: float, T_inf: float, r: float, gamma: float, c_p: float) -> float:
    """
    Calcule la température de récupération (adiabatique wall temperature)
    T_aw = T_inf + r * V² / (2 * c_p)

    Approximation plus simple que la température de stagnation,
    valide pour couche limite turbulente.

    Args:
        V: Vitesse (m/s)
        T_inf: Température ambiante (K)
        r: Facteur de récupération (0.89 turbulent, 0.85 laminaire)
        gamma: Rapport des chaleurs spécifiques
        c_p: Chaleur spécifique (J/kg·K)

    Returns:
        T_aw: Température de récupération (K)
    """
    T_aw = T_inf + r * V**2 / (2 * c_p)
    return T_aw


def compute_heat_transfer_coefficient(V: float, rho: float, mu: float, k: float,
                                      L: float, Pr: float) -> float:
    """
    Calcule le coefficient de transfert thermique par convection α
    Utilise une corrélation de type Nusselt:
        - Laminaire: Nu = 0.664 * Re^0.5 * Pr^(1/3)
        - Turbulent: Nu = 0.037 * Re^0.8 * Pr^(1/3)

    On utilise le régime turbulent pour Re > 5e5

    Args:
        V: Vitesse (m/s)
        rho: Densité (kg/m³)
        mu: Viscosité dynamique (Pa·s)
        k: Conductivité thermique du fluide (W/m·K)
        L: Longueur caractéristique (m)
        Pr: Nombre de Prandtl

    Returns:
        alpha: Coefficient de convection (W/m²·K)
    """
    Re = compute_reynolds_number(V, rho, mu, L)

    # Nombre de Nusselt (régime turbulent)
    if Re > 5e5:
        Nu = 0.037 * Re**0.8 * Pr**(1.0/3.0)
    else:
        Nu = 0.664 * Re**0.5 * Pr**(1.0/3.0)

    # Coefficient de convection: α = Nu * k / L
    alpha = Nu * k / L

    return alpha


def compute_aerothermal_parameters(V: float) -> Tuple[float, float]:
    """
    Calcule les paramètres aérothermiques α(V) et u_E(V)

    Args:
        V: Vitesse du lanceur (m/s)

    Returns:
        (alpha, u_E): Coefficient de convection (W/m²·K) et température extérieure (K)
    """
    # Température extérieure = température de récupération
    u_E = compute_recovery_temperature(V, T_inf, r_recovery, gamma, c_p)

    # Coefficient de convection
    alpha = compute_heat_transfer_coefficient(V, RHO_inf, MU_inf, k_air, L_ref, Pr)

    return alpha, u_E


# ============================================================================
# BOUCLE D'ÉTUDE PARAMÉTRIQUE
# ============================================================================

def parametric_velocity_study(mesh_file: str,
                              velocity_range: np.ndarray,
                              base_temperature: float = 300.0) -> Tuple[List[float], List[float], List[np.ndarray]]:
    """
    Réalise l'étude paramétrique en vitesse

    Pour chaque vitesse V:
        1. Calculer α(V) et u_E(V)
        2. Assembler le système avec ces paramètres
        3. Appliquer les conditions aux limites
        4. Résoudre
        5. Extraire T_max

    Args:
        mesh_file: Chemin vers le fichier .msh
        velocity_range: Array de vitesses à tester (m/s)
        base_temperature: Température imposée à la base (K)

    Returns:
        (velocities, T_max_list, solutions):
            - velocities: Liste des vitesses testées
            - T_max_list: Liste des températures maximales
            - solutions: Liste des champs de température complets
    """
    # Lire le maillage (une seule fois)
    print("=" * 70)
    print("ÉTUDE PARAMÉTRIQUE EN VITESSE")
    print("=" * 70)
    print(f"Lecture du maillage: {mesh_file}")

    mesh = read_gmsh_mesh(mesh_file)
    node_to_dof, num_dofs = create_node_mapping(mesh)

    print(f"Maillage chargé: {len(mesh.nodes)} noeuds, {num_dofs} DOFs")
    print(f"Plage de vitesses: {velocity_range[0]:.0f} - {velocity_range[-1]:.0f} m/s")
    print(f"Nombre de cas: {len(velocity_range)}")
    print("=" * 70)

    velocities = []
    T_max_list = []
    solutions = []

    for idx, V in enumerate(velocity_range):
        print(f"\n[Cas {idx+1}/{len(velocity_range)}] Vitesse V = {V:.0f} m/s")
        print("-" * 70)

        # Calculer les parametres aerothermiques
        alpha, u_E = compute_aerothermal_parameters(V)
        print(f"  alpha(V) = {alpha:.2f} W/m^2.K")
        print(f"  u_E(V) = {u_E:.2f} K")

        # Définir les conditions aux limites Robin (sur Gamma_F)
        # Physical ID 1 = Surface extérieure
        robin_boundaries = {
            1: (alpha, u_E)
        }

        # Assembler le système
        A, F = assemble_global_system(mesh, node_to_dof, KAPPA_material, robin_boundaries)

        # Appliquer les conditions de Dirichlet (base à température fixe)
        # Physical ID 2 = Base
        dirichlet_boundaries = {
            2: base_temperature
        }

        A_bc, F_bc = apply_dirichlet_conditions(A, F, mesh, node_to_dof, dirichlet_boundaries)

        # Résoudre
        U = solve_linear_system(A_bc, F_bc, method='direct')

        # Extraire la température maximale
        T_max = np.max(U)

        print(f"  [OK] Temperature maximale: T_max = {T_max:.2f} K")

        # Stocker les résultats
        velocities.append(V)
        T_max_list.append(T_max)
        solutions.append(U)

    print("\n" + "=" * 70)
    print("ÉTUDE PARAMÉTRIQUE TERMINÉE")
    print("=" * 70)

    return velocities, T_max_list, solutions


if __name__ == '__main__':
    print("Module d'étude paramétrique en vitesse")
    print("Utiliser dans le script principal avec un maillage .msh")
