"""
Validation par méthode des solutions manufacturées (MMS).

Ce module implémente la validation de l'implémentation FEM 1D en utilisant
une solution exacte connue et en vérifiant le taux de convergence.

Exercice 2 : Convergence h² pour éléments P1.
"""
import pytest
import numpy as np
from typing import Tuple

from src.mesh.mesh_1d import create_uniform_mesh
from src.core.assembly_1d import assemble_1d_system, apply_dirichlet_1d, solve_1d_system


class ManufacturedSolution1D:
    """
    Classe helper pour la solution manufacturée.

    Solution exacte : u_exact(x) = sin(πx/(2L)) * (x/L)

    Cette solution satisfait :
    - u(0) = 0 automatiquement (BC Dirichlet)
    - u(L) ≠ 0, permettant une BC de Robin non triviale
    - u est C∞ donc régularité suffisante pour analyse d'erreur
    """

    def __init__(self, L: float, kappa: float):
        """
        Initialise la solution manufacturée.

        Args:
            L: Longueur du domaine
            kappa: Conductivité thermique
        """
        self.L = L
        self.kappa = kappa
        self.pi = np.pi

    def exact_solution(self, x: float) -> float:
        """
        Évalue la solution exacte u_exact(x) = sin(πx/(2L)) * (x/L).

        Args:
            x: Position dans [0, L]

        Returns:
            Valeur de u_exact(x)
        """
        return np.sin(self.pi * x / (2 * self.L)) * (x / self.L)

    def exact_derivative(self, x: float) -> float:
        """
        Calcule la dérivée première u'(x).

        u'(x) = (π/(2L))·cos(πx/(2L))·(x/L) + sin(πx/(2L))/L

        Args:
            x: Position dans [0, L]

        Returns:
            Valeur de u'(x)
        """
        pi_over_2L = self.pi / (2 * self.L)
        term1 = pi_over_2L * np.cos(pi_over_2L * x) * (x / self.L)
        term2 = np.sin(pi_over_2L * x) / self.L
        return term1 + term2

    def exact_second_derivative(self, x: float) -> float:
        """
        Calcule la dérivée seconde u''(x).

        u''(x) = -(π²/(4L²))·sin(πx/(2L))·(x/L) + (2π/(2L²))·cos(πx/(2L))

        Args:
            x: Position dans [0, L]

        Returns:
            Valeur de u''(x)
        """
        pi_over_2L = self.pi / (2 * self.L)
        term1 = -(pi_over_2L**2) * np.sin(pi_over_2L * x) * (x / self.L)
        term2 = 2 * (pi_over_2L / self.L) * np.cos(pi_over_2L * x)
        return term1 + term2

    def source_term(self, x: float) -> float:
        """
        Calcule le terme source f(x) = -κ·u''(x).

        Args:
            x: Position dans [0, L]

        Returns:
            Valeur de f(x)
        """
        return -self.kappa * self.exact_second_derivative(x)

    def robin_bc_parameters(self) -> Tuple[float, float]:
        """
        Calcule les paramètres (α, u_E) pour la BC de Robin compatible.

        À x=L, la BC de Robin est : -κ u'(L) = α(u(L) - u_E)

        Pour que u_exact satisfasse cette BC :
        u_E = u(L) + (κ/α)·u'(L)

        On choisit α arbitraire (ex: 10.0) et on calcule u_E compatible.

        Returns:
            Tuple (alpha, u_E)
        """
        alpha = 10.0  # Choix arbitraire

        # Évaluer u et u' à x=L
        u_L = self.exact_solution(self.L)
        u_prime_L = self.exact_derivative(self.L)

        # Calculer u_E compatible
        u_E = u_L + (self.kappa / alpha) * u_prime_L

        return alpha, u_E


@pytest.mark.validation
@pytest.mark.manufactured
class TestManufacturedSolution:
    """
    Tests de validation par solution manufacturée.

    Vérifie que l'implémentation FEM 1D converge au bon taux (h² pour P1).
    """

    def test_exact_solution_satisfies_equation(self):
        """
        Vérifier numériquement que f = -κ u'' est calculé correctement.
        """
        L = 1.0
        kappa = 1.0
        mms = ManufacturedSolution1D(L, kappa)

        # Tester en plusieurs points
        test_points = np.linspace(0, L, 11)
        for x in test_points:
            f_computed = mms.source_term(x)
            u_second = mms.exact_second_derivative(x)
            f_expected = -kappa * u_second

            assert np.isclose(f_computed, f_expected, rtol=1e-12), \
                f"f({x}) = {f_computed}, attendu {f_expected}"

    def test_boundary_condition_compatibility(self):
        """
        Vérifier que u_exact satisfait la BC de Robin avec (α, u_E) calculés.
        """
        L = 1.0
        kappa = 1.0
        mms = ManufacturedSolution1D(L, kappa)

        alpha, u_E = mms.robin_bc_parameters()

        # Vérifier la BC : -κ u'(L) = α(u(L) - u_E)
        u_L = mms.exact_solution(L)
        u_prime_L = mms.exact_derivative(L)

        lhs = -kappa * u_prime_L
        rhs = alpha * (u_L - u_E)

        residual = abs(lhs - rhs)
        assert residual < 1e-12, \
            f"Résidu BC de Robin : {residual:.2e}, attendu < 1e-12"

    @pytest.mark.parametrize("n_elements", [10, 20, 40, 80])
    def test_solution_convergence(self, n_elements: int):
        """
        Test de convergence pour différentes tailles de maillage.

        Ce test est exécuté plusieurs fois avec des maillages de plus en plus fins.
        """
        L = 1.0
        kappa = 1.0
        mms = ManufacturedSolution1D(L, kappa)

        # Créer maillage
        mesh = create_uniform_mesh(L, n_elements)
        h = L / n_elements

        # Terme source manufacturé
        source = lambda x: mms.source_term(x)

        # BC Robin compatibles
        alpha, u_E = mms.robin_bc_parameters()
        robin_bc = {'right': (alpha, u_E)}

        # Assembler et résoudre
        A, F = assemble_1d_system(mesh, kappa, source, robin_bc)
        A, F = apply_dirichlet_1d(A, F, {'left': 0.0}, mesh)
        U_h = solve_1d_system(A, F)

        # Solution exacte
        U_exact = np.array([mms.exact_solution(x) for x in mesh.nodes])

        # Erreur L²
        error = U_h - U_exact
        L2_error = np.sqrt(np.trapz(error**2, mesh.nodes))

        # L'erreur doit décroître
        assert L2_error < 0.1, \
            f"Erreur L² = {L2_error:.2e} trop grande pour n={n_elements}"

        # Pour le maillage le plus fin, l'erreur doit être très petite
        if n_elements == 80:
            assert L2_error < 1e-4, \
                f"Erreur L² = {L2_error:.2e} pour n=80, attendu < 1e-4"

    def test_convergence_rate_h2(self):
        """
        TEST PRINCIPAL : Vérifie le taux de convergence h² pour éléments P1.

        C'est le test clé pour l'Exercice 2.
        """
        L = 1.0
        kappa = 1.0
        mms = ManufacturedSolution1D(L, kappa)

        # Séquence de raffinement
        n_values = [10, 20, 40, 80, 160]
        h_values = []
        L2_errors = []

        print("\n" + "="*70)
        print("Étude de convergence : Solution manufacturée")
        print("="*70)
        print(f"Solution exacte : u(x) = sin(πx/(2L)) · (x/L)")
        print(f"Séquence de maillages : {n_values}\n")

        for n in n_values:
            # Maillage
            mesh = create_uniform_mesh(L, n)
            h = L / n

            # Source manufacturé
            source = lambda x: mms.source_term(x)

            # BC Robin compatibles
            alpha, u_E = mms.robin_bc_parameters()
            robin_bc = {'right': (alpha, u_E)}

            # Assembler et résoudre
            A, F = assemble_1d_system(mesh, kappa, source, robin_bc)
            A, F = apply_dirichlet_1d(A, F, {'left': 0.0}, mesh)
            U_h = solve_1d_system(A, F)

            # Solution exacte
            U_exact = np.array([mms.exact_solution(x) for x in mesh.nodes])

            # Erreur L²
            error = U_h - U_exact
            L2_error = np.sqrt(np.trapz(error**2, mesh.nodes))

            h_values.append(h)
            L2_errors.append(L2_error)

            print(f"n={n:3d}, h={h:.4f}: L²={L2_error:.2e}")

        # Extraire le taux de convergence via régression log-log
        h_arr = np.array(h_values)
        err_arr = np.array(L2_errors)

        # Fit log(err) = log(C) + p·log(h)
        coeffs = np.polyfit(np.log(h_arr), np.log(err_arr), 1)
        convergence_rate = coeffs[0]

        print(f"\nTaux de convergence (norme L²) : {convergence_rate:.2f}")
        print(f"Attendu pour éléments P1 : 2.00")

        # Vérification : le taux doit être proche de 2
        assert 1.8 < convergence_rate < 2.2, \
            f"Taux de convergence = {convergence_rate:.2f}, attendu ≈ 2.0 (entre 1.8 et 2.2)"

        if 1.9 < convergence_rate < 2.1:
            print("✓ Convergence h² vérifiée avec succès!")
        else:
            print(f"⚠ Convergence vérifiée mais taux = {convergence_rate:.2f} un peu éloigné de 2.0")

        print("="*70 + "\n")

    def test_linfinity_error_convergence(self):
        """
        Test de convergence en norme L∞ (erreur maximum).
        """
        L = 1.0
        kappa = 1.0
        mms = ManufacturedSolution1D(L, kappa)

        n_values = [20, 40, 80]
        Linf_errors = []

        for n in n_values:
            mesh = create_uniform_mesh(L, n)

            source = lambda x: mms.source_term(x)
            alpha, u_E = mms.robin_bc_parameters()
            robin_bc = {'right': (alpha, u_E)}

            A, F = assemble_1d_system(mesh, kappa, source, robin_bc)
            A, F = apply_dirichlet_1d(A, F, {'left': 0.0}, mesh)
            U_h = solve_1d_system(A, F)

            U_exact = np.array([mms.exact_solution(x) for x in mesh.nodes])

            Linf_error = np.max(np.abs(U_h - U_exact))
            Linf_errors.append(Linf_error)

        # Vérifier que l'erreur décroît
        for i in range(len(Linf_errors) - 1):
            assert Linf_errors[i+1] < Linf_errors[i], \
                "L'erreur L∞ doit décroître avec le raffinement"

    def test_quadratic_source_accuracy(self):
        """
        Test avec un terme source polynomial : f(x) = x².

        Pour ce cas, la solution exacte n'est pas polynomiale, mais on peut
        vérifier que le solveur fonctionne correctement.
        """
        L = 1.0
        kappa = 1.0
        mesh = create_uniform_mesh(L, 50)

        # Source f(x) = x²
        source = lambda x: x**2

        # BC simples
        A, F = assemble_1d_system(mesh, kappa, source, robin_bc=None)
        A, F = apply_dirichlet_1d(A, F, {'left': 0.0, 'right': 1.0}, mesh)
        U = solve_1d_system(A, F)

        # Vérifier BC
        assert np.isclose(U[0], 0.0, atol=1e-12)
        assert np.isclose(U[-1], 1.0, atol=1e-12)

        # Vérifier que la solution est raisonnable
        assert np.all(U >= 0.0), "La température doit être non-négative"
        assert np.all(U <= 1.0 + 0.1), "La température ne doit pas exploser"


@pytest.mark.validation
class TestConvergenceRobustness:
    """
    Tests de robustesse pour la convergence.
    """

    def test_very_fine_mesh_stability(self):
        """
        Vérifier qu'un maillage très fin ne cause pas d'instabilité.
        """
        L = 1.0
        kappa = 1.0
        mms = ManufacturedSolution1D(L, kappa)

        # Maillage très fin
        mesh = create_uniform_mesh(L, n_elements=500)

        source = lambda x: mms.source_term(x)
        alpha, u_E = mms.robin_bc_parameters()
        robin_bc = {'right': (alpha, u_E)}

        A, F = assemble_1d_system(mesh, kappa, source, robin_bc)
        A, F = apply_dirichlet_1d(A, F, {'left': 0.0}, mesh)
        U = solve_1d_system(A, F)

        # Vérifier pas de NaN ou Inf
        assert np.all(np.isfinite(U)), "Solution contient NaN ou Inf"

        # Vérifier BC
        assert np.isclose(U[0], 0.0, atol=1e-10)

    def test_different_kappa_values(self):
        """
        Test avec différentes valeurs de κ.
        """
        L = 1.0
        kappa_values = [0.1, 1.0, 10.0, 100.0]

        for kappa in kappa_values:
            mms = ManufacturedSolution1D(L, kappa)
            mesh = create_uniform_mesh(L, 40)

            source = lambda x: mms.source_term(x)
            alpha, u_E = mms.robin_bc_parameters()
            robin_bc = {'right': (alpha, u_E)}

            A, F = assemble_1d_system(mesh, kappa, source, robin_bc)
            A, F = apply_dirichlet_1d(A, F, {'left': 0.0}, mesh)
            U = solve_1d_system(A, F)

            U_exact = np.array([mms.exact_solution(x) for x in mesh.nodes])
            L2_error = np.sqrt(np.trapz((U - U_exact)**2, mesh.nodes))

            # L'erreur doit rester raisonnable quel que soit κ
            assert L2_error < 0.01, \
                f"Erreur L² = {L2_error:.2e} pour κ={kappa}"


if __name__ == '__main__':
    # Exécuter seulement le test principal de convergence
    pytest.main([__file__, '-v', '-k', 'test_convergence_rate_h2', '--tb=short'])
