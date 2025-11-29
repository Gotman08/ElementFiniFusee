"""
Module d'éléments finis P1 (linéaires)
Implémente les fonctions de forme, gradients, et quadrature pour triangles et arêtes
"""
import numpy as np
from typing import Tuple

# ============================================================================
# ÉLÉMENTS TRIANGULAIRES P1 (2D)
# ============================================================================

class TriangleP1:
    """Élément triangulaire P1 (3 noeuds, fonctions linéaires)"""

    @staticmethod
    def shape_functions(xi, eta):
        """
        Fonctions de forme P1 en coordonnées de référence (ξ, η)
        Triangle de référence: (0,0), (1,0), (0,1)

        Args:
            xi, eta: Coordonnées de référence

        Returns:
            Array [φ_1, φ_2, φ_3]
        """
        phi = np.array([
            1 - xi - eta,  # φ_1
            xi,            # φ_2
            eta            # φ_3
        ])
        return phi

    @staticmethod
    def shape_gradients_ref():
        """
        Gradients des fonctions de forme dans l'élément de référence
        ∇φ_i = [∂φ_i/∂ξ, ∂φ_i/∂η]

        Returns:
            Array (3, 2): Chaque ligne = gradient d'une fonction de forme
        """
        grad_phi = np.array([
            [-1.0, -1.0],  # ∇φ_1
            [ 1.0,  0.0],  # ∇φ_2
            [ 0.0,  1.0]   # ∇φ_3
        ])
        return grad_phi

    @staticmethod
    def compute_jacobian(coords):
        """
        Calcule la matrice Jacobienne de la transformation F: ref -> physique
        J = [∂x/∂ξ  ∂x/∂η]
            [∂y/∂ξ  ∂y/∂η]

        Args:
            coords: Array (3, 2) des coordonnées physiques [x, y] des 3 sommets

        Returns:
            (J, det_J): Matrice Jacobienne et son déterminant
        """
        # J = [x_2-x_1, x_3-x_1]
        #     [y_2-y_1, y_3-y_1]
        J = np.array([
            [coords[1, 0] - coords[0, 0], coords[2, 0] - coords[0, 0]],
            [coords[1, 1] - coords[0, 1], coords[2, 1] - coords[0, 1]]
        ])
        det_J = np.linalg.det(J)
        return J, det_J

    @staticmethod
    def physical_gradients(coords):
        """
        Calcule les gradients physiques ∇φ_i = J^(-T) * ∇_ref φ_i

        Args:
            coords: Array (3, 2) des coordonnées physiques

        Returns:
            (grad_phi_phys, area): Gradients physiques (3, 2) et aire du triangle
        """
        J, det_J = TriangleP1.compute_jacobian(coords)
        area = abs(det_J) / 2.0

        # Gradients physiques: ∇φ = J^(-T) * ∇_ref φ
        J_inv_T = np.linalg.inv(J).T
        grad_phi_ref = TriangleP1.shape_gradients_ref()
        grad_phi_phys = grad_phi_ref @ J_inv_T.T  # (3, 2)

        return grad_phi_phys, area

    @staticmethod
    def quadrature_volume():
        """
        Points et poids de quadrature pour intégration sur triangle de référence
        Formule du centre de gravité (ordre 1, exacte pour polynômes degré 1)

        Returns:
            (points, weights): Points (N, 2) et poids (N,)
        """
        # Centre de gravité: (1/3, 1/3), poids = 1/2 (aire du triangle de référence)
        points = np.array([[1.0/3.0, 1.0/3.0]])
        weights = np.array([0.5])
        return points, weights

    @staticmethod
    def local_stiffness_matrix(coords, kappa):
        """
        Calcule la matrice de rigidité élémentaire K^e
        K^e_{ij} = ∫_K κ ∇φ_i · ∇φ_j dK

        Args:
            coords: Coordonnées physiques (3, 2)
            kappa: Conductivité thermique

        Returns:
            K_elem: Matrice 3x3
        """
        grad_phi, area = TriangleP1.physical_gradients(coords)
        # K^e = κ * area * (∇φ^T * ∇φ)
        K_elem = kappa * area * (grad_phi @ grad_phi.T)
        return K_elem


# ============================================================================
# ÉLÉMENTS ARÊTES P1 (1D sur bord)
# ============================================================================

class EdgeP1:
    """Élément arête P1 (2 noeuds) pour intégration sur les bords"""

    @staticmethod
    def shape_functions(xi):
        """
        Fonctions de forme 1D en coordonnée de référence ξ ∈ [-1, 1]

        Args:
            xi: Coordonnée de référence

        Returns:
            Array [φ_1, φ_2]
        """
        phi = np.array([
            (1 - xi) / 2.0,  # φ_1
            (1 + xi) / 2.0   # φ_2
        ])
        return phi

    @staticmethod
    def compute_length(coords):
        """
        Calcule la longueur de l'arête

        Args:
            coords: Array (2, 2) des coordonnées [x, y] des 2 noeuds

        Returns:
            length: Longueur de l'arête
        """
        dx = coords[1, 0] - coords[0, 0]
        dy = coords[1, 1] - coords[0, 1]
        length = np.sqrt(dx**2 + dy**2)
        return length

    @staticmethod
    def quadrature_line():
        """
        Points et poids de quadrature pour intégration sur [-1, 1]
        Formule de Gauss à 2 points (exacte pour polynômes degré 3)

        Returns:
            (points, weights): Points (N,) et poids (N,)
        """
        # Points de Gauss-Legendre à 2 points
        xi_gauss = 1.0 / np.sqrt(3.0)
        points = np.array([-xi_gauss, xi_gauss])
        weights = np.array([1.0, 1.0])
        return points, weights

    @staticmethod
    def local_mass_matrix(coords, alpha):
        """
        Calcule la matrice de masse surfacique élémentaire M^e
        M^e_{ij} = ∫_e α φ_i φ_j dσ

        Args:
            coords: Coordonnées physiques (2, 2)
            alpha: Coefficient de convection

        Returns:
            M_elem: Matrice 2x2
        """
        length = EdgeP1.compute_length(coords)
        points, weights = EdgeP1.quadrature_line()

        M_elem = np.zeros((2, 2))

        for xi, w in zip(points, weights):
            phi = EdgeP1.shape_functions(xi)
            # M^e += α * w * (length/2) * φ ⊗ φ
            M_elem += alpha * w * (length / 2.0) * np.outer(phi, phi)

        return M_elem

    @staticmethod
    def local_load_vector(coords, alpha, u_E):
        """
        Calcule le vecteur de charge surfacique élémentaire F^e
        F^e_i = ∫_e α u_E φ_i dσ

        Args:
            coords: Coordonnées physiques (2, 2)
            alpha: Coefficient de convection
            u_E: Température extérieure

        Returns:
            F_elem: Vecteur de taille 2
        """
        length = EdgeP1.compute_length(coords)
        points, weights = EdgeP1.quadrature_line()

        F_elem = np.zeros(2)

        for xi, w in zip(points, weights):
            phi = EdgeP1.shape_functions(xi)
            # F^e += α * u_E * w * (length/2) * φ
            F_elem += alpha * u_E * w * (length / 2.0) * phi

        return F_elem


if __name__ == '__main__':
    # Tests unitaires
    print("=== Test Triangle P1 ===")
    coords_tri = np.array([[0, 0], [1, 0], [0, 1]])
    grad, area = TriangleP1.physical_gradients(coords_tri)
    print(f"Aire triangle: {area}")
    print(f"Gradients physiques:\n{grad}")

    K = TriangleP1.local_stiffness_matrix(coords_tri, kappa=1.0)
    print(f"Matrice de rigidité (κ=1):\n{K}")

    print("\n=== Test Arête P1 ===")
    coords_edge = np.array([[0, 0], [1, 0]])
    length = EdgeP1.compute_length(coords_edge)
    print(f"Longueur arête: {length}")

    M = EdgeP1.local_mass_matrix(coords_edge, alpha=10.0)
    print(f"Matrice de masse surfacique (α=10):\n{M}")

    F = EdgeP1.local_load_vector(coords_edge, alpha=10.0, u_E=300.0)
    print(f"Vecteur de charge surfacique (α=10, u_E=300):\n{F}")
