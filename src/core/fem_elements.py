"""
@file fem_elements.py
@brief Finite Element Method P1 (linear) elements implementation
@author HPC-Code-Documenter
@date 2025

@details
This module implements P1 (linear) finite elements for thermal analysis.
It provides shape functions, gradients, and quadrature rules for:
- 2D triangular elements (TriangleP1)
- 1D edge elements for boundary integration (EdgeP1)

The mathematical formulation uses reference element transformations:
- Reference triangle: vertices at $(0,0)$, $(1,0)$, $(0,1)$
- Reference edge: $\\xi \\in [-1, 1]$

Key equations implemented:
- Shape functions: $\\phi_i(\\xi, \\eta)$ for interpolation
- Jacobian transformation: $J = \\frac{\\partial(x,y)}{\\partial(\\xi,\\eta)}$
- Physical gradients: $\\nabla\\phi = J^{-T} \\nabla_{ref}\\phi$
- Stiffness matrix: $K^e_{ij} = \\int_K \\kappa \\nabla\\phi_i \\cdot \\nabla\\phi_j \\, dK$
"""
import logging
import numpy as np
from typing import Tuple
from numpy.typing import NDArray

from src.utils.exceptions import ElementError, ValidationError

logger = logging.getLogger(__name__)


class TriangleP1:
    """
    @brief P1 triangular finite element with 3 nodes and linear shape functions.

    @details
    Implements a 2D triangular element using linear (P1) interpolation.
    The reference element has vertices at $(0,0)$, $(1,0)$, and $(0,1)$.

    Shape functions in reference coordinates $(\\xi, \\eta)$:
    - $\\phi_1 = 1 - \\xi - \\eta$
    - $\\phi_2 = \\xi$
    - $\\phi_3 = \\eta$

    This element is suitable for thermal diffusion problems where
    the temperature field varies linearly within each element.
    """

    @staticmethod
    def shape_functions(xi: float, eta: float) -> NDArray[np.float64]:
        """
        @brief Compute P1 shape functions at reference coordinates.

        @details
        Evaluates the three linear shape functions at point $(\\xi, \\eta)$
        in the reference triangle with vertices at $(0,0)$, $(1,0)$, $(0,1)$.

        The shape functions satisfy the partition of unity:
        $\\sum_{i=1}^{3} \\phi_i(\\xi, \\eta) = 1$

        @param xi: Reference coordinate $\\xi \\in [0, 1]$
        @param eta: Reference coordinate $\\eta \\in [0, 1-\\xi]$

        @return NDArray of shape (3,) containing $[\\phi_1, \\phi_2, \\phi_3]$

        @example
        >>> phi = TriangleP1.shape_functions(0.0, 0.0)
        >>> # Returns [1, 0, 0] at vertex 1
        """
        phi = np.array([
            1 - xi - eta,
            xi,
            eta
        ])
        return phi

    @staticmethod
    def shape_gradients_ref() -> NDArray[np.float64]:
        """
        @brief Compute gradients of shape functions in reference element.

        @details
        Returns the constant gradients of P1 shape functions:
        $\\nabla\\phi_i = [\\partial\\phi_i/\\partial\\xi, \\partial\\phi_i/\\partial\\eta]$

        For P1 elements, gradients are constant over the element:
        - $\\nabla\\phi_1 = [-1, -1]$
        - $\\nabla\\phi_2 = [1, 0]$
        - $\\nabla\\phi_3 = [0, 1]$

        @return NDArray of shape (3, 2) where each row is the gradient of one shape function
        """
        grad_phi = np.array([
            [-1.0, -1.0],
            [ 1.0,  0.0],
            [ 0.0,  1.0]
        ])
        return grad_phi

    @staticmethod
    def compute_jacobian(coords: NDArray[np.float64]) -> Tuple[NDArray[np.float64], float]:
        """
        @brief Compute the Jacobian matrix of the reference-to-physical transformation.

        @details
        Computes the Jacobian matrix $J$ of the isoparametric mapping
        from reference to physical coordinates:

        $J = \\begin{bmatrix} \\partial x/\\partial\\xi & \\partial x/\\partial\\eta \\\\
        \\partial y/\\partial\\xi & \\partial y/\\partial\\eta \\end{bmatrix}$

        For P1 elements, $J$ simplifies to:
        $J = \\begin{bmatrix} x_2-x_1 & x_3-x_1 \\\\ y_2-y_1 & y_3-y_1 \\end{bmatrix}$

        The determinant $|J|$ relates to the element area: $A = |J|/2$

        @param coords: Physical coordinates of the 3 vertices, shape (3, 2)

        @return Tuple containing:
            - J: Jacobian matrix of shape (2, 2)
            - det_J: Determinant of the Jacobian (can be negative if nodes are clockwise)

        @raises ValidationError: If coords does not have shape (3, 2)
        """
        if coords.shape != (3, 2):
            raise ValidationError(f"coords doit être de forme (3, 2), reçu {coords.shape}")

        J = np.array([
            [coords[1, 0] - coords[0, 0], coords[2, 0] - coords[0, 0]],
            [coords[1, 1] - coords[0, 1], coords[2, 1] - coords[0, 1]]
        ])
        det_J = np.linalg.det(J)
        return J, det_J

    @staticmethod
    def physical_gradients(coords: NDArray[np.float64]) -> Tuple[NDArray[np.float64], float]:
        """
        @brief Compute physical gradients of shape functions.

        @details
        Transforms reference gradients to physical gradients using:
        $\\nabla\\phi_i = J^{-T} \\nabla_{ref}\\phi_i$

        where $J^{-T}$ is the inverse transpose of the Jacobian matrix.
        Also computes the element area from the Jacobian determinant.

        @param coords: Physical coordinates of the 3 vertices, shape (3, 2)

        @return Tuple containing:
            - grad_phi_phys: Physical gradients of shape (3, 2)
            - area: Area of the triangle in physical space

        @raises ElementError: If the element is degenerate (zero or near-zero area)
        """
        J, det_J = TriangleP1.compute_jacobian(coords)

        if abs(det_J) < 1e-14:
            raise ElementError(
                f"Élément dégénéré: déterminant du Jacobien = {det_J:.2e}. "
                "Le triangle a une aire nulle ou quasi-nulle."
            )

        area = abs(det_J) / 2.0

        J_inv_T = np.linalg.inv(J).T
        grad_phi_ref = TriangleP1.shape_gradients_ref()
        grad_phi_phys = grad_phi_ref @ J_inv_T.T

        return grad_phi_phys, area

    @staticmethod
    def quadrature_volume() -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        @brief Get quadrature points and weights for triangle integration.

        @details
        Returns the centroid quadrature rule (order 1) for the reference triangle.
        This rule uses a single point at the centroid $(1/3, 1/3)$ with
        weight $1/2$ (the area of the reference triangle).

        The rule is exact for polynomials of degree 1 (linear functions).

        @return Tuple containing:
            - points: Quadrature points of shape (N, 2)
            - weights: Quadrature weights of shape (N,)
        """
        points = np.array([[1.0/3.0, 1.0/3.0]])
        weights = np.array([0.5])
        return points, weights

    @staticmethod
    def local_stiffness_matrix(coords: NDArray[np.float64], kappa: float) -> NDArray[np.float64]:
        """
        @brief Compute the element stiffness matrix for thermal diffusion.

        @details
        Computes the local stiffness matrix $K^e$ for the heat equation:

        $K^e_{ij} = \\int_K \\kappa \\nabla\\phi_i \\cdot \\nabla\\phi_j \\, dK$

        For P1 elements with constant conductivity, this simplifies to:
        $K^e = \\kappa \\cdot A \\cdot (\\nabla\\phi)(\\nabla\\phi)^T$

        where $A$ is the element area.

        @param coords: Physical coordinates of the 3 vertices, shape (3, 2)
        @param kappa: Thermal conductivity coefficient $\\kappa > 0$ [W/(m.K)]

        @return Element stiffness matrix of shape (3, 3)

        @raises ValidationError: If kappa <= 0
        @raises ElementError: If the element is degenerate

        @example
        >>> coords = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
        >>> K = TriangleP1.local_stiffness_matrix(coords, kappa=1.0)
        """
        if kappa <= 0:
            raise ValidationError(f"kappa doit être positif, reçu {kappa}")

        grad_phi, area = TriangleP1.physical_gradients(coords)
        K_elem = kappa * area * (grad_phi @ grad_phi.T)
        return K_elem


class EdgeP1:
    """
    @brief P1 edge element with 2 nodes for boundary integration.

    @details
    Implements a 1D edge element for integrating boundary conditions
    (convection, heat flux) along element edges.

    The reference element uses coordinate $\\xi \\in [-1, 1]$ with:
    - Node 1 at $\\xi = -1$
    - Node 2 at $\\xi = +1$

    Shape functions:
    - $\\phi_1 = (1 - \\xi) / 2$
    - $\\phi_2 = (1 + \\xi) / 2$
    """

    @staticmethod
    def shape_functions(xi: float) -> NDArray[np.float64]:
        """
        @brief Compute 1D shape functions at reference coordinate.

        @details
        Evaluates the two linear shape functions at point $\\xi$
        in the reference edge $[-1, 1]$.

        @param xi: Reference coordinate $\\xi \\in [-1, 1]$

        @return NDArray of shape (2,) containing $[\\phi_1, \\phi_2]$
        """
        phi = np.array([
            (1 - xi) / 2.0,
            (1 + xi) / 2.0
        ])
        return phi

    @staticmethod
    def compute_length(coords: NDArray[np.float64]) -> float:
        """
        @brief Compute the physical length of the edge.

        @details
        Calculates the Euclidean distance between the two edge nodes:
        $L = \\sqrt{(x_2-x_1)^2 + (y_2-y_1)^2}$

        @param coords: Physical coordinates of the 2 nodes, shape (2, 2)

        @return Length of the edge in physical units

        @raises ValidationError: If coords does not have shape (2, 2)
        @raises ElementError: If the edge has zero or near-zero length
        """
        if coords.shape != (2, 2):
            raise ValidationError(f"coords doit être de forme (2, 2), reçu {coords.shape}")

        dx = coords[1, 0] - coords[0, 0]
        dy = coords[1, 1] - coords[0, 1]
        length = np.sqrt(dx**2 + dy**2)

        if length < 1e-14:
            raise ElementError(f"Arête dégénérée: longueur = {length:.2e}")

        return length

    @staticmethod
    def quadrature_line() -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        @brief Get Gauss-Legendre quadrature points and weights for line integration.

        @details
        Returns a 2-point Gauss-Legendre quadrature rule for the interval $[-1, 1]$.
        Points are at $\\pm 1/\\sqrt{3}$ with weights of 1.

        This rule is exact for polynomials up to degree 3.

        @return Tuple containing:
            - points: Quadrature points of shape (2,)
            - weights: Quadrature weights of shape (2,)
        """
        xi_gauss = 1.0 / np.sqrt(3.0)
        points = np.array([-xi_gauss, xi_gauss])
        weights = np.array([1.0, 1.0])
        return points, weights

    @staticmethod
    def local_mass_matrix(coords: NDArray[np.float64], alpha: float) -> NDArray[np.float64]:
        """
        @brief Compute the element boundary mass matrix for convection.

        @details
        Computes the local mass matrix for Robin boundary conditions:

        $M^e_{ij} = \\int_e \\alpha \\phi_i \\phi_j \\, d\\sigma$

        This matrix represents the convective heat transfer contribution
        to the global system, where $\\alpha$ is the heat transfer coefficient.

        @param coords: Physical coordinates of the 2 nodes, shape (2, 2)
        @param alpha: Convection heat transfer coefficient $\\alpha \\geq 0$ [W/(m^2.K)]

        @return Element boundary mass matrix of shape (2, 2)

        @raises ValidationError: If alpha < 0
        @raises ElementError: If the edge is degenerate
        """
        if alpha < 0:
            raise ValidationError(f"alpha doit être >= 0, reçu {alpha}")

        length = EdgeP1.compute_length(coords)
        points, weights = EdgeP1.quadrature_line()

        M_elem = np.zeros((2, 2))

        for xi, w in zip(points, weights):
            phi = EdgeP1.shape_functions(xi)
            M_elem += alpha * w * (length / 2.0) * np.outer(phi, phi)

        return M_elem

    @staticmethod
    def local_load_vector(coords: NDArray[np.float64], alpha: float, u_E: float) -> NDArray[np.float64]:
        """
        @brief Compute the element boundary load vector for convection.

        @details
        Computes the local load vector for Robin boundary conditions:

        $F^e_i = \\int_e \\alpha u_E \\phi_i \\, d\\sigma$

        This vector represents the external temperature contribution
        from convective heat transfer, where $u_E$ is the ambient temperature.

        @param coords: Physical coordinates of the 2 nodes, shape (2, 2)
        @param alpha: Convection heat transfer coefficient $\\alpha \\geq 0$ [W/(m^2.K)]
        @param u_E: External/ambient temperature [K or degC]

        @return Element boundary load vector of shape (2,)

        @raises ValidationError: If alpha < 0
        @raises ElementError: If the edge is degenerate
        """
        if alpha < 0:
            raise ValidationError(f"alpha doit être >= 0, reçu {alpha}")

        length = EdgeP1.compute_length(coords)
        points, weights = EdgeP1.quadrature_line()

        F_elem = np.zeros(2)

        for xi, w in zip(points, weights):
            phi = EdgeP1.shape_functions(xi)
            F_elem += alpha * u_E * w * (length / 2.0) * phi

        return F_elem


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

    logger.info("=== Test Triangle P1 ===")
    coords_tri = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
    grad, area = TriangleP1.physical_gradients(coords_tri)
    logger.info(f"Aire triangle: {area}")
    logger.debug(f"Gradients physiques:\n{grad}")

    K = TriangleP1.local_stiffness_matrix(coords_tri, kappa=1.0)
    logger.debug(f"Matrice de rigidité (κ=1):\n{K}")

    logger.info("=== Test Arête P1 ===")
    coords_edge = np.array([[0, 0], [1, 0]], dtype=np.float64)
    length = EdgeP1.compute_length(coords_edge)
    logger.info(f"Longueur arête: {length}")

    M = EdgeP1.local_mass_matrix(coords_edge, alpha=10.0)
    logger.debug(f"Matrice de masse surfacique (α=10):\n{M}")

    F = EdgeP1.local_load_vector(coords_edge, alpha=10.0, u_E=300.0)
    logger.debug(f"Vecteur de charge surfacique (α=10, u_E=300):\n{F}")

    logger.info("Tests terminés avec succès")
