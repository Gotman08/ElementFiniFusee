"""
@file fem_elements_1d.py
@brief Éléments finis P1 (linéaires) pour problèmes 1D
@author ElementFiniFusee
@date 2025

@details
Ce module implémente les éléments finis P1 pour l'analyse thermique 1D.
Il fournit les fonctions de forme, gradients et règles de quadrature pour :
- Segments 1D (SegmentP1) avec 2 nœuds
- Points 0D (PointP1) pour l'intégration des conditions aux limites de Robin

La formulation mathématique utilise des transformations d'éléments de référence :
- Élément de référence : $\\xi \\in [-1, 1]$
- Transformation affine : $x(\\xi) = \\phi_1(\\xi) x_1 + \\phi_2(\\xi) x_2$

Équations clés implémentées :
- Fonctions de forme : $\\phi_i(\\xi)$ pour l'interpolation
- Jacobien : $J = dx/d\\xi = h/2$ où $h = x_2 - x_1$
- Gradients physiques : $d\\phi/dx = (d\\phi/d\\xi) / J$
- Matrice de rigidité : $K^e_{ij} = \\int_K \\kappa \\frac{d\\phi_i}{dx} \\frac{d\\phi_j}{dx} dx$
"""
import logging
import numpy as np
from typing import Tuple, Union, Callable
from numpy.typing import NDArray

from src.utils.exceptions import ElementError, ValidationError

logger = logging.getLogger(__name__)


class SegmentP1:
    """
    @brief Élément fini P1 pour segment 1D avec 2 nœuds et fonctions de forme linéaires.

    @details
    Implémente un élément 1D utilisant une interpolation linéaire (P1).
    L'élément de référence est défini sur $\\xi \\in [-1, 1]$.

    Fonctions de forme en coordonnées de référence $\\xi$ :
    - $\\phi_1 = (1 - \\xi) / 2$
    - $\\phi_2 = (1 + \\xi) / 2$

    Cet élément est adapté pour les problèmes de diffusion thermique où
    le champ de température varie linéairement dans chaque élément.
    """

    @staticmethod
    def shape_functions(xi: float) -> NDArray[np.float64]:
        """
        @brief Calcule les fonctions de forme P1 aux coordonnées de référence.

        @details
        Évalue les deux fonctions de forme linéaires au point $\\xi$
        dans l'élément de référence $[-1, 1]$.

        Les fonctions de forme satisfont la partition de l'unité :
        $\\sum_{i=1}^{2} \\phi_i(\\xi) = 1$

        @param xi: Coordonnée de référence $\\xi \\in [-1, 1]$

        @return NDArray de forme (2,) contenant $[\\phi_1, \\phi_2]$

        @example
        >>> phi = SegmentP1.shape_functions(-1.0)
        >>> # Retourne [1, 0] au nœud 1
        >>> phi = SegmentP1.shape_functions(1.0)
        >>> # Retourne [0, 1] au nœud 2
        """
        phi = np.array([
            (1 - xi) / 2.0,
            (1 + xi) / 2.0
        ])
        return phi

    @staticmethod
    def shape_derivatives_ref() -> NDArray[np.float64]:
        """
        @brief Calcule les gradients des fonctions de forme dans l'élément de référence.

        @details
        Retourne les gradients constants des fonctions de forme P1 :
        $d\\phi_i/d\\xi$

        Pour les éléments P1, les gradients sont constants sur l'élément :
        - $d\\phi_1/d\\xi = -1/2$
        - $d\\phi_2/d\\xi = 1/2$

        @return NDArray de forme (2,) où chaque élément est le gradient d'une fonction de forme
        """
        grad_phi = np.array([-0.5, 0.5])
        return grad_phi

    @staticmethod
    def compute_jacobian(coords: NDArray[np.float64]) -> Tuple[float, float]:
        """
        @brief Calcule le Jacobien de la transformation référence → physique.

        @details
        Calcule le Jacobien $J$ de la transformation isoparamétrique
        des coordonnées de référence aux coordonnées physiques :

        $J = \\frac{dx}{d\\xi} = \\frac{x_2 - x_1}{2} = \\frac{h}{2}$

        où $h = x_2 - x_1$ est la longueur de l'élément.

        @param coords: Coordonnées physiques des 2 nœuds, forme (2,)

        @return Tuple contenant :
            - J: Jacobien (scalaire)
            - det_J: Déterminant du Jacobien (égal à J en 1D)

        @raises ValidationError: Si coords n'a pas la forme (2,)
        @raises ElementError: Si l'élément est dégénéré (longueur nulle)
        """
        if coords.shape != (2,):
            raise ValidationError(f"coords doit être de forme (2,), reçu {coords.shape}")

        h = coords[1] - coords[0]

        if abs(h) < 1e-14:
            raise ElementError(
                f"Élément dégénéré : longueur h = {h:.2e}. "
                "Le segment a une longueur nulle ou quasi-nulle."
            )

        J = h / 2.0
        det_J = J

        return J, det_J

    @staticmethod
    def physical_gradients(coords: NDArray[np.float64]) -> Tuple[NDArray[np.float64], float]:
        """
        @brief Calcule les gradients physiques des fonctions de forme.

        @details
        Transforme les gradients de référence en gradients physiques :
        $\\frac{d\\phi_i}{dx} = \\frac{d\\phi_i}{d\\xi} \\cdot \\frac{d\\xi}{dx} = \\frac{d\\phi_i}{d\\xi} / J$

        Calcule également la longueur de l'élément.

        @param coords: Coordonnées physiques des 2 nœuds, forme (2,)

        @return Tuple contenant :
            - grad_phi_phys: Gradients physiques de forme (2,)
            - length: Longueur de l'élément en espace physique

        @raises ElementError: Si l'élément est dégénéré
        """
        J, _ = SegmentP1.compute_jacobian(coords)
        length = 2.0 * J

        grad_phi_ref = SegmentP1.shape_derivatives_ref()
        grad_phi_phys = grad_phi_ref / J

        return grad_phi_phys, length

    @staticmethod
    def quadrature_volume() -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        @brief Retourne les points et poids de quadrature pour l'intégration sur le segment.

        @details
        Retourne la règle de quadrature de Gauss-Legendre à 2 points pour l'intervalle $[-1, 1]$.
        Points à $\\pm 1/\\sqrt{3}$ avec poids de 1.

        Cette règle est exacte pour les polynômes jusqu'au degré 3.

        @return Tuple contenant :
            - points: Points de quadrature de forme (2,)
            - weights: Poids de quadrature de forme (2,)
        """
        xi_gauss = 1.0 / np.sqrt(3.0)
        points = np.array([-xi_gauss, xi_gauss])
        weights = np.array([1.0, 1.0])
        return points, weights

    @staticmethod
    def local_stiffness_matrix(coords: NDArray[np.float64], kappa: float) -> NDArray[np.float64]:
        """
        @brief Calcule la matrice de rigidité élémentaire pour la diffusion thermique.

        @details
        Calcule la matrice de rigidité locale $K^e$ pour l'équation de la chaleur :

        $K^e_{ij} = \\int_K \\kappa \\frac{d\\phi_i}{dx} \\frac{d\\phi_j}{dx} dx$

        Pour les éléments P1 avec conductivité constante, ceci se simplifie en :
        $K^e = \\frac{\\kappa}{h} \\begin{bmatrix} 1 & -1 \\\\ -1 & 1 \\end{bmatrix}$

        où $h$ est la longueur de l'élément.

        @param coords: Coordonnées physiques des 2 nœuds, forme (2,)
        @param kappa: Coefficient de conductivité thermique $\\kappa > 0$ [W/(m.K)]

        @return Matrice de rigidité élémentaire de forme (2, 2)

        @raises ValidationError: Si kappa <= 0
        @raises ElementError: Si l'élément est dégénéré

        @example
        >>> coords = np.array([0.0, 1.0])
        >>> K = SegmentP1.local_stiffness_matrix(coords, kappa=10.0)
        >>> # Retourne [[10, -10], [-10, 10]]
        """
        if kappa <= 0:
            raise ValidationError(f"kappa doit être positif, reçu {kappa}")

        grad_phi, length = SegmentP1.physical_gradients(coords)

        # K^e = kappa * length * (grad_phi ⊗ grad_phi)
        # Où grad_phi est un vecteur colonne, donc outer product
        K_elem = kappa * length * np.outer(grad_phi, grad_phi)

        return K_elem

    @staticmethod
    def local_mass_matrix(coords: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        @brief Calcule la matrice de masse élémentaire.

        @details
        Calcule la matrice de masse locale $M^e$ :

        $M^e_{ij} = \\int_K \\phi_i \\phi_j dx$

        Utile pour les problèmes transitoires ou les conditions aux limites de type masse.

        @param coords: Coordonnées physiques des 2 nœuds, forme (2,)

        @return Matrice de masse élémentaire de forme (2, 2)

        @raises ElementError: Si l'élément est dégénéré
        """
        J, _ = SegmentP1.compute_jacobian(coords)
        points, weights = SegmentP1.quadrature_volume()

        M_elem = np.zeros((2, 2))

        for xi, w in zip(points, weights):
            phi = SegmentP1.shape_functions(xi)
            # Intégration: dxi → dx via Jacobien
            M_elem += w * J * np.outer(phi, phi)

        return M_elem

    @staticmethod
    def local_load_vector(coords: NDArray[np.float64],
                         source_term: Union[float, Callable[[float], float]]) -> NDArray[np.float64]:
        """
        @brief Calcule le vecteur de charge élémentaire pour un terme source.

        @details
        Calcule le vecteur de charge local $F^e$ :

        $F^e_i = \\int_K f(x) \\phi_i(x) dx$

        où $f(x)$ est le terme source volumique.

        @param coords: Coordonnées physiques des 2 nœuds, forme (2,)
        @param source_term: Terme source constant (float) OU fonction f(x) callable

        @return Vecteur de charge élémentaire de forme (2,)

        @raises ElementError: Si l'élément est dégénéré

        @example
        >>> coords = np.array([0.0, 1.0])
        >>> F = SegmentP1.local_load_vector(coords, source_term=100.0)
        >>> # Source constante
        >>> F_func = SegmentP1.local_load_vector(coords, lambda x: x**2)
        >>> # Source variable
        """
        J, _ = SegmentP1.compute_jacobian(coords)
        points, weights = SegmentP1.quadrature_volume()

        F_elem = np.zeros(2)

        for xi, w in zip(points, weights):
            phi = SegmentP1.shape_functions(xi)

            # Mapper vers coordonnée physique
            x_phys = phi @ coords

            # Évaluer le terme source
            if callable(source_term):
                f_value = source_term(x_phys)
            else:
                f_value = source_term

            # Intégration: dxi → dx via Jacobien
            F_elem += f_value * w * J * phi

        return F_elem


class PointP1:
    """
    @brief Élément 0D pour les conditions aux limites de Robin aux extrémités.

    @details
    Implémente un élément ponctuel pour intégrer les conditions aux limites
    de type Robin (ou Fourier) :

    $-\\kappa \\frac{du}{dn} = \\alpha(u - u_E)$

    En 1D, cela se réduit à ajouter des contributions aux nœuds de bord.
    """

    @staticmethod
    def local_robin_matrix(alpha: float) -> float:
        """
        @brief Calcule la contribution de la condition de Robin à la matrice.

        @details
        Pour une condition de Robin au point $x = x_b$ :

        $-\\kappa u'(x_b) = \\alpha(u(x_b) - u_E)$

        La contribution à la matrice globale est simplement :
        $A[i, i] += \\alpha$

        où $i$ est le DOF au point de bord.

        @param alpha: Coefficient de transfert thermique $\\alpha \\geq 0$ [W/(m^2.K)]

        @return Scalaire $\\alpha$ à ajouter au terme diagonal

        @raises ValidationError: Si alpha < 0
        """
        if alpha < 0:
            raise ValidationError(f"alpha doit être >= 0, reçu {alpha}")

        return alpha

    @staticmethod
    def local_robin_load(alpha: float, u_E: float) -> float:
        """
        @brief Calcule la contribution de la condition de Robin au vecteur charge.

        @details
        Pour une condition de Robin avec température extérieure $u_E$ :

        La contribution au vecteur charge global est :
        $F[i] += \\alpha u_E$

        où $i$ est le DOF au point de bord.

        @param alpha: Coefficient de transfert thermique $\\alpha \\geq 0$ [W/(m^2.K)]
        @param u_E: Température extérieure/ambiante [K ou °C]

        @return Scalaire $\\alpha u_E$ à ajouter au vecteur charge

        @raises ValidationError: Si alpha < 0
        """
        if alpha < 0:
            raise ValidationError(f"alpha doit être >= 0, reçu {alpha}")

        return alpha * u_E


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

    logger.info("=== Test Segment P1 ===")
    coords_seg = np.array([0.0, 1.0])
    grad, length = SegmentP1.physical_gradients(coords_seg)
    logger.info(f"Longueur segment: {length}")
    logger.debug(f"Gradients physiques: {grad}")

    K = SegmentP1.local_stiffness_matrix(coords_seg, kappa=10.0)
    logger.debug(f"Matrice de rigidité (κ=10):\n{K}")

    M = SegmentP1.local_mass_matrix(coords_seg)
    logger.debug(f"Matrice de masse:\n{M}")

    F_const = SegmentP1.local_load_vector(coords_seg, source_term=100.0)
    logger.debug(f"Vecteur charge (f=100):\n{F_const}")

    F_func = SegmentP1.local_load_vector(coords_seg, lambda x: x**2)
    logger.debug(f"Vecteur charge (f=x²):\n{F_func}")

    logger.info("=== Test Point P1 ===")
    alpha_robin = PointP1.local_robin_matrix(alpha=50.0)
    logger.info(f"Contribution Robin matrice: {alpha_robin}")

    F_robin = PointP1.local_robin_load(alpha=50.0, u_E=300.0)
    logger.info(f"Contribution Robin charge: {F_robin}")

    logger.info("Tests terminés avec succès")
