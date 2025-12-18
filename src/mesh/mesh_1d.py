"""
@file mesh_1d.py
@brief Structures de maillage 1D pour domaines intervalles
@author ElementFiniFusee
@date 2025

@details
Ce module implémente les structures de données de maillage pour les domaines 1D.
Il fournit une classe légère `Mesh1D` pour représenter un maillage sur $[0, L]$
et des fonctions de génération de maillages uniformes et raffinés.

Le maillage 1D stocke :
- Les coordonnées des nœuds (array 1D)
- La connectivité des éléments (implicite : élément i connecte nœuds [i, i+1])
- Les nœuds de bord (gauche et droite)

Fonctions de génération disponibles :
- `create_uniform_mesh`: Maillage uniforme avec espacement constant
- `create_graded_mesh`: Maillage raffiné avec progression géométrique
"""
import logging
import numpy as np
from typing import Dict
from numpy.typing import NDArray

from src.utils.exceptions import ValidationError

logger = logging.getLogger(__name__)


class Mesh1D:
    """
    @brief Maillage 1D pour domaine intervalle [0, L].

    @details
    Structure de données légère pour un maillage 1D. Stocke les coordonnées
    des nœuds et génère automatiquement la connectivité élément-nœud.

    Conventions :
    - Nœuds numérotés de 0 à n_nodes-1
    - Élément i connecte les nœuds [i, i+1]
    - Nœud de gauche : indice 0
    - Nœud de droite : indice n_nodes-1

    Attributs :
        nodes: Array 1D des coordonnées x des nœuds
        n_nodes: Nombre de nœuds
        n_elements: Nombre d'éléments (= n_nodes - 1)
        elements: Array 2D de connectivité, forme (n_elements, 2)
        L: Longueur totale du domaine
    """

    def __init__(self, nodes: NDArray[np.float64]):
        """
        @brief Initialise le maillage depuis les coordonnées des nœuds.

        @param nodes: Array 1D des coordonnées x, doit être trié en ordre croissant

        @raises ValidationError: Si nodes n'est pas 1D ou n'est pas trié
        """
        if nodes.ndim != 1:
            raise ValidationError(
                f"nodes doit être un array 1D, reçu forme {nodes.shape}"
            )

        if len(nodes) < 2:
            raise ValidationError(
                f"Le maillage doit avoir au moins 2 nœuds, reçu {len(nodes)}"
            )

        # Vérifier que les nœuds sont triés
        if not np.all(nodes[1:] >= nodes[:-1]):
            raise ValidationError(
                "Les nœuds doivent être triés en ordre croissant"
            )

        # Vérifier qu'il n'y a pas de nœuds dupliqués
        if np.any(np.diff(nodes) == 0):
            raise ValidationError(
                "Le maillage contient des nœuds dupliqués (espacement nul)"
            )

        self.nodes = nodes.copy()
        self.n_nodes = len(nodes)
        self.n_elements = self.n_nodes - 1

        # Construire la connectivité : élément i = [nœud i, nœud i+1]
        self.elements = np.array([[i, i+1] for i in range(self.n_elements)])

        self.L = nodes[-1] - nodes[0]

        logger.debug(
            f"Maillage 1D créé : {self.n_nodes} nœuds, {self.n_elements} éléments, "
            f"L={self.L:.4f}"
        )

    def get_element_coords(self, elem_id: int) -> NDArray[np.float64]:
        """
        @brief Retourne les coordonnées physiques des nœuds d'un élément.

        @param elem_id: Identifiant de l'élément (0 à n_elements-1)

        @return Array de forme (2,) contenant [x1, x2]

        @raises ValidationError: Si elem_id est hors limites
        """
        if not 0 <= elem_id < self.n_elements:
            raise ValidationError(
                f"elem_id invalide : {elem_id}. "
                f"Doit être entre 0 et {self.n_elements-1}"
            )

        node_ids = self.elements[elem_id]
        return self.nodes[node_ids]

    def get_element_length(self, elem_id: int) -> float:
        """
        @brief Retourne la longueur d'un élément.

        @param elem_id: Identifiant de l'élément (0 à n_elements-1)

        @return Longueur h = x2 - x1

        @raises ValidationError: Si elem_id est hors limites
        """
        coords = self.get_element_coords(elem_id)
        return coords[1] - coords[0]

    def get_boundary_nodes(self) -> Dict[str, int]:
        """
        @brief Retourne les indices des nœuds de bord.

        @return Dictionnaire {'left': index_gauche, 'right': index_droit}

        @example
        >>> mesh = Mesh1D(np.array([0.0, 0.5, 1.0]))
        >>> bnodes = mesh.get_boundary_nodes()
        >>> # Retourne {'left': 0, 'right': 2}
        """
        return {
            'left': 0,
            'right': self.n_nodes - 1
        }

    def get_mesh_spacing(self) -> Dict[str, float]:
        """
        @brief Calcule les statistiques d'espacement du maillage.

        @return Dictionnaire contenant :
            - 'h_min': Plus petit espacement
            - 'h_max': Plus grand espacement
            - 'h_mean': Espacement moyen
            - 'uniformity': Ratio h_min/h_max (1.0 = parfaitement uniforme)
        """
        element_lengths = np.array([self.get_element_length(i)
                                   for i in range(self.n_elements)])

        h_min = element_lengths.min()
        h_max = element_lengths.max()
        h_mean = element_lengths.mean()
        uniformity = h_min / h_max if h_max > 0 else 0.0

        return {
            'h_min': h_min,
            'h_max': h_max,
            'h_mean': h_mean,
            'uniformity': uniformity
        }

    def __repr__(self) -> str:
        """@brief Représentation textuelle du maillage."""
        spacing = self.get_mesh_spacing()
        return (
            f"Mesh1D(n_nodes={self.n_nodes}, n_elements={self.n_elements}, "
            f"L={self.L:.4f}, h_mean={spacing['h_mean']:.4e}, "
            f"uniformity={spacing['uniformity']:.3f})"
        )


def create_uniform_mesh(L: float, n_elements: int) -> Mesh1D:
    """
    @brief Crée un maillage uniforme sur l'intervalle [0, L].

    @details
    Génère un maillage avec espacement constant $h = L / n_{elements}$.
    Les nœuds sont placés à :
    $x_i = i \\cdot h$ pour $i = 0, 1, ..., n_{elements}$

    @param L: Longueur du domaine [m], doit être > 0
    @param n_elements: Nombre d'éléments, doit être >= 1

    @return Objet Mesh1D avec espacement uniforme

    @raises ValidationError: Si L <= 0 ou n_elements < 1

    @example
    >>> mesh = create_uniform_mesh(L=1.0, n_elements=10)
    >>> # Crée 11 nœuds de 0.0 à 1.0 avec h=0.1
    """
    if L <= 0:
        raise ValidationError(f"L doit être positif, reçu {L}")

    if n_elements < 1:
        raise ValidationError(f"n_elements doit être >= 1, reçu {n_elements}")

    nodes = np.linspace(0, L, n_elements + 1)

    logger.info(f"Maillage uniforme créé : L={L}, n_elem={n_elements}, h={L/n_elements:.4e}")

    return Mesh1D(nodes)


def create_graded_mesh(L: float, n_elements: int, grading: float = 1.0) -> Mesh1D:
    """
    @brief Crée un maillage avec raffinement progressif (progression géométrique).

    @details
    Génère un maillage avec espacement qui varie géométriquement.
    Le rapport de progression $r$ est calculé pour que :
    $r^{n_{elements}} = grading$

    - Si grading > 1 : Raffinement vers x=L (utile pour BC de Robin)
    - Si grading < 1 : Raffinement vers x=0
    - Si grading = 1 : Maillage uniforme

    L'espacement entre nœuds suit : $\\Delta x_i = r \\cdot \\Delta x_{i-1}$

    @param L: Longueur du domaine [m], doit être > 0
    @param n_elements: Nombre d'éléments, doit être >= 1
    @param grading: Facteur de raffinement, typiquement dans [0.5, 2.0]
        - grading = 1.0 : uniforme
        - grading > 1.0 : raffiné à droite
        - grading < 1.0 : raffiné à gauche

    @return Objet Mesh1D avec espacement gradué

    @raises ValidationError: Si paramètres invalides

    @example
    >>> mesh = create_graded_mesh(L=1.0, n_elements=10, grading=2.0)
    >>> # Crée maillage raffiné près de x=1.0
    >>> spacing = mesh.get_mesh_spacing()
    >>> print(f"h_min={spacing['h_min']:.4e}, h_max={spacing['h_max']:.4e}")
    """
    if L <= 0:
        raise ValidationError(f"L doit être positif, reçu {L}")

    if n_elements < 1:
        raise ValidationError(f"n_elements doit être >= 1, reçu {n_elements}")

    if grading <= 0:
        raise ValidationError(f"grading doit être positif, reçu {grading}")

    # Cas particulier : grading = 1 → maillage uniforme
    if abs(grading - 1.0) < 1e-10:
        logger.info("grading ≈ 1.0, création d'un maillage uniforme")
        return create_uniform_mesh(L, n_elements)

    # Ratio de progression géométrique : r^n = grading
    ratio = grading ** (1.0 / n_elements)

    # Générer la séquence en coordonnées normalisées [0, 1]
    # s_i = sum(r^j for j=0..i) / sum(r^j for j=0..n)
    # Formule de la somme géométrique : sum(r^j, j=0..n) = (r^(n+1) - 1)/(r - 1)

    if abs(ratio - 1.0) < 1e-10:
        # Cas limite : ratio ≈ 1
        s_norm = np.linspace(0, 1, n_elements + 1)
    else:
        # Somme géométrique
        indices = np.arange(n_elements + 1)
        s = np.array([ratio**i for i in indices])
        s_norm = (s - s[0]) / (s[-1] - s[0])  # Normaliser à [0, 1]

    # Mapper vers [0, L]
    nodes = L * s_norm

    logger.info(
        f"Maillage gradué créé : L={L}, n_elem={n_elements}, "
        f"grading={grading:.3f}, ratio={ratio:.4f}"
    )

    mesh = Mesh1D(nodes)
    spacing = mesh.get_mesh_spacing()
    logger.debug(
        f"Espacement : h_min={spacing['h_min']:.4e}, "
        f"h_max={spacing['h_max']:.4e}, uniformity={spacing['uniformity']:.3f}"
    )

    return mesh


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

    logger.info("=== Test Maillage Uniforme ===")
    mesh_uni = create_uniform_mesh(L=1.0, n_elements=10)
    logger.info(mesh_uni)
    logger.debug(f"Nœuds : {mesh_uni.nodes}")
    logger.debug(f"Éléments (connectivité) :\n{mesh_uni.elements}")

    logger.info("\n=== Test Maillage Gradué (raffiné à droite) ===")
    mesh_grad_right = create_graded_mesh(L=1.0, n_elements=10, grading=2.0)
    logger.info(mesh_grad_right)
    logger.debug(f"Nœuds : {mesh_grad_right.nodes}")

    logger.info("\n=== Test Maillage Gradué (raffiné à gauche) ===")
    mesh_grad_left = create_graded_mesh(L=1.0, n_elements=10, grading=0.5)
    logger.info(mesh_grad_left)
    logger.debug(f"Nœuds : {mesh_grad_left.nodes}")

    logger.info("\n=== Test Accès aux Éléments ===")
    coords_elem_0 = mesh_uni.get_element_coords(0)
    logger.info(f"Élément 0 : nœuds à {coords_elem_0}")

    bnodes = mesh_uni.get_boundary_nodes()
    logger.info(f"Nœuds de bord : {bnodes}")

    logger.info("\nTests terminés avec succès")
