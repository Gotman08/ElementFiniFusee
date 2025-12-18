"""
Tests unitaires pour les éléments finis 1D.

Vérifie la correction des éléments P1, du maillage 1D, et de l'assemblage.
"""
import pytest
import numpy as np

from src.core.fem_elements_1d import SegmentP1, PointP1
from src.mesh.mesh_1d import Mesh1D, create_uniform_mesh, create_graded_mesh
from src.core.assembly_1d import assemble_1d_system, apply_dirichlet_1d, solve_1d_system
from src.utils.exceptions import ElementError, ValidationError


@pytest.mark.unit
@pytest.mark.fast
class TestSegmentP1:
    """Tests unitaires pour l'élément SegmentP1."""

    def test_shape_functions_partition_unity(self):
        """Les fonctions de forme doivent satisfaire la partition de l'unité : sum(phi_i) = 1."""
        test_points = [-1.0, -0.5, 0.0, 0.5, 1.0]
        for xi in test_points:
            phi = SegmentP1.shape_functions(xi)
            assert np.isclose(phi.sum(), 1.0), \
                f"sum(phi) = {phi.sum()} != 1.0 pour xi={xi}"

    def test_shape_functions_nodal_values(self):
        """Propriété de Kronecker : phi_i(x_j) = delta_ij."""
        # Au nœud 1 (xi = -1)
        phi_left = SegmentP1.shape_functions(-1.0)
        assert np.allclose(phi_left, [1.0, 0.0]), \
            f"phi(-1) = {phi_left}, attendu [1, 0]"

        # Au nœud 2 (xi = +1)
        phi_right = SegmentP1.shape_functions(1.0)
        assert np.allclose(phi_right, [0.0, 1.0]), \
            f"phi(+1) = {phi_right}, attendu [0, 1]"

    def test_shape_derivatives_constant(self):
        """Les dérivées des fonctions de forme P1 doivent être constantes."""
        grad_phi = SegmentP1.shape_derivatives_ref()
        assert grad_phi.shape == (2,)
        assert np.allclose(grad_phi, [-0.5, 0.5])

    def test_jacobian_calculation(self):
        """Le Jacobien doit être J = h/2 où h est la longueur de l'élément."""
        coords = np.array([0.0, 2.0])  # h = 2.0
        J, det_J = SegmentP1.compute_jacobian(coords)
        assert np.isclose(J, 1.0), f"J = {J}, attendu 1.0"
        assert np.isclose(det_J, 1.0), f"det_J = {det_J}, attendu 1.0"

    def test_jacobian_degenerate_element(self):
        """Un élément dégénéré (longueur nulle) doit lever une exception."""
        coords = np.array([1.0, 1.0])  # Longueur nulle
        with pytest.raises(ElementError, match="dégénéré"):
            SegmentP1.compute_jacobian(coords)

    def test_physical_gradients(self):
        """Les gradients physiques doivent être dφ/dx = (dφ/dξ) / J."""
        coords = np.array([0.0, 1.0])  # h = 1.0, J = 0.5
        grad_phi, length = SegmentP1.physical_gradients(coords)

        assert np.isclose(length, 1.0), f"length = {length}, attendu 1.0"
        # dφ/dx = dφ/dξ / J = [-0.5, 0.5] / 0.5 = [-1, 1]
        assert np.allclose(grad_phi, [-1.0, 1.0]), \
            f"grad_phi = {grad_phi}, attendu [-1, 1]"

    def test_stiffness_matrix_symmetry(self):
        """La matrice de rigidité doit être symétrique."""
        coords = np.array([0.0, 1.0])
        K = SegmentP1.local_stiffness_matrix(coords, kappa=1.0)
        assert np.allclose(K, K.T), "K doit être symétrique"

    def test_stiffness_matrix_formula(self):
        """Pour P1, K^e = (κ/h) * [[1, -1], [-1, 1]]."""
        coords = np.array([0.0, 2.0])  # h = 2.0
        kappa = 10.0
        K = SegmentP1.local_stiffness_matrix(coords, kappa)

        K_expected = (kappa / 2.0) * np.array([[1, -1], [-1, 1]])
        assert np.allclose(K, K_expected), \
            f"K =\n{K}\nattendu\n{K_expected}"

    def test_stiffness_matrix_null_space(self):
        """La constante doit être dans le noyau : K·[1, 1]^T = 0."""
        coords = np.array([0.0, 0.5])
        K = SegmentP1.local_stiffness_matrix(coords, kappa=2.0)
        result = K @ np.ones(2)
        assert np.allclose(result, 0.0, atol=1e-12), \
            f"K·[1,1]^T = {result}, attendu [0,0]"

    def test_mass_matrix_symmetry(self):
        """La matrice de masse doit être symétrique."""
        coords = np.array([0.0, 1.0])
        M = SegmentP1.local_mass_matrix(coords)
        assert np.allclose(M, M.T), "M doit être symétrique"

    def test_mass_matrix_positive(self):
        """La matrice de masse doit être définie positive."""
        coords = np.array([0.0, 1.0])
        M = SegmentP1.local_mass_matrix(coords)
        # Vérifier que tous les termes diagonaux sont positifs
        assert np.all(np.diag(M) > 0), "Les termes diagonaux de M doivent être > 0"

    def test_load_vector_constant_source(self):
        """Pour un terme source constant, F^e = (f·h/2)·[1, 1]^T."""
        coords = np.array([0.0, 1.0])  # h = 1.0
        f = 100.0
        F = SegmentP1.local_load_vector(coords, source_term=f)

        # Intégrale exacte pour source constante avec P1
        F_expected = (f * 1.0 / 2.0) * np.ones(2)
        assert np.allclose(F, F_expected, rtol=1e-10), \
            f"F = {F}, attendu {F_expected}"

    def test_load_vector_variable_source(self):
        """Test avec un terme source variable f(x) = x."""
        coords = np.array([0.0, 1.0])
        F = SegmentP1.local_load_vector(coords, lambda x: x)
        # Avec quadrature de Gauss à 2 points, on doit obtenir l'intégrale exacte
        # ∫₀¹ x·φ₁ dx et ∫₀¹ x·φ₂ dx
        assert F.shape == (2,)
        assert np.all(F > 0), "F doit avoir des composantes positives pour f(x)=x"

    def test_invalid_kappa(self):
        """kappa <= 0 doit lever une exception."""
        coords = np.array([0.0, 1.0])
        with pytest.raises(ValidationError, match="kappa doit être positif"):
            SegmentP1.local_stiffness_matrix(coords, kappa=-1.0)


@pytest.mark.unit
@pytest.mark.fast
class TestPointP1:
    """Tests unitaires pour l'élément PointP1 (BC de Robin)."""

    def test_robin_matrix_positive(self):
        """alpha doit être positif ou nul."""
        alpha = 50.0
        contrib = PointP1.local_robin_matrix(alpha)
        assert contrib == alpha

    def test_robin_matrix_zero(self):
        """alpha = 0 correspond à une condition de Neumann homogène."""
        contrib = PointP1.local_robin_matrix(0.0)
        assert contrib == 0.0

    def test_robin_load_calculation(self):
        """F_robin = alpha * u_E."""
        alpha, u_E = 50.0, 300.0
        F_robin = PointP1.local_robin_load(alpha, u_E)
        assert np.isclose(F_robin, alpha * u_E)

    def test_invalid_alpha(self):
        """alpha < 0 doit lever une exception."""
        with pytest.raises(ValidationError, match="alpha doit être >= 0"):
            PointP1.local_robin_matrix(-10.0)


@pytest.mark.unit
@pytest.mark.fast
class TestMesh1D:
    """Tests unitaires pour la classe Mesh1D."""

    def test_mesh_creation(self):
        """Création d'un maillage simple."""
        nodes = np.array([0.0, 0.5, 1.0])
        mesh = Mesh1D(nodes)
        assert mesh.n_nodes == 3
        assert mesh.n_elements == 2
        assert np.isclose(mesh.L, 1.0)

    def test_mesh_elements_connectivity(self):
        """La connectivité doit être correcte."""
        nodes = np.array([0.0, 0.5, 1.0])
        mesh = Mesh1D(nodes)
        assert np.array_equal(mesh.elements[0], [0, 1])
        assert np.array_equal(mesh.elements[1], [1, 2])

    def test_get_element_coords(self):
        """Récupération des coordonnées d'un élément."""
        nodes = np.array([0.0, 0.5, 1.0])
        mesh = Mesh1D(nodes)
        coords_0 = mesh.get_element_coords(0)
        assert np.allclose(coords_0, [0.0, 0.5])

    def test_get_boundary_nodes(self):
        """Les nœuds de bord doivent être aux indices 0 et n-1."""
        nodes = np.array([0.0, 0.5, 1.0, 1.5])
        mesh = Mesh1D(nodes)
        bnodes = mesh.get_boundary_nodes()
        assert bnodes['left'] == 0
        assert bnodes['right'] == 3

    def test_unsorted_nodes_raises_error(self):
        """Les nœuds non triés doivent lever une exception."""
        nodes = np.array([0.0, 1.0, 0.5])  # Non trié
        with pytest.raises(ValidationError, match="triés en ordre croissant"):
            Mesh1D(nodes)

    def test_duplicate_nodes_raises_error(self):
        """Les nœuds dupliqués doivent lever une exception."""
        nodes = np.array([0.0, 0.5, 0.5, 1.0])  # 0.5 dupliqué
        with pytest.raises(ValidationError, match="dupliqués"):
            Mesh1D(nodes)


@pytest.mark.unit
@pytest.mark.fast
class TestUniformMesh:
    """Tests pour la génération de maillages uniformes."""

    def test_uniform_mesh_spacing(self):
        """Le maillage uniforme doit avoir un espacement constant."""
        mesh = create_uniform_mesh(L=1.0, n_elements=10)
        spacing = mesh.get_mesh_spacing()
        assert np.isclose(spacing['uniformity'], 1.0, atol=1e-10), \
            "Le maillage doit être parfaitement uniforme"

    def test_uniform_mesh_node_count(self):
        """n_elements éléments → n_elements + 1 nœuds."""
        n_elem = 20
        mesh = create_uniform_mesh(L=2.0, n_elements=n_elem)
        assert mesh.n_nodes == n_elem + 1
        assert mesh.n_elements == n_elem

    def test_uniform_mesh_endpoints(self):
        """Les nœuds extrêmes doivent être à 0 et L."""
        L = 3.14
        mesh = create_uniform_mesh(L, n_elements=50)
        assert np.isclose(mesh.nodes[0], 0.0)
        assert np.isclose(mesh.nodes[-1], L)

    def test_invalid_parameters(self):
        """Paramètres invalides doivent lever des exceptions."""
        with pytest.raises(ValidationError, match="L doit être positif"):
            create_uniform_mesh(L=-1.0, n_elements=10)

        with pytest.raises(ValidationError, match="n_elements doit être >= 1"):
            create_uniform_mesh(L=1.0, n_elements=0)


@pytest.mark.unit
@pytest.mark.fast
class TestGradedMesh:
    """Tests pour la génération de maillages raffinés."""

    def test_graded_mesh_grading_one_is_uniform(self):
        """grading = 1.0 doit produire un maillage uniforme."""
        mesh = create_graded_mesh(L=1.0, n_elements=10, grading=1.0)
        spacing = mesh.get_mesh_spacing()
        assert spacing['uniformity'] > 0.99, \
            "grading=1 doit être essentiellement uniforme"

    def test_graded_mesh_refinement_direction(self):
        """grading > 1 doit raffiner vers la droite."""
        mesh = create_graded_mesh(L=1.0, n_elements=10, grading=2.0)
        h_first = mesh.get_element_length(0)
        h_last = mesh.get_element_length(9)
        assert h_last > h_first, \
            "Pour grading>1, les éléments à droite doivent être plus grands"

    def test_graded_mesh_endpoints(self):
        """Les nœuds extrêmes doivent être à 0 et L même pour maillage gradué."""
        L = 2.5
        mesh = create_graded_mesh(L, n_elements=20, grading=1.5)
        assert np.isclose(mesh.nodes[0], 0.0)
        assert np.isclose(mesh.nodes[-1], L)


@pytest.mark.unit
class TestAssembly1D:
    """Tests d'assemblage pour les systèmes 1D."""

    def test_system_size(self):
        """Le système global doit avoir les bonnes dimensions."""
        mesh = create_uniform_mesh(1.0, 10)
        A, F = assemble_1d_system(mesh, kappa=1.0)
        assert A.shape == (11, 11)
        assert F.shape == (11,)

    def test_matrix_symmetry_without_bc(self):
        """Sans BC de Dirichlet, la matrice doit être symétrique."""
        mesh = create_uniform_mesh(1.0, 5)
        A, F = assemble_1d_system(mesh, kappa=1.0)
        assert np.allclose(A, A.T), "A doit être symétrique"

    def test_constant_solution_laplace(self):
        """Pour l'équation de Laplace (f=0), la constante est dans le noyau."""
        mesh = create_uniform_mesh(1.0, 10)
        A, F = assemble_1d_system(mesh, kappa=1.0, source_term=0.0)
        # Sans BC, A·1 doit être proche de zéro aux nœuds intérieurs
        result = A @ np.ones(mesh.n_nodes)
        # Les nœuds de bord peuvent avoir des contributions non nulles
        # mais les nœuds intérieurs devraient être proches de zéro
        assert np.allclose(result[1:-1], 0.0, atol=1e-10), \
            "A·1 devrait être zéro aux nœuds intérieurs pour Laplace"

    def test_robin_bc_adds_to_diagonal(self):
        """La BC de Robin doit ajouter alpha au terme diagonal."""
        mesh = create_uniform_mesh(1.0, 5)
        alpha = 50.0
        u_E = 300.0

        # Assembler sans Robin
        A_no_robin, F_no_robin = assemble_1d_system(mesh, kappa=1.0)

        # Assembler avec Robin à droite
        robin_bc = {'right': (alpha, u_E)}
        A_robin, F_robin = assemble_1d_system(mesh, kappa=1.0, robin_bc=robin_bc)

        # Vérifier que seul le terme diagonal du dernier nœud a changé
        dof_right = mesh.get_boundary_nodes()['right']
        diff_A = A_robin - A_no_robin
        assert np.isclose(diff_A[dof_right, dof_right], alpha), \
            f"diff_A[{dof_right},{dof_right}] = {diff_A[dof_right, dof_right]}, attendu {alpha}"

        # Vérifier la contribution au vecteur F
        diff_F = F_robin - F_no_robin
        assert np.isclose(diff_F[dof_right], alpha * u_E), \
            f"diff_F[{dof_right}] = {diff_F[dof_right]}, attendu {alpha * u_E}"

    def test_source_term_callable(self):
        """Le terme source peut être une fonction f(x)."""
        mesh = create_uniform_mesh(1.0, 10)
        A, F = assemble_1d_system(mesh, kappa=1.0, source_term=lambda x: x**2)
        # F doit être non nul et croissant
        assert np.all(F > 0), "F doit être positif pour f(x)=x² sur [0,1]"

    def test_invalid_kappa_raises_error(self):
        """kappa <= 0 doit lever une exception."""
        mesh = create_uniform_mesh(1.0, 5)
        with pytest.raises(ValidationError, match="kappa doit être positif"):
            assemble_1d_system(mesh, kappa=-1.0)


@pytest.mark.unit
class TestDirichletBC:
    """Tests pour l'application des conditions de Dirichlet."""

    def test_dirichlet_row_modification(self):
        """La ligne d'un DOF Dirichlet doit être [0,...,0,1,0,...,0]."""
        mesh = create_uniform_mesh(1.0, 10)
        A, F = assemble_1d_system(mesh, kappa=1.0, source_term=100.0)

        bc = {'left': 0.0}
        A_bc, F_bc = apply_dirichlet_1d(A, F, bc, mesh)

        # Vérifier la ligne 0
        row_0 = A_bc[0, :]
        expected_row = np.zeros(mesh.n_nodes)
        expected_row[0] = 1.0
        assert np.allclose(row_0, expected_row), \
            f"Ligne 0 = {row_0}, attendu {expected_row}"

    def test_dirichlet_rhs_value(self):
        """F_bc[i] doit être égal à la valeur imposée."""
        mesh = create_uniform_mesh(1.0, 10)
        A, F = assemble_1d_system(mesh, kappa=1.0)

        value_left = 25.0
        value_right = 100.0
        bc = {'left': value_left, 'right': value_right}
        A_bc, F_bc = apply_dirichlet_1d(A, F, bc, mesh)

        bnodes = mesh.get_boundary_nodes()
        assert np.isclose(F_bc[bnodes['left']], value_left)
        assert np.isclose(F_bc[bnodes['right']], value_right)

    def test_solution_satisfies_bc(self):
        """La solution doit satisfaire exactement les BC de Dirichlet."""
        mesh = create_uniform_mesh(1.0, 20)
        A, F = assemble_1d_system(mesh, kappa=1.0, source_term=50.0)

        u_left = 0.0
        u_right = 100.0
        bc = {'left': u_left, 'right': u_right}
        A_bc, F_bc = apply_dirichlet_1d(A, F, bc, mesh)

        U = solve_1d_system(A_bc, F_bc)

        bnodes = mesh.get_boundary_nodes()
        assert np.isclose(U[bnodes['left']], u_left, atol=1e-12), \
            f"u(0) = {U[bnodes['left']]}, attendu {u_left}"
        assert np.isclose(U[bnodes['right']], u_right, atol=1e-12), \
            f"u(L) = {U[bnodes['right']]}, attendu {u_right}"


@pytest.mark.unit
class TestLinearSolution:
    """Test avec solution linéaire exacte."""

    def test_linear_solution_exact(self):
        """Les éléments P1 doivent reproduire exactement une solution linéaire."""
        # Solution exacte : u(x) = 2*x  →  u'' = 0  →  f = 0
        # BC : u(0) = 0, u(1) = 2

        mesh = create_uniform_mesh(L=1.0, n_elements=10)
        A, F = assemble_1d_system(mesh, kappa=1.0, source_term=0.0)

        bc = {'left': 0.0, 'right': 2.0}
        A_bc, F_bc = apply_dirichlet_1d(A, F, bc, mesh)

        U = solve_1d_system(A_bc, F_bc)
        U_exact = 2.0 * mesh.nodes

        error = np.max(np.abs(U - U_exact))
        assert error < 1e-10, \
            f"Erreur max = {error:.2e}, P1 doit reproduire linéaire exactement"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
