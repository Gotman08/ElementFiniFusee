# Agent: Testing & Validation Expert

Agent spécialisé dans les tests unitaires, la validation numérique et l'intégration continue.

## Domaine d'expertise

- Tests pytest avec fixtures
- Tests de convergence en h (raffinement de maillage)
- Solutions manufacturées (Method of Manufactured Solutions)
- Validation physique avec nombres adimensionnels
- Configuration CI/CD GitHub Actions
- Couverture de code

## Fichiers cibles

- `tests/` - Dossier de tests (à créer)
- `pytest.ini` ou `pyproject.toml` - Configuration pytest
- `.github/workflows/ci.yml` - CI/CD

## Structure des tests

```
tests/
├── __init__.py
├── conftest.py           # Fixtures partagées
├── test_fem_elements.py  # Tests éléments finis
├── test_mesh_reader.py   # Tests lecture maillage
├── test_assembly.py      # Tests assemblage
├── test_solver.py        # Tests solveurs
├── test_convergence.py   # Tests convergence h
└── test_physics.py       # Tests validation physique
```

## Fixtures pytest

### conftest.py

```python
import pytest
import numpy as np
from mesh_reader import Mesh

@pytest.fixture
def reference_triangle():
    """Triangle de référence pour tests."""
    return np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)

@pytest.fixture
def unit_square_mesh():
    """Maillage carré unitaire simple."""
    mesh = Mesh()
    # 4 noeuds, 2 triangles
    mesh.nodes = {
        1: [0, 0, 0], 2: [1, 0, 0],
        3: [1, 1, 0], 4: [0, 1, 0]
    }
    mesh.elements = {
        1: {'type': 2, 'tags': [1, 1], 'nodes': [1, 2, 3]},
        2: {'type': 2, 'tags': [1, 1], 'nodes': [1, 3, 4]}
    }
    return mesh

@pytest.fixture
def simple_system():
    """Système linéaire simple pour tests solveur."""
    from scipy.sparse import csr_matrix
    A = csr_matrix([[4, -1, 0], [-1, 4, -1], [0, -1, 4]])
    F = np.array([1, 2, 1])
    return A, F
```

## Tests unitaires

### test_fem_elements.py

```python
import pytest
import numpy as np
from fem_elements import TriangleP1, EdgeP1
from exceptions import ValidationError, ElementError

class TestTriangleP1:
    def test_shape_functions_sum_to_one(self, reference_triangle):
        """Partition de l'unité: Σφ_i = 1"""
        for xi, eta in [(0.25, 0.25), (0.5, 0.25), (1/3, 1/3)]:
            phi = TriangleP1.shape_functions(xi, eta)
            assert np.isclose(phi.sum(), 1.0)

    def test_reference_triangle_area(self, reference_triangle):
        """Aire du triangle de référence = 0.5"""
        _, area = TriangleP1.physical_gradients(reference_triangle)
        assert np.isclose(area, 0.5)

    def test_stiffness_matrix_symmetric(self, reference_triangle):
        """Matrice de rigidité symétrique"""
        K = TriangleP1.local_stiffness_matrix(reference_triangle, kappa=1.0)
        assert np.allclose(K, K.T)

    def test_stiffness_matrix_row_sum_zero(self, reference_triangle):
        """Somme des lignes = 0 (Laplacien)"""
        K = TriangleP1.local_stiffness_matrix(reference_triangle, kappa=1.0)
        assert np.allclose(K.sum(axis=1), 0)

    def test_invalid_coords_shape(self):
        """Erreur si coords n'a pas la bonne forme"""
        bad_coords = np.array([[0, 0], [1, 0]])  # 2 points au lieu de 3
        with pytest.raises(ValidationError):
            TriangleP1.compute_jacobian(bad_coords)

    def test_degenerate_element(self):
        """Erreur si élément dégénéré (aire nulle)"""
        degenerate = np.array([[0, 0], [1, 0], [2, 0]])  # Points alignés
        with pytest.raises(ElementError):
            TriangleP1.physical_gradients(degenerate)

    def test_negative_kappa(self, reference_triangle):
        """Erreur si kappa <= 0"""
        with pytest.raises(ValidationError):
            TriangleP1.local_stiffness_matrix(reference_triangle, kappa=-1.0)
```

## Tests de convergence

### test_convergence.py

```python
import pytest
import numpy as np

class TestConvergenceH:
    """Tests de convergence en raffinement de maillage."""

    @pytest.mark.parametrize("n_refine", [2, 4, 8, 16])
    def test_laplacian_convergence_rate(self, n_refine):
        """
        Test de convergence pour -Δu = f sur [0,1]²
        Solution exacte: u = sin(πx)sin(πy)
        Ordre attendu: O(h²) pour P1
        """
        # Générer maillage h = 1/n_refine
        # Résoudre
        # Calculer erreur L2
        # Vérifier ordre de convergence ≈ 2
        pass

    def test_manufactured_solution(self):
        """
        Method of Manufactured Solutions:
        1. Choisir u_exact
        2. Calculer f = -κΔu_exact
        3. Résoudre avec f
        4. Comparer u_h avec u_exact
        """
        pass
```

## Tests physiques

### test_physics.py

```python
import pytest
import numpy as np

class TestPhysicalValidation:
    """Validation des modèles physiques."""

    def test_recovery_temperature_positive(self):
        """T_récup > T_∞ toujours"""
        from reentry_profile import compute_recovery_temperature
        T_inf = 250  # K
        for mach in [1, 5, 10, 20]:
            T_r = compute_recovery_temperature(T_inf, mach)
            assert T_r > T_inf

    def test_nusselt_positive(self):
        """Nu > 0 pour Re > 0"""
        from parametric_study import compute_nusselt
        for Re in [1e3, 1e5, 1e7]:
            Nu = compute_nusselt(Re, Pr=0.7)
            assert Nu > 0

    def test_convection_coefficient_range(self):
        """α dans plage réaliste [10, 500] W/m²K"""
        from parametric_study import compute_alpha
        for V in [1000, 3000, 5000]:
            alpha = compute_alpha(V, altitude=30000)
            assert 10 <= alpha <= 500
```

## Configuration CI/CD

### .github/workflows/ci.yml

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10', '3.11']

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run tests
      run: pytest --cov=. --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### pytest.ini

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
addopts = -v --tb=short
markers =
    slow: marks tests as slow
    convergence: marks convergence tests
```

## Métriques de qualité

| Métrique | Cible |
|----------|-------|
| Couverture | > 80% |
| Tests passants | 100% |
| Temps CI | < 5 min |
| Tests de convergence | Ordre h² vérifié |

## Checklist de validation

- [ ] Tests unitaires pour chaque fonction publique
- [ ] Tests de bord (entrées invalides, cas limites)
- [ ] Tests de convergence en h
- [ ] Validation avec solution manufacturée
- [ ] Tests de non-régression
- [ ] CI/CD configuré et fonctionnel
