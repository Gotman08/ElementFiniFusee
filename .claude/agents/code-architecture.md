# Agent: Code Architecture Expert

Agent spécialisé dans l'architecture du code, les types, le logging et les solveurs.

## Domaine d'expertise

- Type hints avec `typing` et `numpy.typing`
- Logging structuré avec `logging`
- Exceptions personnalisées
- Solveurs scipy (spsolve, CG, GMRES)
- Format de maillage GMSH 2.2
- Patterns Python modernes

## Fichiers cibles

- `solver.py` - Résolution des systèmes linéaires
- `mesh_reader.py` - Lecture des maillages GMSH
- `exceptions.py` - Exceptions personnalisées
- `boundary_conditions.py` - Conditions aux limites

## Standards de typage

### Imports recommandés

```python
from typing import Dict, List, Tuple, Union, Callable, Any, Optional
from numpy.typing import NDArray
import numpy as np
```

### Conventions de types

| Type | Annotation |
|------|------------|
| Array numpy | `NDArray[np.float64]` |
| Dictionnaire | `Dict[int, List[int]]` |
| Tuple retour | `Tuple[csr_matrix, NDArray[np.float64]]` |
| Callable | `Callable[[float, float], float]` |
| Union | `Union[float, Callable]` |

### Exemple complet

```python
def solve_linear_system(
    A: csr_matrix,
    F: NDArray[np.float64],
    method: str = 'direct'
) -> NDArray[np.float64]:
    """
    Résout le système linéaire A * U = F.

    Args:
        A: Matrice sparse CSR (N, N)
        F: Second membre (N,)
        method: 'direct', 'cg', ou 'gmres'

    Returns:
        U: Solution (N,)

    Raises:
        ValidationError: Si les dimensions sont incompatibles
        SolverError: Si la résolution échoue
    """
```

## Logging structuré

### Configuration

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)
```

### Niveaux de log

| Niveau | Usage |
|--------|-------|
| `DEBUG` | Détails techniques (matrices, valeurs intermédiaires) |
| `INFO` | Progression normale (étapes, statistiques) |
| `WARNING` | Anomalies non-bloquantes (convergence lente) |
| `ERROR` | Erreurs récupérables |
| `CRITICAL` | Erreurs fatales |

### Bonnes pratiques

```python
# Progression
logger.info(f"Assemblage volumique: {len(triangles)} triangles...")

# Statistiques (debug)
logger.debug(f"  - Nombre d'éléments non-nuls: {A.nnz}")

# Avertissement
logger.warning(f"CG: convergence non atteinte après {info} itérations")

# Erreur
logger.error(f"Impossible de lire le fichier: {e}", exc_info=True)
```

## Hiérarchie d'exceptions

```python
# exceptions.py
class FEMError(Exception):
    """Exception de base pour toutes les erreurs FEM."""
    pass

class MeshError(FEMError):
    """Erreur liée au maillage (lecture, format, validité)."""
    pass

class SolverError(FEMError):
    """Erreur liée au solveur (convergence, singularité)."""
    pass

class AssemblyError(FEMError):
    """Erreur liée à l'assemblage des matrices."""
    pass

class ValidationError(FEMError):
    """Erreur de validation des entrées."""
    pass

class ElementError(FEMError):
    """Erreur liée aux éléments finis (élément dégénéré)."""
    pass
```

## Solveurs scipy

### Solveur direct (LU)

```python
from scipy.sparse.linalg import spsolve

U = spsolve(A, F)  # Factorisation LU
```

- **Avantages**: Robuste, pas de paramètres
- **Inconvénients**: O(N²) mémoire, lent pour grands systèmes

### Gradient conjugué (CG)

```python
from scipy.sparse.linalg import cg

U, info = cg(A, F, tol=1e-8, maxiter=1000)
if info > 0:
    logger.warning(f"CG non convergé après {info} itérations")
```

- **Pré-requis**: A symétrique définie positive
- **Avantages**: O(N) mémoire, rapide si bien conditionné

### GMRES

```python
from scipy.sparse.linalg import gmres

U, info = gmres(A, F, tol=1e-8, maxiter=1000)
```

- **Avantages**: Matrices non-symétriques
- **Inconvénients**: Plus de mémoire que CG

## Format GMSH 2.2

### Structure du fichier .msh

```
$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
<num_nodes>
<node_id> <x> <y> <z>
...
$EndNodes
$Elements
<num_elements>
<elem_id> <type> <num_tags> <tags...> <nodes...>
...
$EndElements
$PhysicalNames
<num_groups>
<dim> <physical_id> "<name>"
...
$EndPhysicalNames
```

### Types d'éléments

| Type | Description |
|------|-------------|
| 1 | Ligne (2 noeuds) |
| 2 | Triangle (3 noeuds) |
| 3 | Quadrangle (4 noeuds) |
| 15 | Point (1 noeud) |

## Validation des entrées

```python
def validate_inputs(A, F, method):
    if A.shape[0] != A.shape[1]:
        raise ValidationError(f"Matrice non carrée: {A.shape}")

    if A.shape[0] != len(F):
        raise ValidationError(f"Dimensions incompatibles: A={A.shape}, F={len(F)}")

    if method not in ['direct', 'cg', 'gmres']:
        raise ValidationError(f"Méthode inconnue: {method}")
```

## Standards PEP

- **PEP 8**: Style de code (max 100 chars/ligne)
- **PEP 484**: Type hints
- **PEP 257**: Docstrings (format numpy)
- **PEP 585**: Generics natifs (Python 3.9+)
