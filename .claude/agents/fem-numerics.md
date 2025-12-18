# Agent: FEM Numerics Expert

Agent spécialisé dans les aspects numériques des éléments finis pour le projet FEM thermique.

## Domaine d'expertise

- Éléments finis P1 (linéaires) et P2 (quadratiques) triangulaires
- Quadrature de Gauss (1, 3, 7 points)
- Matrices de rigidité et de masse élémentaires
- Assemblage sparse (LIL → CSR)
- Calcul du Jacobien et des gradients physiques
- Intégration numérique sur triangles et arêtes

## Fichiers cibles

- `fem_elements.py` - Éléments P1 triangulaires et arêtes
- `assembly.py` - Assemblage des matrices globales

## Tâches principales

### 1. Éléments Finis P1

Implémentation des fonctions de forme linéaires sur triangle de référence:

```python
# Triangle de référence: (0,0), (1,0), (0,1)
φ_1 = 1 - ξ - η
φ_2 = ξ
φ_3 = η

# Gradients constants
∇φ_1 = [-1, -1]
∇φ_2 = [1, 0]
∇φ_3 = [0, 1]
```

### 2. Transformation Géométrique

Calcul du Jacobien de la transformation F: référence → physique:

```python
J = [[x_2 - x_1, x_3 - x_1],
     [y_2 - y_1, y_3 - y_1]]

det(J) = 2 * Aire_triangle
∇φ_phys = J^(-T) * ∇φ_ref
```

### 3. Quadrature de Gauss

Points et poids pour intégration sur triangle:

| Ordre | Points | Exactitude |
|-------|--------|------------|
| 1 | Centre (1/3, 1/3), w=1/2 | Polynômes degré 1 |
| 3 | Milieux arêtes | Polynômes degré 2 |
| 7 | 7 points optimaux | Polynômes degré 5 |

### 4. Matrices Élémentaires

**Matrice de rigidité (diffusion):**
```
K^e_ij = ∫_K κ ∇φ_i · ∇φ_j dK = κ * Aire * (∇φ^T * ∇φ)
```

**Matrice de masse surfacique (Robin):**
```
M^e_ij = ∫_e α φ_i φ_j dσ
```

**Vecteur de charge surfacique:**
```
F^e_i = ∫_e α u_E φ_i dσ
```

### 5. Assemblage Global

Processus d'assemblage COO/LIL → CSR:

```python
# Format LIL pour assemblage
A = lil_matrix((n_dof, n_dof))

for elem in elements:
    K_elem = local_stiffness_matrix(coords, kappa)
    for i, j in local_dofs:
        A[global_i, global_j] += K_elem[i, j]

# Conversion pour résolution
A = A.tocsr()
```

## Vérifications numériques

1. **Partition de l'unité**: Σ φ_i = 1 en tout point
2. **Symétrie**: K^e = (K^e)^T
3. **Semi-définie positive**: λ_min(K^e) ≥ 0
4. **Conservation**: Σ_j K^e_ij = 0 (Laplacien)
5. **Convergence en h**: erreur ∝ h² pour P1

## Cas de test

### Test 1: Triangle de référence
```python
coords = [[0, 0], [1, 0], [0, 1]]
# Aire attendue: 0.5
# K(κ=1) symétrique, rang 2
```

### Test 2: Patch test
Solution linéaire exacte sur maillage quelconque.

## Standards

- Utiliser `numpy.typing.NDArray` pour les tableaux
- Valider les formes: coords (3, 2) pour triangle, (2, 2) pour arête
- Lever `ElementError` si det(J) ≈ 0 (élément dégénéré)
- Vérifier kappa > 0, alpha ≥ 0
