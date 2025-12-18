# Analyse Thermique de Fusée par Éléments Finis

Étude paramétrique de l'influence de la vitesse sur la température d'une coque de fusée lors de la rentrée atmosphérique.

## 📋 Description du Projet

Ce projet implémente une **méthode des éléments finis P1** pour résoudre le problème stationnaire de la thermique avec conditions aux limites de Robin sur la surface extérieure de la fusée.

### Problème Mathématique

On cherche la température `u(X)` telle que :

```
-div(κ ∇u) = 0          dans Ω (paroi)
u = u_D                 sur Γ_D (base, condition de Dirichlet)
-κ ∂u/∂n = α(u - u_E)   sur Γ_F (surface extérieure, condition de Robin)
-κ ∂u/∂n = 0            sur Γ_N (surface intérieure, condition de Neumann)
```

### Termes de Bord Essentiels

**L'implémentation correcte des conditions de Robin est cruciale** :
- Matrice de masse surfacique : `A[i,j] += ∫_{Γ_F} α φ_i φ_j dσ`
- Vecteur de charge surfacique : `F[i] += ∫_{Γ_F} α u_E φ_i dσ`

### Modèles Aérothermiques

**Coefficient de convection** α(V) :
- Corrélation turbulente : `Nu = 0.037 Re^0.8 Pr^(1/3)`
- `α = Nu · k / L`

**Température extérieure** u_E(V) :
- Température de récupération : `u_E = T_∞ + r · V² / (2 c_p)`
- Facteur de récupération turbulent : r = 0.89

## 🗂️ Structure du Code

```
ElementFiniFusee/
├── mesh_reader.py              # Lecture de maillages GMSH
├── fem_elements.py             # Éléments P1 (triangles et arêtes)
├── assembly.py                 # Assemblage avec termes de Robin ⭐
├── boundary_conditions.py      # Application des CL de Dirichlet
├── solver.py                   # Résolution du système linéaire
├── parametric_study.py         # Boucle d'étude en vitesse
├── visualization.py            # Graphiques et exports
├── main.py                     # Script principal
├── rocket_geometry.geo         # Géométrie GMSH
├── requirements.txt            # Dépendances Python
└── README.md                   # Ce fichier
```

## 🚀 Installation

### Prérequis

1. **Python 3.8+**
2. **GMSH** (générateur de maillage)
   - Windows : Télécharger depuis [gmsh.info](https://gmsh.info)
   - Linux : `sudo apt install gmsh`
   - macOS : `brew install gmsh`

### Dépendances Python

```bash
pip install -r requirements.txt
```

## 📐 Génération du Maillage

Avant d'exécuter l'étude, générer le maillage avec GMSH :

```bash
gmsh -2 rocket_geometry.geo -o rocket_mesh.msh
```

Alternative (interface graphique) :
1. Ouvrir GMSH
2. File → Open → `rocket_geometry.geo`
3. Mesh → 2D
4. File → Export → `rocket_mesh.msh` (format MSH2 ASCII)

## 🔧 Utilisation

### Exécution de l'Étude Complète

```bash
python main.py
```

### Sorties Générées

Le dossier `resultats/` contiendra :
- `T_max_vs_velocity.png` : **Courbe T_max(V)** (résultat principal)
- `temperature_field_critical.png` : Champ de température au cas critique
- `temperature_fields_comparison.png` : Comparaison de plusieurs vitesses
- `results_parametric_study.csv` : Données numériques exportées

### Personnalisation

Modifier dans [main.py](main.py:43-53) :
```python
V_min = 1000.0          # Vitesse minimale (m/s)
V_max = 5000.0          # Vitesse maximale (m/s)
n_velocities = 15       # Nombre de points
T_base = 300.0          # Température à la base (K)
```

Modifier les lois physiques dans [parametric_study.py](parametric_study.py:18-38) :
```python
RHO_inf = 0.02          # Densité air (kg/m³)
T_inf = 230.0           # Température ambiante (K)
KAPPA_material = 160.0  # Conductivité thermique (W/m·K)
```

## 📊 Interprétation des Résultats

### Graphique Principal : T_max(V)

Ce graphique montre l'évolution de la **température maximale** en fonction de la vitesse. C'est le résultat attendu d'une étude d'ingénieur.

**Observations typiques :**
- Croissance quadratique de T_max avec V (échauffement cinétique ∝ V²)
- Zone critique : ogive de la fusée (point d'arrêt)
- Limite matériau : vérifier T_max < T_fusion

### Validation Physique

**Cas limites à vérifier :**
1. α → 0 (paroi isolée) : T uniforme ≈ T_base
2. α → ∞ (Dirichlet) : T_surface = u_E
3. V = 0 : T_max ≈ T_base (pas de convection forcée)

## 🧪 Tests Unitaires

Chaque module possède un `if __name__ == '__main__'` pour tests isolés :

```bash
python fem_elements.py    # Test des fonctions de forme
python solver.py          # Test sur problème 1D
```

## 📚 Références Théoriques

1. **Formulation variationnelle** : Chapitre 3 du cours
2. **Conditions de Robin** : Page 3, équation (iii)
3. **Assemblage des termes de bord** : Application directe de la formulation faible

## ⚠️ Points Critiques

### Matrice de Masse Surfacique

**Sans les termes de Robin, la simulation est fausse !**

Le code [assembly.py](assembly.py:66-101) implémente correctement :
```python
# Ligne 89-101 : Assemblage Robin
M_elem = EdgeP1.local_mass_matrix(coords, alpha)
F_elem = EdgeP1.local_load_vector(coords, alpha, u_E)
# ... assemblage dans A et F
```

### Vérification

Comparer avec cas Neumann homogène (α=0) : la température doit être uniforme si pas de terme source.

## 🤝 Contribution

Pour améliorer le projet :
1. Implémenter des éléments d'ordre supérieur (P2, P3)
2. Ajouter un terme source volumique (dissipation interne)
3. Passer en 3D (tétraèdres)
4. Problème transitoire (évolution temporelle)

## 📄 Licence

Projet académique - Libre d'utilisation pour l'enseignement

---

**Auteur** : Claude
**Date** : 2025
**Cours** : Méthode des Éléments Finis
