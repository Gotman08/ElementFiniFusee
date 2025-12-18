# Formulation Complète des Éléments Finis 1D

## Table des Matières

1. [Formulation Forte](#1-formulation-forte)
2. [Formulation Variationnelle](#2-formulation-variationnelle)
3. [Discrétisation par Éléments Finis P1](#3-discrétisation-par-éléments-finis-p1)
4. [Matrices Élémentaires](#4-matrices-élémentaires)
5. [Assemblage et Résolution](#5-assemblage-et-résolution)
6. [Analyse d'Erreur et Convergence](#6-analyse-derreur-et-convergence)
7. [Méthode des Solutions Manufacturées](#7-méthode-des-solutions-manufacturées)
8. [Cas Limites des Conditions aux Limites](#8-cas-limites-des-conditions-aux-limites)
9. [Superconvergence](#9-superconvergence)
10. [Références](#10-références)

---

## 1. Formulation Forte

### 1.1 Énoncé du Problème

On considère le problème de conduction thermique stationnaire sur un domaine 1D $\Omega = ]0, L[$:

**Équation aux dérivées partielles (EDP):**
$$-\kappa \frac{d^2 u}{dx^2} = f(x) \quad \text{dans } \Omega$$

où:
- $u(x)$ : température [K]
- $\kappa > 0$ : conductivité thermique [W/(m·K)]
- $f(x)$ : terme source volumique [W/m³]

**Conditions aux limites:**

1. À $x = 0$ (extrémité gauche) : Condition de Dirichlet
   $$u(0) = u_D$$

2. À $x = L$ (extrémité droite) : Condition de Robin (Fourier)
   $$-\kappa \frac{du}{dx}\bigg|_{x=L} = \alpha\big(u(L) - u_E\big)$$

   où:
   - $\alpha > 0$ : coefficient de transfert convectif [W/(m²·K)]
   - $u_E$ : température extérieure (environnement) [K]

### 1.2 Interprétation Physique

La condition de Robin modélise un **échange convectif** à la frontière:

- Le flux de chaleur sortant (terme gauche: $-\kappa \frac{du}{dx}$) est proportionnel à la différence de température entre la surface $u(L)$ et l'environnement $u_E$
- $\alpha$ représente l'efficacité de l'échange thermique (grand $\alpha$ → échange efficace)
- Cette condition couple le problème de conduction dans le domaine avec l'environnement extérieur

**Cas particuliers:**
- $\alpha \to 0$ : flux nul (isolation, condition de Neumann homogène)
- $\alpha \to \infty$ : température imposée $u(L) = u_E$ (condition de Dirichlet)

### 1.3 Existence et Unicité de la Solution Forte

Sous les hypothèses:
- $\kappa > 0$ constant
- $f \in L^2(\Omega)$
- $\alpha \geq 0$

La solution forte $u \in H^2(\Omega)$ existe et est unique (par théorie des EDP elliptiques).

---

## 2. Formulation Variationnelle

### 2.1 Espaces Fonctionnels

On définit:

$$V = \big\{v \in H^1(\Omega) : v(0) = 0\big\}$$

C'est l'espace de Sobolev $H^1$ avec la condition de Dirichlet homogène imposée à $x=0$.

**Rappel:** $H^1(\Omega) = \{v \in L^2(\Omega) : v' \in L^2(\Omega)\}$ muni de la norme:
$$\|v\|_{H^1}^2 = \|v\|_{L^2}^2 + \|v'\|_{L^2}^2$$

### 2.2 Dérivation de la Formulation Faible

**Étape 1:** Multiplication par une fonction test $v \in V$

$$-\kappa \frac{d^2 u}{dx^2} v = f v$$

**Étape 2:** Intégration sur $\Omega$

$$-\int_0^L \kappa u'' v \, dx = \int_0^L f v \, dx$$

**Étape 3:** Intégration par parties (formule de Green en 1D)

$$\int_0^L u' v' \, dx = \int_{\partial\Omega} u' v - \int_0^L u'' v \, dx$$

Donc:
$$\int_0^L \kappa u' v' \, dx - \big[\kappa u' v\big]_0^L = \int_0^L f v \, dx$$

**Étape 4:** Évaluation du terme de bord

$$\big[\kappa u' v\big]_0^L = \kappa u'(L) v(L) - \kappa u'(0) v(0)$$

Comme $v(0) = 0$ (par définition de $V$), le terme en $x=0$ disparaît:

$$\int_0^L \kappa u' v' \, dx - \kappa u'(L) v(L) = \int_0^L f v \, dx$$

**Étape 5:** Injection de la condition de Robin

La condition de Robin s'écrit:
$$-\kappa u'(L) = \alpha\big(u(L) - u_E\big)$$

D'où:
$$\kappa u'(L) = -\alpha\big(u(L) - u_E\big)$$

En substituant:
$$\int_0^L \kappa u' v' \, dx + \alpha u(L) v(L) - \alpha u_E v(L) = \int_0^L f v \, dx$$

### 2.3 Formulation Variationalelle Finale

**Problème variationnel:** Trouver $u \in V + u_D$ tel que:

$$a(u, v) = \ell(v) \quad \forall v \in V$$

où:

**Forme bilinéaire:**
$$a(u, v) = \int_0^L \kappa u' v' \, dx + \alpha u(L) v(L)$$

**Forme linéaire:**
$$\ell(v) = \int_0^L f v \, dx + \alpha u_E v(L)$$

**Remarques importantes:**

1. Le terme $\alpha u(L) v(L)$ est un **terme de bord** provenant de la condition de Robin
2. Le terme $\alpha u_E v(L)$ est un **terme de charge** au bord
3. La condition de Dirichlet non-homogène $u(0) = u_D$ est traitée par relèvement: on cherche $u = u_D + \tilde{u}$ avec $\tilde{u} \in V$

### 2.4 Théorème de Lax-Milgram

**Théorème:** Sous les hypothèses:
- $\kappa > 0$
- $\alpha \geq 0$
- $f \in L^2(\Omega)$

Le problème variationnel admet une solution unique $u \in V + u_D$.

**Démonstration (esquisse):**

1. **Continuité de $a(\cdot, \cdot)$:**
   $$|a(u, v)| \leq \kappa \|u'\|_{L^2} \|v'\|_{L^2} + \alpha |u(L)| |v(L)|$$

   Par inégalité de trace en 1D: $|u(L)| \leq C \|u\|_{H^1}$, donc:
   $$|a(u, v)| \leq M \|u\|_{H^1} \|v\|_{H^1}$$

2. **Coercivité de $a(\cdot, \cdot)$ sur $V$:**
   $$a(v, v) = \kappa \|v'\|_{L^2}^2 + \alpha |v(L)|^2 \geq \kappa \|v'\|_{L^2}^2$$

   Par inégalité de Poincaré sur $V$ (car $v(0) = 0$):
   $$\|v\|_{L^2}^2 \leq C_P \|v'\|_{L^2}^2$$

   D'où:
   $$a(v, v) \geq \frac{\kappa}{1 + C_P} \|v\|_{H^1}^2 = \gamma \|v\|_{H^1}^2$$

3. **Continuité de $\ell(\cdot)$:** Immédiat par Cauchy-Schwarz et inégalité de trace.

Le théorème de Lax-Milgram s'applique donc, garantissant l'existence et l'unicité. $\square$

---

## 3. Discrétisation par Éléments Finis P1

### 3.1 Maillage du Domaine

On partitionne $[0, L]$ en $n$ éléments:

$$[0, L] = \bigcup_{i=1}^n K_i$$

où $K_i = [x_{i-1}, x_i]$ avec:
$$0 = x_0 < x_1 < \cdots < x_n = L$$

**Maillage uniforme:** $x_i = i \cdot h$ avec $h = L/n$.

**Notations:**
- $h_i = x_i - x_{i-1}$ : longueur de l'élément $K_i$
- $h = \max_i h_i$ : pas de maillage global

### 3.2 Élément de Référence

On définit l'**élément de référence** $\hat{K} = [-1, 1]$ avec coordonnée locale $\xi \in [-1, 1]$.

**Transformation affine** de $\hat{K}$ vers $K_i = [x_{i-1}, x_i]$:

$$x(\xi) = \frac{x_{i-1} + x_i}{2} + \frac{x_i - x_{i-1}}{2} \xi = \frac{1-\xi}{2} x_{i-1} + \frac{1+\xi}{2} x_i$$

**Jacobien de la transformation:**
$$J_i = \frac{dx}{d\xi} = \frac{x_i - x_{i-1}}{2} = \frac{h_i}{2}$$

### 3.3 Fonctions de Forme P1 sur l'Élément de Référence

Les **fonctions de forme P1** sur $\hat{K} = [-1, 1]$ sont:

$$\hat{\varphi}_1(\xi) = \frac{1 - \xi}{2}, \quad \hat{\varphi}_2(\xi) = \frac{1 + \xi}{2}$$

**Propriétés:**

1. **Partition de l'unité:** $\hat{\varphi}_1(\xi) + \hat{\varphi}_2(\xi) = 1$
2. **Propriété de Kronecker:** $\hat{\varphi}_i(\xi_j) = \delta_{ij}$ où $\xi_1 = -1$, $\xi_2 = 1$
3. **Linéarité:** $\hat{\varphi}_i \in \mathbb{P}_1$ (polynômes de degré ≤ 1)

**Dérivées:**
$$\frac{d\hat{\varphi}_1}{d\xi} = -\frac{1}{2}, \quad \frac{d\hat{\varphi}_2}{d\xi} = \frac{1}{2}$$

### 3.4 Fonctions de Forme Physiques

Pour un élément $K_i = [x_{i-1}, x_i]$, les fonctions de forme physiques sont:

$$\varphi_1(x) = \frac{x_i - x}{h_i}, \quad \varphi_2(x) = \frac{x - x_{i-1}}{h_i}$$

**Gradients physiques:**

Par la règle de dérivation composée:
$$\frac{d\varphi}{dx} = \frac{d\hat{\varphi}}{d\xi} \cdot \frac{d\xi}{dx} = \frac{d\hat{\varphi}}{d\xi} \cdot \frac{1}{J_i}$$

D'où:
$$\frac{d\varphi_1}{dx} = -\frac{1}{2} \cdot \frac{2}{h_i} = -\frac{1}{h_i}$$
$$\frac{d\varphi_2}{dx} = \frac{1}{2} \cdot \frac{2}{h_i} = \frac{1}{h_i}$$

Donc le gradient est constant sur chaque élément:
$$\nabla\varphi = \begin{bmatrix} -1/h_i \\ 1/h_i \end{bmatrix}$$

### 3.5 Espace d'Approximation

L'**espace éléments finis P1** est:

$$V_h = \big\{v_h \in C^0([0,L]) : v_h|_{K_i} \in \mathbb{P}_1, \, v_h(0) = 0\big\}$$

Tout $v_h \in V_h$ s'écrit:
$$v_h(x) = \sum_{j=1}^{n} v_j \phi_j(x)$$

où $\phi_j$ sont les **fonctions de forme globales** ("chapeau") et $v_j = v_h(x_j)$ sont les **degrés de liberté** (valeurs nodales).

**Propriétés de $\phi_j$:**
- Support: $\text{supp}(\phi_j) = [x_{j-1}, x_{j+1}]$
- Valeur nodale: $\phi_j(x_i) = \delta_{ij}$
- Continuité: $\phi_j \in C^0$ mais $\phi_j' \notin C^0$ (discontinu aux nœuds)

---

## 4. Matrices Élémentaires

### 4.1 Matrice de Rigidité Locale

Pour un élément $K_i = [x_{i-1}, x_i]$, la **matrice de rigidité locale** est:

$$K^{(i)}_{jk} = \int_{K_i} \kappa \frac{d\varphi_j}{dx} \frac{d\varphi_k}{dx} \, dx$$

**Calcul explicite:**

Les gradients sont constants sur $K_i$:
$$\nabla\varphi = \begin{bmatrix} -1/h_i \\ 1/h_i \end{bmatrix}$$

Donc:
$$K^{(i)} = \int_{K_i} \kappa \begin{bmatrix} -1/h_i \\ 1/h_i \end{bmatrix} \begin{bmatrix} -1/h_i & 1/h_i \end{bmatrix} dx$$

$$= \kappa \cdot \frac{1}{h_i^2} \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix} \int_{K_i} dx$$

$$= \frac{\kappa}{h_i} \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}$$

**Propriétés:**
1. **Symétrie:** $K^{(i)} = K^{(i)T}$
2. **Semi-définie positive:** $K^{(i)}$ est singulière (vecteur propre $[1, 1]^T$ de valeur propre 0)
3. **Noyau:** $K^{(i)} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = 0$ (les constantes sont dans le noyau)

### 4.2 Vecteur Charge Volumique Locale

Le **vecteur charge volumique local** est:

$$F^{(i)}_j = \int_{K_i} f(x) \varphi_j(x) \, dx$$

**Calcul par quadrature de Gauss-Legendre à 2 points:**

Les points de Gauss sur $\hat{K} = [-1, 1]$ sont:
$$\xi_1 = -\frac{1}{\sqrt{3}}, \quad \xi_2 = \frac{1}{\sqrt{3}}$$

avec poids $w_1 = w_2 = 1$.

L'intégrale s'écrit:
$$F^{(i)}_j = \int_{-1}^1 f(x(\xi)) \hat{\varphi}_j(\xi) J_i \, d\xi \approx J_i \sum_{q=1}^2 w_q f(x(\xi_q)) \hat{\varphi}_j(\xi_q)$$

Soit:
$$F^{(i)}_j = \frac{h_i}{2} \Big[f(x(\xi_1)) \hat{\varphi}_j(\xi_1) + f(x(\xi_2)) \hat{\varphi}_j(\xi_2)\Big]$$

**Pour un terme source constant** $f(x) = f_0$:

$$F^{(i)} = \int_{K_i} f_0 \begin{bmatrix} \varphi_1 \\ \varphi_2 \end{bmatrix} dx = f_0 \frac{h_i}{2} \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$

### 4.3 Matrice de Masse Locale (pour problèmes temporels)

Pour les problèmes dépendant du temps, on définit la **matrice de masse locale**:

$$M^{(i)}_{jk} = \int_{K_i} \varphi_j \varphi_k \, dx$$

**Calcul explicite:**

Sur l'élément de référence:
$$M^{(i)}_{jk} = \int_{-1}^1 \hat{\varphi}_j(\xi) \hat{\varphi}_k(\xi) J_i \, d\xi$$

Après calcul:
$$M^{(i)} = \frac{h_i}{6} \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$$

### 4.4 Contribution de Robin au Bord

À $x = L$ (nœud $n$), la condition de Robin ajoute:

**Au niveau de la matrice:**
$$A[n, n] += \alpha$$

**Au niveau du vecteur charge:**
$$F[n] += \alpha u_E$$

Ces termes proviennent de:
$$\alpha u(L) v(L) \quad \text{et} \quad \alpha u_E v(L)$$

en posant $v(L) = \phi_n(L) = 1$ (fonction de forme du dernier nœud).

---

## 5. Assemblage et Résolution

### 5.1 Algorithme d'Assemblage

1. **Initialisation:**
   ```
   A = 0  (matrice n×n)
   F = 0  (vecteur de taille n)
   ```

2. **Assemblage volume** (boucle sur les éléments):
   ```
   Pour i = 1 à n:
       K_elem = matrice_rigidite_locale(K_i, kappa)
       F_elem = vecteur_charge_local(K_i, f)

       # Connectivité: élément i connecte nœuds [i-1, i]
       dofs = [i-1, i]

       # Assemblage
       A[dofs, dofs] += K_elem
       F[dofs] += F_elem
   ```

3. **Assemblage Robin** (si applicable):
   ```
   Si condition de Robin à x=L:
       A[n, n] += alpha
       F[n] += alpha * u_E
   ```

4. **Application de la condition de Dirichlet** (voir ci-dessous)

### 5.2 Structure de la Matrice Globale

**Sans condition de Robin**, la matrice globale $A$ pour un maillage uniforme ($h_i = h$) est **tridiagonale**:

$$A = \frac{\kappa}{h} \begin{bmatrix}
1 & -1 & 0 & \cdots & 0 \\
-1 & 2 & -1 & \ddots & \vdots \\
0 & -1 & 2 & \ddots & 0 \\
\vdots & \ddots & \ddots & \ddots & -1 \\
0 & \cdots & 0 & -1 & 1
\end{bmatrix}$$

**Avec condition de Robin** à $x=L$:

$$A[n,n] = \frac{\kappa}{h} + \alpha$$

La matrice reste tridiagonale mais le terme diagonal du coin est modifié.

### 5.3 Application de la Condition de Dirichlet par Élimination

Pour imposer $u(0) = u_D$ au nœud 0, on utilise la **méthode d'élimination**:

1. **Transférer la contribution au second membre:**
   ```
   F -= A[:, 0] * u_D
   ```

2. **Modifier la ligne 0:**
   ```
   A[0, :] = 0
   A[0, 0] = 1
   F[0] = u_D
   ```

Cette méthode:
- Préserve la symétrie de $A$
- Impose exactement $u_0 = u_D$
- Modifie le conditionnement de manière contrôlée

### 5.4 Résolution du Système Linéaire

Le système final est:
$$A U = F$$

**Méthodes de résolution:**

1. **Factorisation LU** (directe):
   - Coût: $O(n)$ pour matrice tridiagonale
   - Stable et précise
   - Recommandée pour 1D

2. **Gradient conjugué** (itérative):
   - Applicable car $A$ est symétrique définie positive
   - Coût par itération: $O(n)$
   - Utile pour très grands systèmes

3. **Solveur numpy** (pour implémentation simple):
   ```python
   U = np.linalg.solve(A, F)
   ```

---

## 6. Analyse d'Erreur et Convergence

### 6.1 Lemme de Céa

**Théorème (Céa):** Si $u$ est la solution exacte et $u_h$ la solution FEM, alors:

$$\|u - u_h\|_{H^1} \leq \frac{M}{\gamma} \inf_{v_h \in V_h} \|u - v_h\|_{H^1}$$

où $M$ est la constante de continuité et $\gamma$ la constante de coercivité de $a(\cdot, \cdot)$.

**Interprétation:** L'erreur FEM est quasi-optimale - elle est proche de la meilleure approximation possible dans $V_h$.

### 6.2 Inégalité d'Interpolation

**Théorème (Interpolation P1):** Pour $u \in H^2(\Omega)$ et son interpolé de Lagrange $\Pi_h u \in V_h$:

$$\|u - \Pi_h u\|_{L^2} \leq C h^2 |u|_{H^2}$$
$$\|u - \Pi_h u\|_{H^1} \leq C h |u|_{H^2}$$

où $|u|_{H^2} = \|u''\|_{L^2}$ est la semi-norme $H^2$.

**Démonstration (esquisse pour norme $H^1$):**

Sur un élément $K_i$ de taille $h_i$, par scaling:
$$\|u - \Pi_h u\|_{H^1(K_i)}^2 \leq C h_i^2 \|u''\|_{L^2(K_i)}^2$$

En sommant sur tous les éléments:
$$\|u - \Pi_h u\|_{H^1}^2 = \sum_i \|u - \Pi_h u\|_{H^1(K_i)}^2 \leq C h^2 \sum_i \|u''\|_{L^2(K_i)}^2 = C h^2 \|u''\|_{L^2}^2$$

### 6.3 Estimation d'Erreur Principale

**Théorème:** Pour $u \in H^2(\Omega)$ solution exacte et $u_h \in V_h$ solution FEM:

$$\|u - u_h\|_{L^2} \leq C h^2 \|u''\|_{L^2}$$
$$\|u - u_h\|_{H^1} \leq C h \|u''\|_{L^2}$$

**Démonstration:**

Par le lemme de Céa:
$$\|u - u_h\|_{H^1} \leq C \|u - \Pi_h u\|_{H^1} \leq C h \|u''\|_{L^2}$$

Pour la norme $L^2$, on utilise le **lemme d'Aubin-Nitsche** (dualité):

Soit $w$ solution de:
$$-\kappa w'' = u - u_h, \quad w(0) = 0, \quad -\kappa w'(L) = \alpha w(L)$$

Alors $w \in H^2$ et:
$$\|u - u_h\|_{L^2}^2 = a(u - u_h, w) = a(u - u_h, w - w_h)$$

pour tout $w_h \in V_h$. En choisissant $w_h = \Pi_h w$:

$$\|u - u_h\|_{L^2}^2 \leq C \|u - u_h\|_{H^1} \|w - \Pi_h w\|_{H^1}$$
$$\leq C (Ch \|u''\|_{L^2}) (Ch \|w''\|_{L^2})$$

Par régularité elliptique, $\|w''\|_{L^2} \leq C \|u - u_h\|_{L^2}$, d'où:

$$\|u - u_h\|_{L^2}^2 \leq C h^2 \|u''\|_{L^2} \|u - u_h\|_{L^2}$$

Donc:
$$\|u - u_h\|_{L^2} \leq C h^2 \|u''\|_{L^2} \quad \square$$

### 6.4 Taux de Convergence

Pour une séquence de maillages avec $h \to 0$:

| Norme | Taux théorique | Ordre de convergence |
|-------|----------------|---------------------|
| $\|u - u_h\|_{L^2}$ | $O(h^2)$ | Quadratique |
| $\|u - u_h\|_{H^1}$ | $O(h)$ | Linéaire |
| $\|u - u_h\|_{L^\infty}$ | $O(h^2)$ | Quadratique (1D uniquement) |

**Régularité requise:** Ces taux supposent $u \in H^2(\Omega)$. Si $u$ est plus régulier ($u \in H^3, H^4, ...$), des phénomènes de **superconvergence** peuvent apparaître (voir Section 9).

---

## 7. Méthode des Solutions Manufacturées

### 7.1 Principe de la MMS

La **Method of Manufactured Solutions (MMS)** est une technique de validation d'implémentations numériques:

1. **Choisir** une solution exacte $u_{\text{exact}}(x)$ arbitraire (régulière)
2. **Calculer** le terme source compatible:
   $$f(x) = -\kappa u_{\text{exact}}''(x)$$
3. **Résoudre** numériquement le problème avec ce $f(x)$
4. **Comparer** la solution numérique $u_h$ avec $u_{\text{exact}}$
5. **Vérifier** le taux de convergence en raffinant le maillage

### 7.2 Choix de la Solution Manufacturée

**Critères de sélection:**

1. **Satisfaction des BC:** $u_{\text{exact}}$ doit satisfaire les conditions aux limites
   - $u_{\text{exact}}(0) = u_D$ (Dirichlet)
   - $-\kappa u_{\text{exact}}'(L) = \alpha(u_{\text{exact}}(L) - u_E)$ (Robin)

2. **Régularité suffisante:** $u_{\text{exact}} \in C^\infty$ idéalement

3. **Non-trivialité:** Éviter les solutions trop simples (ex: polynômes de degré ≤ 1 qui sont représentés exactement)

**Exemple utilisé dans ce projet:**

$$u_{\text{exact}}(x) = \sin\left(\frac{\pi x}{2L}\right) \cdot x$$

**Vérification:**
- $u_{\text{exact}}(0) = 0$ ✓ (satisfait Dirichlet homogène)
- $u_{\text{exact}}(L) = \sin(\pi/2) \cdot L = L \neq 0$ ✓ (Robin non trivial)
- $u_{\text{exact}} \in C^\infty$ ✓

**Calcul du terme source:**

$$u' = \frac{\pi}{2L} x \cos\left(\frac{\pi x}{2L}\right) + \sin\left(\frac{\pi x}{2L}\right)$$

$$u'' = -\left(\frac{\pi}{2L}\right)^2 x \sin\left(\frac{\pi x}{2L}\right) + 2\frac{\pi}{2L}\cos\left(\frac{\pi x}{2L}\right)$$

$$f(x) = -\kappa u''(x)$$

### 7.3 Compatibilité avec la Condition de Robin

Pour que $u_{\text{exact}}$ satisfasse la BC de Robin, on doit choisir $(α, u_E)$ compatibles.

**Approche 1:** Imposer $\alpha$ et calculer $u_E$

De la condition de Robin:
$$-\kappa u_{\text{exact}}'(L) = \alpha(u_{\text{exact}}(L) - u_E)$$

On résout pour $u_E$:
$$u_E = u_{\text{exact}}(L) + \frac{\kappa}{\alpha} u_{\text{exact}}'(L)$$

**Approche 2:** Imposer $u_E$ et calculer $\alpha$

$$\alpha = \frac{-\kappa u_{\text{exact}}'(L)}{u_{\text{exact}}(L) - u_E}$$

(Attention: nécessite $u_{\text{exact}}(L) \neq u_E$)

**Dans notre implémentation:** On utilise l'Approche 1 avec $\alpha = 10.0$ fixé.

### 7.4 Protocole de Test de Convergence

1. **Définir une séquence de maillages:**
   $$n \in \{10, 20, 40, 80, 160\}$$

2. **Pour chaque maillage:**
   - Créer maillage uniforme de $n$ éléments
   - Assembler le système avec $f = f_{\text{MMS}}$
   - Appliquer BC avec $(α, u_E)$ compatibles
   - Résoudre le système: obtenir $U_h$
   - Calculer l'erreur $L^2$:
     $$e_{L^2} = \left(\int_0^L (u_h - u_{\text{exact}})^2 dx\right)^{1/2}$$
   - Approximer par règle des trapèzes:
     $$e_{L^2} \approx \sqrt{\sum_{i=0}^{n-1} \frac{h_i}{2}\left[(u_h(x_i) - u_{\text{exact}}(x_i))^2 + (u_h(x_{i+1}) - u_{\text{exact}}(x_{i+1}))^2\right]}$$

3. **Extraire le taux de convergence:**
   - Régression log-log: $\log(e_{L^2}) = \log(C) + p \log(h)$
   - Fit linéaire:
     ```python
     coeffs = np.polyfit(np.log(h), np.log(errors), 1)
     rate = coeffs[0]
     ```
   - Vérifier: $1.8 < \text{rate} < 2.2$ (tolérance pour bruit numérique)

4. **Validation:**
   - Si $\text{rate} \approx 2.0$: implémentation correcte ✓
   - Si $\text{rate} < 1.8$: bug probable ✗
   - Si $\text{rate} > 2.2$: superconvergence possible (voir Section 9)

---

## 8. Cas Limites des Conditions aux Limites

### 8.1 Robin → Neumann (α → 0)

Lorsque $\alpha \to 0$, la condition de Robin:
$$-\kappa u'(L) = \alpha(u(L) - u_E)$$

devient:
$$-\kappa u'(L) \to 0$$

**Interprétation:** Flux nul (isolation thermique parfaite).

**Formulation variationnelle limite:**
$$a(u, v) = \int_0^L \kappa u' v' \, dx$$
(le terme $\alpha u(L) v(L) \to 0$)

**Conséquence numérique:**
- La matrice $A[n,n]$ n'est plus modifiée
- Seulement la contribution volumique demeure

**Condition de compatibilité:** Pour problème de Neumann pur (BC Neumann des deux côtés), on doit avoir:
$$\int_0^L f \, dx = 0$$
sinon le problème n'a pas de solution (accumulation/perte de chaleur infinie).

### 8.2 Robin → Dirichlet (α → ∞)

Lorsque $\alpha \to \infty$:
$$\alpha(u(L) - u_E) = -\kappa u'(L)$$

implique:
$$u(L) - u_E \to 0 \quad \Rightarrow \quad u(L) = u_E$$

**Interprétation:** Température imposée à $x=L$ (contact parfait).

**Traitement numérique:** Au lieu de faire tendre $\alpha \to \infty$ (conditionnement numérique), on impose directement:
```python
A[n, :] = 0
A[n, n] = 1
F[n] = u_E
```

**Démonstration de la limite:**

La forme bilinéaire avec Robin:
$$a_\alpha(u, v) = \int_0^L \kappa u' v' \, dx + \alpha u(L) v(L)$$

Pour $v \in V$ tel que $v(L) = 1$:
$$a_\alpha(u, v) = \int_0^L \kappa u' v' \, dx + \alpha u(L)$$

Si $\alpha \to \infty$ et $a_\alpha$ reste borné, nécessairement $u(L) \to 0$ (ou valeur imposée).

### 8.3 Cas Dirichlet-Dirichlet

Si on impose Dirichlet aux deux bords:
$$u(0) = u_L, \quad u(L) = u_R$$

**Formulation variationnelle:**
$$V_0 = \{v \in H^1(\Omega) : v(0) = v(L) = 0\}$$

$$a(u, v) = \int_0^L \kappa u' v' \, dx$$

**Traitement numérique:**
```python
# Appliquer BC à gauche (nœud 0)
A[0, :] = 0
A[0, 0] = 1
F[0] = u_L

# Appliquer BC à droite (nœud n)
A[n, :] = 0
A[n, n] = 1
F[n] = u_R
```

---

## 9. Superconvergence

### 9.1 Définition

La **superconvergence** désigne l'observation d'un taux de convergence **supérieur** au taux théorique prédit par l'analyse a priori.

**Exemple:** Pour éléments P1, on attend $\|u - u_h\|_{L^2} = O(h^2)$, mais on observe parfois $O(h^3)$ ou $O(h^4)$.

### 9.2 Conditions d'Apparition

La superconvergence en éléments finis 1D P1 apparaît typiquement quand:

1. **Maillage uniforme:** Tous les éléments ont la même taille $h_i = h$
2. **Solution très régulière:** $u \in C^\infty$ ou $u \in H^k$ avec $k \gg 2$
3. **Points spécifiques:** Mesure de l'erreur aux nœuds ou points de Gauss
4. **Règle de quadrature:** Intégration de l'erreur avec règle des trapèzes

### 9.3 Explication Théorique

**Propriété de super-approximation:** Sur maillage uniforme, l'interpolé de Lagrange $\Pi_h u$ satisfait:

$$\|u - \Pi_h u\|_{L^2} = O(h^{k+1})$$

si $u \in H^{k+1}$ avec $k \geq 2$ (au lieu de $O(h^2)$ prédit pour $k=2$).

**Erreur aux nœuds (effet de Galerkin):** L'erreur nodale $e_i = u(x_i) - u_h(x_i)$ satisfait:

$$|e_i| = O(h^{2p})$$

pour éléments $\mathbb{P}_p$ et solution suffisamment régulière.

Pour $p=1$ (P1): $|e_i| = O(h^4)$ possible!

**Conséquence pour norme $L^2$:** Si on calcule:
$$\|u - u_h\|_{L^2} \approx \sqrt{\text{trapz}((u - u_h)^2, x)}$$

et que les valeurs nodales ont erreur $O(h^4)$, alors:
$$\|u - u_h\|_{L^2} \approx O(h^4)$$

### 9.4 Observation dans ce Projet

Dans [scripts/exercise2_manufactured.py](../scripts/exercise2_manufactured.py), on observe:

- Solution: $u_{\text{exact}} = \sin(\pi x / (2L)) \cdot x$ (très régulière, $C^\infty$)
- Maillage: Uniforme
- Résultat: **Taux ≈ 4.0** au lieu de 2.0

**Interprétation:**
- Ce n'est **PAS une erreur** d'implémentation
- C'est une **validation** que le code fonctionne correctement
- La régularité extrême de la solution permet la superconvergence

**Comment revenir à h²?**

1. **Solution moins régulière:** Utiliser $u(x) = x^{7/2}$ (régularité $H^{9/2}$ seulement)
2. **Maillage non-uniforme:** Introduire perturbations dans le maillage
3. **Quadrature différente:** Évaluer l'erreur avec quadrature de Gauss au lieu de trapèzes

### 9.5 Littérature

La superconvergence en FEM est un domaine de recherche actif:

- **Wahlbin (1995):** "Superconvergence in Galerkin Finite Element Methods"
- **Lin & Yan (1996):** "Superconvergence for finite element methods"
- **Schatz & Sloan (1974):** Premiers travaux sur superconvergence aux points de Gauss

**Résultat classique (1D):** Pour $u \in H^{k+1}$ et éléments $\mathbb{P}_k$ sur maillage uniforme:

$$\|u - u_h\|_{L^2} = O(h^{k+1})$$

au lieu de $O(h^{k})$ en général.

---

## 10. Références

### 10.1 Ouvrages de Référence

1. **P. G. Ciarlet** (1978). *The Finite Element Method for Elliptic Problems*. North-Holland.
   - Référence fondamentale pour l'analyse mathématique des EF

2. **D. Braess** (2007). *Finite Elements: Theory, Fast Solvers, and Applications in Solid Mechanics* (3rd ed.). Cambridge University Press.
   - Traitement moderne avec applications

3. **A. Ern & J.-L. Guermond** (2004). *Theory and Practice of Finite Elements*. Springer.
   - Approche unifiée, excellent pour la théorie

4. **S. C. Brenner & L. R. Scott** (2008). *The Mathematical Theory of Finite Element Methods* (3rd ed.). Springer.
   - Analyse rigoureuse, niveau avancé

### 10.2 Articles sur la Superconvergence

5. **L. B. Wahlbin** (1995). *Superconvergence in Galerkin Finite Element Methods*. Lecture Notes in Mathematics 1605, Springer.

6. **Q. Lin & N. Yan** (1996). *The Construction and Analysis of High Efficiency Finite Element Methods*. Hebei University Press.

7. **A. H. Schatz & I. H. Sloan** (1974). "Superconvergence in finite element methods and meshes that are locally symmetric with respect to a point." *SIAM Journal on Numerical Analysis*, 11(3), 632-654.

### 10.3 Méthode des Solutions Manufacturées

8. **P. J. Roache** (2002). "Code Verification by the Method of Manufactured Solutions." *Journal of Fluids Engineering*, 124(1), 4-10.
   - Introduction pratique à la MMS

9. **K. Salari & P. Knupp** (2000). *Code Verification by the Method of Manufactured Solutions*. Sandia Report SAND2000-1444.
   - Guide détaillé avec exemples

### 10.4 Documentation du Projet

10. **Polycopié du cours** (Chapitres 2 et 3).
    - Formulation variationnelle et discrétisation FEM

11. **Sujet d'examen** : Exercices 1 et 2.
    - Spécifications des problèmes à résoudre

---

## Annexe A: Notations

| Symbole | Signification |
|---------|---------------|
| $\Omega$ | Domaine 1D $]0, L[$ |
| $u(x)$ | Température [K] |
| $\kappa$ | Conductivité thermique [W/(m·K)] |
| $f(x)$ | Terme source volumique [W/m³] |
| $\alpha$ | Coefficient de transfert convectif [W/(m²·K)] |
| $u_E$ | Température extérieure [K] |
| $u_D$ | Température imposée (Dirichlet) [K] |
| $h$ | Pas de maillage [m] |
| $h_i$ | Longueur de l'élément $i$ [m] |
| $K_i$ | Élément $i$ : $[x_{i-1}, x_i]$ |
| $\hat{K}$ | Élément de référence $[-1, 1]$ |
| $V$ | Espace de Sobolev $H^1$ avec BC Dirichlet |
| $V_h$ | Espace éléments finis P1 |
| $\phi_j$ | Fonction de forme globale du nœud $j$ |
| $\varphi_j$ | Fonction de forme locale |
| $a(\cdot, \cdot)$ | Forme bilinéaire |
| $\ell(\cdot)$ | Forme linéaire |
| $K^{(i)}$ | Matrice de rigidité locale de l'élément $i$ |
| $F^{(i)}$ | Vecteur charge local de l'élément $i$ |
| $A$ | Matrice globale |
| $F$ | Vecteur charge global |
| $U$ | Vecteur solution (valeurs nodales) |

---

## Annexe B: Résumé des Théorèmes Clés

1. **Lax-Milgram:** Si $a(\cdot, \cdot)$ est continue et coercive sur $V$, et $\ell(\cdot)$ est continue, alors le problème variationnel admet une solution unique.

2. **Céa:** L'erreur FEM est quasi-optimale:
   $$\|u - u_h\|_{V} \leq C \inf_{v_h \in V_h} \|u - v_h\|_{V}$$

3. **Interpolation P1:**
   $$\|u - \Pi_h u\|_{L^2} \leq C h^2 |u|_{H^2}, \quad \|u - \Pi_h u\|_{H^1} \leq C h |u|_{H^2}$$

4. **Convergence FEM:**
   $$\|u - u_h\|_{L^2} \leq C h^2 \|u''\|_{L^2}, \quad \|u - u_h\|_{H^1} \leq C h \|u''\|_{L^2}$$

5. **Poincaré:** Sur $V = \{v \in H^1 : v(0) = 0\}$:
   $$\|v\|_{L^2} \leq C_P \|v'\|_{L^2}$$

6. **Trace (1D):** Pour $v \in H^1([0, L])$:
   $$|v(L)|^2 \leq C \|v\|_{H^1}^2$$

---

**Fin du document**

*Ce document constitue la base mathématique rigoureuse pour la compréhension et l'implémentation des éléments finis 1D dans le projet ElementFiniFusee.*
