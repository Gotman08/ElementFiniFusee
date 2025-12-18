// ============================================================================
// Géométrie simplifiée de fusée pour analyse thermique
// Domaine 2D axisymétrique: Ogive conique + Corps cylindrique
// ============================================================================

// Paramètres géométriques (en mètres)
L_nose = 0.5;      // Longueur de l'ogive
L_body = 2.0;      // Longueur du corps
R_body = 0.3;      // Rayon du corps
thickness = 0.05;  // Épaisseur de paroi

// Taille de maille
lc = 0.05;         // Taille caractéristique

// ============================================================================
// POINTS DE LA GÉOMÉTRIE
// ============================================================================

// Pointe de l'ogive (sur l'axe)
Point(1) = {0, 0, 0, lc};

// Jonction ogive-cylindre (extérieur)
Point(2) = {L_nose, R_body, 0, lc};

// Base du cylindre (extérieur)
Point(3) = {L_nose + L_body, R_body, 0, lc};

// Base du cylindre (intérieur)
Point(4) = {L_nose + L_body, R_body - thickness, 0, lc};

// Jonction ogive-cylindre (intérieur)
Point(5) = {L_nose, R_body - thickness, 0, lc};

// Pointe intérieure de l'ogive
Point(6) = {0.1, 0, 0, lc};

// ============================================================================
// LIGNES DE LA GÉOMÉTRIE
// ============================================================================

// Surface extérieure de l'ogive (ligne droite simplifiée)
Line(1) = {1, 2};

// Surface extérieure du cylindre
Line(2) = {2, 3};

// Base (ouverture arrière)
Line(3) = {3, 4};

// Surface intérieure du cylindre
Line(4) = {4, 5};

// Surface intérieure de l'ogive
Line(5) = {5, 6};

// Pointe (petit segment sur l'axe)
Line(6) = {6, 1};

// ============================================================================
// SURFACE (Domaine de calcul = paroi de la fusée)
// ============================================================================

Line Loop(1) = {1, 2, 3, 4, 5, 6};
Plane Surface(1) = {1};

// ============================================================================
// GROUPES PHYSIQUES (pour conditions aux limites)
// ============================================================================

// Physical ID 1: Surface extérieure (ogive + cylindre) - Condition de Robin
Physical Line("Gamma_F", 1) = {1, 2};

// Physical ID 2: Base (ouverture arrière) - Condition de Dirichlet
Physical Line("Gamma_D", 2) = {3};

// Physical ID 3: Surface intérieure - Condition de Neumann homogène (isolée)
Physical Line("Gamma_N", 3) = {4, 5};

// Physical ID 4: Pointe (axe de symétrie) - Condition de Neumann homogène
Physical Line("Gamma_axis", 4) = {6};

// Physical ID 10: Domaine de calcul (paroi)
Physical Surface("Omega", 10) = {1};

// ============================================================================
// OPTIONS DE MAILLAGE
// ============================================================================

// Algorithme de maillage (5 = Delaunay, 6 = Frontal-Delaunay)
Mesh.Algorithm = 6;

// Ordre des éléments (1 = P1 linéaire)
Mesh.ElementOrder = 1;

// Optimisation du maillage
Mesh.Optimize = 1;

// Format de sortie (1 = .msh version 2 ASCII)
Mesh.MshFileVersion = 2.2;
