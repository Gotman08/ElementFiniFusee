"""
Generateur de maillage ARIANE 5 - Version amelioree
Fusee complete avec corps central (jupe + reservoir + ogive) + 2 boosters avec ogives biseautees
Maillage de qualite optimale avec triangles quasi-equilateraux
"""
import numpy as np


def generate_ariane5_mesh(output_file="rocket_mesh.msh",
                         # Dimensions (echelle ~1:25, proportions reelles Ariane 5)
                         H_nose=0.6,           # Hauteur ogive corps (m)
                         H_body=2.0,           # Hauteur corps principal (m)
                         H_booster=1.5,        # Hauteur boosters (m)
                         R_core=0.15,          # Rayon corps central/reservoir (m)
                         R_booster=0.10,       # Rayon boosters (m)
                         gap=0.02,             # Espacement boosters-corps (m)
                         thickness=0.03,       # Epaisseur paroi (m)
                         # NOUVEAUX PARAMETRES : Jupe moteur Vulcain
                         H_skirt=0.15,         # Hauteur jupe moteur (m) - proportions reelles
                         R_skirt=0.09,         # Rayon jupe (61% de R_core) - ratio reel 3.3/5.4
                         # NOUVEAUX PARAMETRES : Ogive boosters
                         H_booster_nose=0.15,  # Hauteur ogive boosters (m)
                         # Parametres maillage
                         nr=5):                # Divisions radiales (epaisseur)
    """
    Genere un maillage de fusee ARIANE 5 avec geometrie realiste

    Ameliorations par rapport a la version precedente:
    1. Corps central avec jupe moteur Vulcain (plus etroite a la base)
    2. Boosters avec ogives biseautees vers le corps central
    3. Calcul dynamique de ny pour triangles quasi-equilateraux (Dy ~ Dx)
    """
    print("=" * 70)
    print("GENERATION MAILLAGE ARIANE 5 - VERSION AMELIOREE")
    print("=" * 70)

    # ========================================================================
    # CALCUL DYNAMIQUE DU PAS SPATIAL (selon le cours: qualite Q_T)
    # ========================================================================
    # Le pas h est contraint par l'epaisseur de paroi (dimension la plus petite)
    h = thickness / (nr - 1)  # h ~ 0.0075m pour thickness=0.03, nr=5

    print(f"Pas spatial cible h = {h:.4f} m (pour triangles equilateraux)")

    H_total = H_nose + H_body
    H_total_booster = H_booster + H_booster_nose

    # Calcul dynamique du nombre de divisions verticales
    ny_core = int(np.ceil(H_total / h))
    ny_booster = int(np.ceil(H_total_booster / h))

    print(f"Divisions verticales: corps={ny_core}, boosters={ny_booster}")

    nodes = {}
    node_id = 1
    triangles = {}
    elem_id = 1
    edges = {}
    edge_id = 1

    # ========================================================================
    # FONCTION : RAYON DU CORPS CENTRAL EN FONCTION DE LA HAUTEUR
    # ========================================================================
    def get_core_radius(y):
        """
        Retourne le rayon exterieur du corps central a la hauteur y
        3 zones: jupe (base), reservoir (principal), ogive (sommet)
        """
        if y < H_skirt:
            # Zone jupe moteur Vulcain : rayon constant R_skirt
            return R_skirt
        elif y < H_body:
            # Zone reservoir : rayon constant R_core
            return R_core
        else:
            # Zone ogive : rayon decroit lineairement vers 0
            ratio = (H_total - y) / H_nose
            return R_core * max(0.01, ratio)  # Eviter rayon nul

    # ========================================================================
    # FONCTION : GEOMETRIE DES BOOSTERS (avec ogive biseautee)
    # ========================================================================
    def get_booster_geometry(y, x_center_base):
        """
        Retourne (x_center, radius) pour un booster a la hauteur y
        L'ogive est biseautee vers le corps central
        """
        if y <= H_booster:
            # Zone cylindrique : centre et rayon constants
            return x_center_base, R_booster
        else:
            # Zone ogive biseautee
            t = (y - H_booster) / H_booster_nose  # t: 0 -> 1
            t = min(1.0, t)  # Securite
            # Rayon decroit lineairement
            radius = R_booster * (1 - t)
            radius = max(0.01, radius)  # Eviter rayon nul
            # Centre se rapproche de x=0 (corps central)
            x_center = x_center_base * (1 - 0.5 * t)  # Biseau a 50% vers le centre
            return x_center, radius

    # ========================================================================
    # FONCTION : CREER UN MAILLAGE DE PAROI AVEC PROFIL VARIABLE
    # ========================================================================
    def create_mesh_with_profile(y_coords, get_geometry_func, x_base, nr_local, node_id_start):
        """
        Cree un maillage de paroi avec rayon et centre variables en fonction de y

        Args:
            y_coords: coordonnees y des lignes de noeuds
            get_geometry_func: fonction(y) -> (x_center, R_outer) ou fonction(y) -> R_outer
            x_base: centre x de base (pour corps central = 0)
            nr_local: nombre de divisions radiales
            node_id_start: ID du premier noeud

        Returns:
            local_nodes, local_triangles, outer_nodes, inner_nodes,
            bottom_nodes, top_nodes, next_node_id
        """
        local_nodes = {}
        local_triangles = []
        ny_local = len(y_coords)

        node_matrix = np.zeros((ny_local, nr_local), dtype=int)
        current_node_id = node_id_start

        for i, y in enumerate(y_coords):
            # Obtenir la geometrie a cette hauteur
            result = get_geometry_func(y)
            if isinstance(result, tuple):
                x_center, R_outer = result
            else:
                x_center = x_base
                R_outer = result

            R_inner = max(0.005, R_outer - thickness)

            for j in range(nr_local):
                ratio_r = j / (nr_local - 1) if nr_local > 1 else 0
                r_actual = R_inner + ratio_r * (R_outer - R_inner)
                x_actual = x_center + r_actual

                local_nodes[current_node_id] = [x_actual, y, 0.0]
                node_matrix[i, j] = current_node_id
                current_node_id += 1

        # Creer les triangles
        for i in range(ny_local - 1):
            for j in range(nr_local - 1):
                n1 = node_matrix[i, j]
                n2 = node_matrix[i + 1, j]
                n3 = node_matrix[i + 1, j + 1]
                n4 = node_matrix[i, j + 1]

                local_triangles.append([n1, n2, n3])
                local_triangles.append([n1, n3, n4])

        # Extraire les noeuds de bord
        outer_nodes = [node_matrix[i, nr_local - 1] for i in range(ny_local)]
        inner_nodes = [node_matrix[i, 0] for i in range(ny_local)]
        bottom_nodes = [node_matrix[0, j] for j in range(nr_local)]
        top_nodes = [node_matrix[ny_local - 1, j] for j in range(nr_local)]

        return local_nodes, local_triangles, outer_nodes, inner_nodes, bottom_nodes, top_nodes, current_node_id

    # ========================================================================
    # CREER LE CORPS CENTRAL (jupe + reservoir + ogive)
    # ========================================================================
    print("Generation du corps central (jupe + reservoir + ogive)...")

    y_coords_core = np.linspace(0, H_total, ny_core)

    core_nodes, core_triangles, core_outer, core_inner, core_bottom, core_top, node_id = \
        create_mesh_with_profile(y_coords_core, get_core_radius, 0.0, nr, node_id)

    nodes.update(core_nodes)
    for tri in core_triangles:
        triangles[elem_id] = {'type': 2, 'tags': [10], 'nodes': tri}
        elem_id += 1

    print(f"  - Corps central: {len(core_nodes)} noeuds, {len(core_triangles)} triangles")

    # ========================================================================
    # CREER LE BOOSTER GAUCHE (avec ogive biseautee)
    # ========================================================================
    print("Generation du booster gauche (avec ogive biseautee)...")

    x_booster_left = -(R_core + gap + R_booster)
    y_coords_booster = np.linspace(0, H_total_booster, ny_booster)

    # Fonction lambda pour le booster gauche
    get_left_geom = lambda y: get_booster_geometry(y, x_booster_left)

    left_nodes, left_triangles, left_outer, left_inner, left_bottom, left_top, node_id = \
        create_mesh_with_profile(y_coords_booster, get_left_geom, x_booster_left, nr, node_id)

    nodes.update(left_nodes)
    for tri in left_triangles:
        triangles[elem_id] = {'type': 2, 'tags': [10], 'nodes': tri}
        elem_id += 1

    print(f"  - Booster gauche: {len(left_nodes)} noeuds, {len(left_triangles)} triangles")

    # ========================================================================
    # CREER LE BOOSTER DROIT (avec ogive biseautee)
    # ========================================================================
    print("Generation du booster droit (avec ogive biseautee)...")

    x_booster_right = R_core + gap + R_booster

    # Fonction lambda pour le booster droit
    get_right_geom = lambda y: get_booster_geometry(y, x_booster_right)

    right_nodes, right_triangles, right_outer, right_inner, right_bottom, right_top, node_id = \
        create_mesh_with_profile(y_coords_booster, get_right_geom, x_booster_right, nr, node_id)

    nodes.update(right_nodes)
    for tri in right_triangles:
        triangles[elem_id] = {'type': 2, 'tags': [10], 'nodes': tri}
        elem_id += 1

    print(f"  - Booster droit: {len(right_nodes)} noeuds, {len(right_triangles)} triangles")

    # ========================================================================
    # CREER LES ARETES DE BORD
    # ========================================================================
    print("Generation des aretes de bord...")

    # Surface exterieure (Robin) : corps + boosters
    all_outer_nodes = [
        (core_outer, "Corps"),
        (left_outer, "Booster G"),
        (right_outer, "Booster D")
    ]

    edge_count = 0
    for outer_nodes_list, name in all_outer_nodes:
        for i in range(len(outer_nodes_list) - 1):
            edges[edge_id] = {
                'type': 1,
                'tags': [1],  # Physical ID 1 = Gamma_F (Robin)
                'nodes': [outer_nodes_list[i], outer_nodes_list[i + 1]]
            }
            edge_id += 1
            edge_count += 1

    print(f"  - Aretes Gamma_F (Robin): {edge_count}")

    # Base (Dirichlet) : corps + boosters
    all_bottom_nodes = [core_bottom, left_bottom, right_bottom]

    edge_count_d = 0
    for bottom_nodes_list in all_bottom_nodes:
        for i in range(len(bottom_nodes_list) - 1):
            edges[edge_id] = {
                'type': 1,
                'tags': [2],  # Physical ID 2 = Gamma_D (Dirichlet)
                'nodes': [bottom_nodes_list[i], bottom_nodes_list[i + 1]]
            }
            edge_id += 1
            edge_count_d += 1

    print(f"  - Aretes Gamma_D (Dirichlet): {edge_count_d}")

    # ========================================================================
    # CALCULER LA QUALITE DES TRIANGLES
    # ========================================================================
    def compute_triangle_quality(nodes_dict, triangle):
        """Calcule Q_T = (sqrt(3)/6) * (h_T / r_T) pour un triangle"""
        coords = np.array([nodes_dict[n][:2] for n in triangle])

        # Longueurs des cotes
        a = np.linalg.norm(coords[1] - coords[0])
        b = np.linalg.norm(coords[2] - coords[1])
        c = np.linalg.norm(coords[0] - coords[2])

        # Semi-perimetre et aire
        s = (a + b + c) / 2
        area_sq = s * (s - a) * (s - b) * (s - c)

        if area_sq <= 0:
            return 0.0

        area = np.sqrt(area_sq)

        # Rayon du cercle inscrit
        r = area / s

        # Plus grande hauteur
        h_T = 2 * area / min(a, b, c)

        # Qualite
        Q_T = (np.sqrt(3) / 6) * (h_T / r) if r > 0 else 0

        return min(Q_T, 1.0)

    # Calculer la qualite moyenne
    qualities = []
    for tri_data in triangles.values():
        q = compute_triangle_quality(nodes, tri_data['nodes'])
        qualities.append(q)

    avg_quality = np.mean(qualities) if qualities else 0
    min_quality = np.min(qualities) if qualities else 0

    # ========================================================================
    # ECRIRE LE FICHIER .msh
    # ========================================================================
    with open(output_file, 'w') as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")

        f.write("$PhysicalNames\n3\n")
        f.write('1 1 "Gamma_F"\n')
        f.write('1 2 "Gamma_D"\n')
        f.write('2 10 "Omega"\n')
        f.write("$EndPhysicalNames\n")

        f.write("$Nodes\n")
        f.write(f"{len(nodes)}\n")
        for nid, coords in nodes.items():
            f.write(f"{nid} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f}\n")
        f.write("$EndNodes\n")

        all_elements = {**edges, **triangles}
        f.write("$Elements\n")
        f.write(f"{len(all_elements)}\n")
        for eid, elem in all_elements.items():
            elem_type = elem['type']
            tags = elem['tags']
            nodes_list = elem['nodes']
            num_tags = len(tags)
            f.write(f"{eid} {elem_type} {num_tags} {' '.join(map(str, tags))} {' '.join(map(str, nodes_list))}\n")
        f.write("$EndElements\n")

    print(f"\n[OK] Maillage ecrit: {output_file}")
    print("=" * 70)
    print(f"ARIANE 5 - RESUME:")
    print(f"  - Noeuds:              {len(nodes)}")
    print(f"  - Triangles:           {len(triangles)}")
    print(f"  - Aretes:              {len(edges)}")
    print(f"  - Structure:           Corps (jupe+reservoir+ogive) + 2 boosters biseautes")
    print(f"  - Largeur totale:      {2*(R_core + gap + R_booster):.2f} m")
    print(f"  - Hauteur totale:      {H_total:.2f} m")
    print(f"  - Pas spatial h:       {h:.4f} m")
    print(f"  - Qualite Q_T moyenne: {avg_quality:.3f}")
    print(f"  - Qualite Q_T min:     {min_quality:.3f}")
    print("=" * 70)

    return {
        'nodes': len(nodes),
        'triangles': len(triangles),
        'edges': len(edges),
        'avg_quality': avg_quality,
        'min_quality': min_quality,
        'h': h
    }


if __name__ == '__main__':
    stats = generate_ariane5_mesh()
    print("\n[OK] Maillage ARIANE 5 genere!")
    print("Relancer: python demo_reentry.py")
