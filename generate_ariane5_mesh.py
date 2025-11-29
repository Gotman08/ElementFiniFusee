"""
Generateur de maillage ARIANE 5
Fusee complete avec corps central + 2 boosters lateraux + ogive
"""
import numpy as np

def generate_ariane5_mesh(output_file="rocket_mesh.msh",
                         # Dimensions (echelle reduite, proportions reelles)
                         H_nose=0.6,        # Hauteur ogive (m)
                         H_body=2.0,        # Hauteur corps principal (m)
                         H_booster=1.5,     # Hauteur boosters (m)
                         R_core=0.15,       # Rayon corps central (m)
                         R_booster=0.10,    # Rayon boosters (m)
                         gap=0.02,          # Espacement boosters-corps (m)
                         thickness=0.03,    # Epaisseur paroi (m)
                         ny=40,             # Divisions verticales
                         nr=5):             # Divisions radiales (epaisseur)
    """
    Genere un maillage de fusee ARIANE 5
    Vue de face : Corps central + 2 boosters lateraux
    """
    print("=" * 70)
    print("GENERATION MAILLAGE ARIANE 5")
    print("=" * 70)

    nodes = {}
    node_id = 1
    triangles = {}
    elem_id = 1
    edges = {}
    edge_id = 1

    H_total = H_nose + H_body

    # ========================================================================
    # FONCTION AUXILIAIRE : CREER UN CYLINDRE VERTICAL
    # ========================================================================
    def create_vertical_cylinder(x_center, y_start, y_end, radius, thickness, ny_local, nr_local, node_id_start):
        """
        Cree un cylindre vertical (ou cone)
        Retourne : nodes_dict, triangles_list, outer_edge_nodes, inner_edge_nodes, node_id_next
        """
        local_nodes = {}
        local_triangles = []

        y_coords = np.linspace(y_start, y_end, ny_local)

        # Pour chaque hauteur
        node_matrix = np.zeros((ny_local, nr_local), dtype=int)
        current_node_id = node_id_start

        for i, y in enumerate(y_coords):
            # Calculer le rayon a cette hauteur (pour ogive conique)
            if y > H_body:
                # Dans l'ogive : rayon decroit
                ratio = (H_total - y) / H_nose
                R_outer = R_core * ratio
            else:
                R_outer = radius

            R_inner = max(0.01, R_outer - thickness)

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
    # CREER LE CORPS CENTRAL (avec ogive)
    # ========================================================================
    print("Generation du corps central...")

    core_nodes, core_triangles, core_outer, core_inner, core_bottom, core_top, node_id = \
        create_vertical_cylinder(0.0, 0.0, H_total, R_core, thickness, ny, nr, node_id)

    nodes.update(core_nodes)
    for tri in core_triangles:
        triangles[elem_id] = {'type': 2, 'tags': [10], 'nodes': tri}
        elem_id += 1

    print(f"  - Corps central: {len(core_nodes)} noeuds")

    # ========================================================================
    # CREER LE BOOSTER GAUCHE
    # ========================================================================
    print("Generation du booster gauche...")

    x_booster_left = -(R_core + gap + R_booster)
    y_booster_start = 0.0
    y_booster_end = H_booster

    left_nodes, left_triangles, left_outer, left_inner, left_bottom, left_top, node_id = \
        create_vertical_cylinder(x_booster_left, y_booster_start, y_booster_end,
                                R_booster, thickness, int(ny * H_booster / H_total), nr, node_id)

    nodes.update(left_nodes)
    for tri in left_triangles:
        triangles[elem_id] = {'type': 2, 'tags': [10], 'nodes': tri}
        elem_id += 1

    print(f"  - Booster gauche: {len(left_nodes)} noeuds")

    # ========================================================================
    # CREER LE BOOSTER DROIT
    # ========================================================================
    print("Generation du booster droit...")

    x_booster_right = R_core + gap + R_booster

    right_nodes, right_triangles, right_outer, right_inner, right_bottom, right_top, node_id = \
        create_vertical_cylinder(x_booster_right, y_booster_start, y_booster_end,
                                R_booster, thickness, int(ny * H_booster / H_total), nr, node_id)

    nodes.update(right_nodes)
    for tri in right_triangles:
        triangles[elem_id] = {'type': 2, 'tags': [10], 'nodes': tri}
        elem_id += 1

    print(f"  - Booster droit: {len(right_nodes)} noeuds")

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

    edge_count = 0
    for bottom_nodes_list in all_bottom_nodes:
        for i in range(len(bottom_nodes_list) - 1):
            edges[edge_id] = {
                'type': 1,
                'tags': [2],  # Physical ID 2 = Gamma_D (Dirichlet)
                'nodes': [bottom_nodes_list[i], bottom_nodes_list[i + 1]]
            }
            edge_id += 1
            edge_count += 1

    print(f"  - Aretes Gamma_D (Dirichlet): {edge_count}")

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
    print(f"  - Noeuds:           {len(nodes)}")
    print(f"  - Triangles:        {len(triangles)}")
    print(f"  - Aretes:           {len(edges)}")
    print(f"  - Structure:        Corps central + 2 boosters lateraux + ogive")
    print(f"  - Largeur totale:   {2*(R_core + gap + R_booster):.2f} m")
    print(f"  - Hauteur totale:   {H_total:.2f} m")
    print("=" * 70)


if __name__ == '__main__':
    generate_ariane5_mesh()
    print("\n[OK] Maillage ARIANE 5 genere!")
    print("Relancer: python demo_reentry.py")
