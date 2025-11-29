"""
Generateur ARIANE 5 SIMPLIFIE
Vue de face simplifiee : 3 rectangles (corps + 2 boosters)
"""
import numpy as np

def generate_ariane5_simple(output_file="rocket_mesh.msh"):
    """
    Ariane 5 simplifie : vue de face
    - Corps central (rectangle large)
    - Booster gauche (rectangle)
    - Booster droit (rectangle)
    """
    print("=" * 70)
    print("GENERATION ARIANE 5 SIMPLIFIEE")
    print("=" * 70)

    # Dimensions (proportions Ariane 5)
    H_total = 2.5      # Hauteur totale (m)
    W_core = 0.3       # Largeur corps central (m)
    W_booster = 0.2    # Largeur boosters (m)
    gap = 0.05         # Espacement (m)

    nx_core = 8        # Divisions horizontales corps
    nx_booster = 6     # Divisions horizontales boosters
    ny = 35            # Divisions verticales

    nodes = {}
    node_id = 1
    triangles = {}
    elem_id = 1

    # ========================================================================
    # FONCTION: CREER UN RECTANGLE VERTICAL MAILLE
    # ========================================================================
    def create_rectangle(x_left, x_right, y_bottom, y_top, nx_div, ny_div, node_id_start):
        """Cree un rectangle maille"""
        local_nodes = {}
        local_triangles = []

        x_coords = np.linspace(x_left, x_right, nx_div)
        y_coords = np.linspace(y_bottom, y_top, ny_div)

        node_matrix = np.zeros((ny_div, nx_div), dtype=int)
        current_id = node_id_start

        # Creer les noeuds
        for i, y in enumerate(y_coords):
            for j, x in enumerate(x_coords):
                local_nodes[current_id] = [x, y, 0.0]
                node_matrix[i, j] = current_id
                current_id += 1

        # Creer les triangles
        for i in range(ny_div - 1):
            for j in range(nx_div - 1):
                n1 = node_matrix[i, j]
                n2 = node_matrix[i, j + 1]
                n3 = node_matrix[i + 1, j + 1]
                n4 = node_matrix[i + 1, j]

                local_triangles.append([n1, n2, n3])
                local_triangles.append([n1, n3, n4])

        # Extraire les bords
        left_edge = [node_matrix[i, 0] for i in range(ny_div)]
        right_edge = [node_matrix[i, nx_div - 1] for i in range(ny_div)]
        bottom_edge = [node_matrix[0, j] for j in range(nx_div)]
        top_edge = [node_matrix[ny_div - 1, j] for j in range(nx_div)]

        return local_nodes, local_triangles, left_edge, right_edge, bottom_edge, top_edge, current_id

    # ========================================================================
    # CORPS CENTRAL
    # ========================================================================
    x_core_left = -W_core / 2
    x_core_right = W_core / 2

    core_nodes, core_tri, core_left, core_right, core_bottom, core_top, node_id = \
        create_rectangle(x_core_left, x_core_right, 0.0, H_total, nx_core, ny, node_id)

    nodes.update(core_nodes)
    for tri in core_tri:
        triangles[elem_id] = {'type': 2, 'tags': [10], 'nodes': tri}
        elem_id += 1

    print(f"Corps central: {len(core_nodes)} noeuds, {len(core_tri)} triangles")

    # ========================================================================
    # BOOSTER GAUCHE
    # ========================================================================
    x_left_left = -W_core / 2 - gap - W_booster
    x_left_right = -W_core / 2 - gap
    H_booster = H_total * 0.6  # Boosters plus courts

    left_nodes, left_tri, left_left, left_right, left_bottom, left_top, node_id = \
        create_rectangle(x_left_left, x_left_right, 0.0, H_booster, nx_booster, int(ny * 0.6), node_id)

    nodes.update(left_nodes)
    for tri in left_tri:
        triangles[elem_id] = {'type': 2, 'tags': [10], 'nodes': tri}
        elem_id += 1

    print(f"Booster gauche: {len(left_nodes)} noeuds, {len(left_tri)} triangles")

    # ========================================================================
    # BOOSTER DROIT
    # ========================================================================
    x_right_left = W_core / 2 + gap
    x_right_right = W_core / 2 + gap + W_booster

    right_nodes, right_tri, right_left, right_right, right_bottom, right_top, node_id = \
        create_rectangle(x_right_left, x_right_right, 0.0, H_booster, nx_booster, int(ny * 0.6), node_id)

    nodes.update(right_nodes)
    for tri in right_tri:
        triangles[elem_id] = {'type': 2, 'tags': [10], 'nodes': tri}
        elem_id += 1

    print(f"Booster droit: {len(right_nodes)} noeuds, {len(right_tri)} triangles")

    # ========================================================================
    # ARETES DE BORD
    # ========================================================================
    edges = {}
    edge_id = 1

    # Surfaces exterieures (Robin)
    all_outer = [
        (core_left, "Core gauche"),
        (core_right, "Core droit"),
        (core_top, "Core haut"),
        (left_left, "Booster G gauche"),
        (left_top, "Booster G haut"),
        (right_right, "Booster D droit"),
        (right_top, "Booster D haut")
    ]

    robin_count = 0
    for edge_nodes, name in all_outer:
        for i in range(len(edge_nodes) - 1):
            edges[edge_id] = {
                'type': 1,
                'tags': [1],
                'nodes': [edge_nodes[i], edge_nodes[i + 1]]
            }
            edge_id += 1
            robin_count += 1

    print(f"Aretes Gamma_F (Robin): {robin_count}")

    # Base (Dirichlet)
    all_bottom = [core_bottom, left_bottom, right_bottom]

    dir_count = 0
    for bottom_nodes in all_bottom:
        for i in range(len(bottom_nodes) - 1):
            edges[edge_id] = {
                'type': 1,
                'tags': [2],
                'nodes': [bottom_nodes[i], bottom_nodes[i + 1]]
            }
            edge_id += 1
            dir_count += 1

    print(f"Aretes Gamma_D (Dirichlet): {dir_count}")

    # ========================================================================
    # ECRIRE LE FICHIER
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

    print(f"\n[OK] Maillage Ariane 5 simplifie: {output_file}")
    print("=" * 70)
    print(f"TOTAL:")
    print(f"  - Noeuds:    {len(nodes)}")
    print(f"  - Triangles: {len(triangles)}")
    print(f"  - Structure: ARIANE 5 (corps + 2 boosters)")
    print("=" * 70)


if __name__ == '__main__':
    generate_ariane5_simple()
    print("\nRelancer: python demo_reentry.py")
