"""
Generateur de maillage en forme de FUSEE VERTICALE (debout)
Forme reconnaissable : ogive pointue en haut + corps cylindrique
"""
import numpy as np

def generate_rocket_vertical_mesh(output_file="rocket_mesh.msh",
                                  H_nose=0.8,      # Hauteur ogive (m)
                                  H_body=2.0,      # Hauteur corps (m)
                                  R_body=0.4,      # Rayon corps (m)
                                  thickness=0.05,  # Epaisseur paroi (m)
                                  ny=30,           # Divisions verticales
                                  nr=6):           # Divisions radiales
    """
    Genere un maillage de fusee VERTICALE (debout)

    Coordonnees:
    - y : axe vertical (hauteur)
    - x : axe horizontal (rayon)

    Forme:
    - Ogive conique pointue en haut
    - Corps cylindrique
    - Paroi d'epaisseur constante
    """
    print("=" * 70)
    print("GENERATION MAILLAGE FUSEE VERTICALE")
    print("=" * 70)

    nodes = {}
    node_id = 1

    # Hauteur totale
    H_total = H_nose + H_body

    # Coordonnees y (verticales)
    y_coords = np.linspace(0, H_total, ny)

    # Coordonnees x (radiales) : de R_inner a R_outer
    R_inner = R_body - thickness
    x_coords = np.linspace(R_inner, R_body, nr)

    # Fonction de profil de la fusee (retourne le rayon en fonction de la hauteur)
    def rocket_radius(y, H_nose, H_body, R_body):
        """
        Retourne le rayon de la fusee a la hauteur y
        - Si y < H_body : corps cylindrique (rayon constant)
        - Si y >= H_body : ogive conique (rayon decroit lineairement)
        """
        if y < H_body:
            return R_body  # Corps cylindrique
        else:
            # Ogive conique : rayon decroit de R_body a 0
            ratio = (H_total - y) / H_nose
            return R_body * ratio

    # Matrice de noeuds
    node_matrix = np.zeros((ny, nr), dtype=int)

    # Generer les noeuds
    for i, y in enumerate(y_coords):
        for j, x in enumerate(x_coords):
            # Calculer le rayon exterieur a cette hauteur
            R_outer = rocket_radius(y, H_nose, H_body, R_body)
            R_inner_local = max(0.02, R_outer - thickness)

            # Interpoler entre rayon interieur et exterieur
            if nr > 1:
                ratio = j / (nr - 1)
            else:
                ratio = 0

            x_actual = R_inner_local + ratio * (R_outer - R_inner_local)

            # Creer le noeud (x, y, 0)
            nodes[node_id] = [x_actual, y, 0.0]
            node_matrix[i, j] = node_id
            node_id += 1

    print(f"[OK] Noeuds generes: {len(nodes)}")

    # Generer les triangles
    triangles = {}
    elem_id = 1

    for i in range(ny - 1):
        for j in range(nr - 1):
            # Quadrangle -> 2 triangles
            n1 = node_matrix[i, j]
            n2 = node_matrix[i + 1, j]
            n3 = node_matrix[i + 1, j + 1]
            n4 = node_matrix[i, j + 1]

            # Triangle 1
            triangles[elem_id] = {
                'type': 2,
                'tags': [10],
                'nodes': [n1, n2, n3]
            }
            elem_id += 1

            # Triangle 2
            triangles[elem_id] = {
                'type': 2,
                'tags': [10],
                'nodes': [n1, n3, n4]
            }
            elem_id += 1

    print(f"[OK] Triangles generes: {len(triangles)}")

    # Generer les aretes de bord
    edges = {}
    edge_id = elem_id

    # Bord exterieur (surface de la fusee - Robin)
    for i in range(ny - 1):
        n1 = node_matrix[i, nr - 1]
        n2 = node_matrix[i + 1, nr - 1]
        edges[edge_id] = {
            'type': 1,
            'tags': [1],  # Physical ID 1 = Gamma_F (Robin)
            'nodes': [n1, n2]
        }
        edge_id += 1

    print(f"[OK] Aretes Gamma_F (Robin - surface ext) generees: {ny - 1}")

    # Base (Dirichlet)
    for j in range(nr - 1):
        n1 = node_matrix[0, j]
        n2 = node_matrix[0, j + 1]
        edges[edge_id] = {
            'type': 1,
            'tags': [2],  # Physical ID 2 = Gamma_D (Dirichlet)
            'nodes': [n1, n2]
        }
        edge_id += 1

    print(f"[OK] Aretes Gamma_D (Dirichlet - base) generees: {nr - 1}")

    # Surface interieure (Neumann)
    for i in range(ny - 1):
        n1 = node_matrix[i, 0]
        n2 = node_matrix[i + 1, 0]
        edges[edge_id] = {
            'type': 1,
            'tags': [3],  # Physical ID 3 = Gamma_N (Neumann)
            'nodes': [n1, n2]
        }
        edge_id += 1

    print(f"[OK] Aretes Gamma_N (Neumann - surface int) generees: {ny - 1}")

    # Sommet de l'ogive (pointe)
    for j in range(nr - 1):
        n1 = node_matrix[ny - 1, j]
        n2 = node_matrix[ny - 1, j + 1]
        edges[edge_id] = {
            'type': 1,
            'tags': [4],  # Physical ID 4 = Pointe
            'nodes': [n1, n2]
        }
        edge_id += 1

    print(f"[OK] Aretes Pointe generees: {nr - 1}")

    # Ecrire le fichier .msh
    with open(output_file, 'w') as f:
        f.write("$MeshFormat\n")
        f.write("2.2 0 8\n")
        f.write("$EndMeshFormat\n")

        f.write("$PhysicalNames\n")
        f.write("5\n")
        f.write('1 1 "Gamma_F"\n')
        f.write('1 2 "Gamma_D"\n')
        f.write('1 3 "Gamma_N"\n')
        f.write('1 4 "Gamma_pointe"\n')
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

    print(f"[OK] Maillage ecrit: {output_file}")
    print("=" * 70)
    print(f"RESUME:")
    print(f"  - Noeuds:    {len(nodes)}")
    print(f"  - Triangles: {len(triangles)}")
    print(f"  - Aretes:    {len(edges)}")
    print(f"  - Forme:     FUSEE VERTICALE (ogive pointue en haut)")
    print("=" * 70)


if __name__ == '__main__':
    generate_rocket_vertical_mesh()
    print("\n[OK] Maillage fusee verticale genere!")
    print("Relancer: python demo_reentry.py")
