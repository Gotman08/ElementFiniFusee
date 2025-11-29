"""
Générateur de maillage simple en Python (sans dépendance GMSH)
Crée un maillage triangulaire structuré pour la géométrie de fusée
"""
import numpy as np

def generate_rocket_mesh(output_file="rocket_mesh.msh",
                        L_nose=0.5,
                        L_body=2.0,
                        R_body=0.3,
                        thickness=0.05,
                        nx=25,
                        ny=8):
    """
    Génère un maillage simplifié de la paroi de fusée

    Args:
        output_file: Nom du fichier .msh de sortie
        L_nose: Longueur de l'ogive (m)
        L_body: Longueur du corps cylindrique (m)
        R_body: Rayon du corps (m)
        thickness: Épaisseur de paroi (m)
        nx: Nombre de divisions longitudinales
        ny: Nombre de divisions radiales
    """
    print("=" * 70)
    print("GÉNÉRATION DU MAILLAGE PYTHON")
    print("=" * 70)

    # ========================================================================
    # GÉNÉRATION DES POINTS
    # ========================================================================

    nodes = {}
    node_id = 1

    # Coordonnées x (longitudinal)
    x_coords = np.linspace(0, L_nose + L_body, nx)

    # Coordonnées y (radial) : de R_inner à R_outer
    R_inner = R_body - thickness
    y_coords = np.linspace(R_inner, R_body, ny)

    # Fonction définissant le profil de l'ogive
    def rocket_profile(x, R_body, L_nose):
        """Profil linéaire de l'ogive"""
        if x <= L_nose:
            return R_body * (x / L_nose)  # Cône
        else:
            return R_body  # Cylindre

    # Créer la grille de noeuds
    node_matrix = np.zeros((nx, ny), dtype=int)

    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            # Ajuster y en fonction du profil (pour l'extérieur seulement)
            if j == ny - 1:  # Surface extérieure
                y_actual = rocket_profile(x, R_body, L_nose)
            elif j == 0:  # Surface intérieure
                y_actual = max(0.05, rocket_profile(x, R_body, L_nose) - thickness)
            else:  # Intérieur de la paroi
                ratio = j / (ny - 1)
                y_outer = rocket_profile(x, R_body, L_nose)
                y_inner = max(0.05, y_outer - thickness)
                y_actual = y_inner + ratio * (y_outer - y_inner)

            nodes[node_id] = [x, y_actual, 0.0]
            node_matrix[i, j] = node_id
            node_id += 1

    print(f"[OK] Noeuds generes: {len(nodes)}")

    # ========================================================================
    # GÉNÉRATION DES ÉLÉMENTS TRIANGULAIRES
    # ========================================================================

    triangles = {}
    elem_id = 1

    for i in range(nx - 1):
        for j in range(ny - 1):
            # Récupérer les 4 coins du quadrangle
            n1 = node_matrix[i, j]
            n2 = node_matrix[i + 1, j]
            n3 = node_matrix[i + 1, j + 1]
            n4 = node_matrix[i, j + 1]

            # Diviser en 2 triangles
            # Triangle 1: n1 - n2 - n3
            triangles[elem_id] = {
                'type': 2,  # Type 2 = triangle dans GMSH
                'tags': [10],  # Physical ID 10 = domaine
                'nodes': [n1, n2, n3]
            }
            elem_id += 1

            # Triangle 2: n1 - n3 - n4
            triangles[elem_id] = {
                'type': 2,
                'tags': [10],
                'nodes': [n1, n3, n4]
            }
            elem_id += 1

    print(f"[OK] Triangles generes: {len(triangles)}")

    # ========================================================================
    # GENERATION DES ARETES DE BORD
    # ========================================================================

    edges = {}
    edge_id = elem_id

    # Bord exterieur (Gamma_F - Robin)
    # Physical ID 1
    for i in range(nx - 1):
        n1 = node_matrix[i, ny - 1]
        n2 = node_matrix[i + 1, ny - 1]
        edges[edge_id] = {
            'type': 1,  # Type 1 = ligne dans GMSH
            'tags': [1],  # Physical ID 1 = Gamma_F
            'nodes': [n1, n2]
        }
        edge_id += 1

    print(f"[OK] Aretes Gamma_F (Robin) generees: {nx - 1}")

    # Bord base (Gamma_D - Dirichlet)
    # Physical ID 2
    for j in range(ny - 1):
        n1 = node_matrix[nx - 1, j]
        n2 = node_matrix[nx - 1, j + 1]
        edges[edge_id] = {
            'type': 1,
            'tags': [2],  # Physical ID 2 = Gamma_D
            'nodes': [n1, n2]
        }
        edge_id += 1

    print(f"[OK] Aretes Gamma_D (Dirichlet) generees: {ny - 1}")

    # Bord interieur (Gamma_N - Neumann)
    # Physical ID 3
    for i in range(nx - 1):
        n1 = node_matrix[i, 0]
        n2 = node_matrix[i + 1, 0]
        edges[edge_id] = {
            'type': 1,
            'tags': [3],  # Physical ID 3 = Gamma_N
            'nodes': [n1, n2]
        }
        edge_id += 1

    print(f"[OK] Aretes Gamma_N (Neumann) generees: {nx - 1}")

    # Bord pointe (axe)
    # Physical ID 4
    for j in range(ny - 1):
        n1 = node_matrix[0, j]
        n2 = node_matrix[0, j + 1]
        edges[edge_id] = {
            'type': 1,
            'tags': [4],  # Physical ID 4 = Gamma_axis
            'nodes': [n1, n2]
        }
        edge_id += 1

    print(f"[OK] Aretes Gamma_axis generees: {ny - 1}")

    # ========================================================================
    # ECRITURE DU FICHIER .msh (FORMAT 2.2)
    # ========================================================================

    with open(output_file, 'w') as f:
        # Header
        f.write("$MeshFormat\n")
        f.write("2.2 0 8\n")
        f.write("$EndMeshFormat\n")

        # Physical names
        f.write("$PhysicalNames\n")
        f.write("5\n")
        f.write('1 1 "Gamma_F"\n')
        f.write('1 2 "Gamma_D"\n')
        f.write('1 3 "Gamma_N"\n')
        f.write('1 4 "Gamma_axis"\n')
        f.write('2 10 "Omega"\n')
        f.write("$EndPhysicalNames\n")

        # Nodes
        f.write("$Nodes\n")
        f.write(f"{len(nodes)}\n")
        for nid, coords in nodes.items():
            f.write(f"{nid} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f}\n")
        f.write("$EndNodes\n")

        # Elements
        all_elements = {**edges, **triangles}
        f.write("$Elements\n")
        f.write(f"{len(all_elements)}\n")
        for eid, elem in all_elements.items():
            elem_type = elem['type']
            tags = elem['tags']
            nodes_list = elem['nodes']
            num_tags = len(tags)

            # Format: elem_id elem_type num_tags tags... nodes...
            f.write(f"{eid} {elem_type} {num_tags} {' '.join(map(str, tags))} {' '.join(map(str, nodes_list))}\n")
        f.write("$EndElements\n")

    print(f"[OK] Maillage ecrit dans: {output_file}")
    print("=" * 70)
    print(f"RESUME:")
    print(f"  - Noeuds:    {len(nodes)}")
    print(f"  - Triangles: {len(triangles)}")
    print(f"  - Aretes:    {len(edges)}")
    print(f"  - Total:     {len(all_elements)} elements")
    print("=" * 70)


if __name__ == '__main__':
    generate_rocket_mesh()
    print("\n[OK] Maillage genere avec succes!")
    print("Vous pouvez maintenant lancer: python main.py")
