"""
@file ariane5_core_mesh.py
@brief Mesh generator for Ariane 5 central body (without boosters)
@details Generates triangular FEM mesh for atmospheric reentry simulation
         in tail-first mode (base forward for retropropulsive braking)

Geometry (vertical rocket):
- Jupe Vulcain (base): H=0.15m, R=0.09m (narrower)
- Reservoir (body): H=2.0m, R=0.15m
- Ogive (nose): H=0.6m, conical to tip

Boundary conditions (tail-first reentry):
- Gamma_F (Robin): BASE only (y=0) - heated zone
- Gamma_N (Neumann): flanks + ogive (in wake, insulated)
"""

import numpy as np


def compute_triangle_quality(coords):
    """Compute triangle quality Q_T = (sqrt(3)/6) * (h_T / r_T)."""
    a = np.linalg.norm(coords[1] - coords[0])
    b = np.linalg.norm(coords[2] - coords[1])
    c = np.linalg.norm(coords[0] - coords[2])

    h_T = max(a, b, c)
    s = (a + b + c) / 2
    area_sq = s * (s - a) * (s - b) * (s - c)

    if area_sq <= 0:
        return float('inf')

    area = np.sqrt(area_sq)
    r_T = area / s
    Q_T = (np.sqrt(3) / 6) * (h_T / r_T)

    return Q_T


def generate_ariane5_core_mesh(output_file="rocket_mesh.msh",
                               # Ariane 5 geometry parameters
                               H_nose=0.6,
                               H_body=2.0,
                               R_core=0.15,
                               H_skirt=0.15,
                               R_skirt=0.09,
                               thickness=0.03,
                               # Mesh parameters
                               nr=5,
                               auto_ny=True):
    """
    Generate Ariane 5 central body mesh (without boosters) for tail-first reentry.

    Args:
        output_file: Output .msh file path
        H_nose: Nose cone height [m]
        H_body: Reservoir body height [m]
        R_core: Body radius [m]
        H_skirt: Vulcain engine skirt height [m]
        R_skirt: Skirt radius [m]
        thickness: Wall thickness [m]
        nr: Number of radial divisions (through wall thickness)
        auto_ny: Auto-calculate vertical divisions for equilateral triangles

    Returns:
        Dictionary with mesh statistics
    """
    print("=" * 70)
    print("ARIANE 5 - CORPS CENTRAL (sans boosters)")
    print("Mode: TAIL-FIRST (rentree atmospherique)")
    print("=" * 70)

    H_total = H_skirt + H_body + H_nose

    # Spatial step from wall thickness
    h = thickness / (nr - 1)
    print(f"\nPas spatial h = {h*1000:.2f} mm")

    # Auto-calculate vertical divisions
    if auto_ny:
        ny_total = int(np.ceil(H_total / h)) + 1
    else:
        ny_total = 100

    print(f"Divisions verticales: ny = {ny_total}")
    print(f"Divisions radiales: nr = {nr}")

    # Y coordinates (vertical axis)
    y_coords = np.linspace(0, H_total, ny_total)

    # Define the outer radius profile
    def get_outer_radius(y):
        """Return outer radius at height y."""
        if y <= H_skirt:
            # Skirt zone: constant R_skirt
            return R_skirt
        elif y <= H_skirt + H_body:
            # Body zone: constant R_core
            return R_core
        else:
            # Ogive zone: linear taper to tip
            y_ogive = y - (H_skirt + H_body)
            ratio = 1.0 - (y_ogive / H_nose)
            return R_core * max(0.01, ratio)

    # Generate nodes
    nodes = {}
    node_id = 1
    node_matrix = np.zeros((ny_total, nr), dtype=int)

    for i, y in enumerate(y_coords):
        R_outer = get_outer_radius(y)
        R_inner = max(0.005, R_outer - thickness)

        for j in range(nr):
            ratio_r = j / (nr - 1) if nr > 1 else 0
            x = R_inner + ratio_r * (R_outer - R_inner)

            nodes[node_id] = [x, y, 0.0]
            node_matrix[i, j] = node_id
            node_id += 1

    print(f"\n[OK] Noeuds generes: {len(nodes)}")

    # Generate triangles
    triangles = {}
    elem_id = 1
    quality_list = []

    for i in range(ny_total - 1):
        for j in range(nr - 1):
            n1 = node_matrix[i, j]
            n2 = node_matrix[i + 1, j]
            n3 = node_matrix[i + 1, j + 1]
            n4 = node_matrix[i, j + 1]

            # First triangle
            triangles[elem_id] = {
                'type': 2,
                'tags': [10],  # Omega domain
                'nodes': [n1, n2, n3]
            }
            coords = np.array([nodes[n1][:2], nodes[n2][:2], nodes[n3][:2]])
            quality_list.append(compute_triangle_quality(coords))
            elem_id += 1

            # Second triangle
            triangles[elem_id] = {
                'type': 2,
                'tags': [10],
                'nodes': [n1, n3, n4]
            }
            coords = np.array([nodes[n1][:2], nodes[n3][:2], nodes[n4][:2]])
            quality_list.append(compute_triangle_quality(coords))
            elem_id += 1

    print(f"[OK] Triangles generes: {len(triangles)}")

    # Quality metrics
    Q_mean = np.mean(quality_list)
    Q_max = np.max(quality_list)
    Q_min = np.min(quality_list)

    print(f"\n[QUALITE DES TRIANGLES]")
    print(f"  Q_T: min={Q_min:.2f}, moy={Q_mean:.2f}, max={Q_max:.2f}")
    print(f"  (Q_T = 1 pour triangle equilateral)")

    # Generate boundary edges for TAIL-FIRST mode
    edges = {}
    edge_id = elem_id

    print(f"\n[CONDITIONS AUX LIMITES - MODE TAIL-FIRST]")

    # ==========================================================================
    # Gamma_F (Robin): BASE ONLY (y=0) - heated zone
    # ==========================================================================
    base_edge_count = 0
    for j in range(nr - 1):
        n1 = node_matrix[0, j]
        n2 = node_matrix[0, j + 1]
        edges[edge_id] = {
            'type': 1,
            'tags': [1],  # Physical ID 1 = Gamma_F (Robin)
            'nodes': [n1, n2]
        }
        edge_id += 1
        base_edge_count += 1
    print(f"  [Gamma_F] BASE (Robin, flux entrant): {base_edge_count} aretes")

    # ==========================================================================
    # Gamma_N (Neumann): External surface (flanks + ogive) - insulated (in wake)
    # ==========================================================================
    outer_edge_count = 0
    for i in range(ny_total - 1):
        n1 = node_matrix[i, nr - 1]
        n2 = node_matrix[i + 1, nr - 1]
        edges[edge_id] = {
            'type': 1,
            'tags': [3],  # Physical ID 3 = Gamma_N (Neumann)
            'nodes': [n1, n2]
        }
        edge_id += 1
        outer_edge_count += 1
    print(f"  [Gamma_N] FLANCS EXTERNES (Neumann, isoles): {outer_edge_count} aretes")

    # ==========================================================================
    # Gamma_N (Neumann): Internal surface - insulated
    # ==========================================================================
    inner_edge_count = 0
    for i in range(ny_total - 1):
        n1 = node_matrix[i, 0]
        n2 = node_matrix[i + 1, 0]
        edges[edge_id] = {
            'type': 1,
            'tags': [3],  # Physical ID 3 = Gamma_N (Neumann)
            'nodes': [n1, n2]
        }
        edge_id += 1
        inner_edge_count += 1
    print(f"  [Gamma_N] FLANCS INTERNES (Neumann, isoles): {inner_edge_count} aretes")

    # ==========================================================================
    # Gamma_N (Neumann): Top (ogive tip) - insulated
    # ==========================================================================
    top_edge_count = 0
    for j in range(nr - 1):
        n1 = node_matrix[ny_total - 1, j]
        n2 = node_matrix[ny_total - 1, j + 1]
        edges[edge_id] = {
            'type': 1,
            'tags': [3],  # Physical ID 3 = Gamma_N (Neumann)
            'nodes': [n1, n2]
        }
        edge_id += 1
        top_edge_count += 1
    print(f"  [Gamma_N] OGIVE (Neumann, sillage): {top_edge_count} aretes")

    total_edges = base_edge_count + outer_edge_count + inner_edge_count + top_edge_count
    print(f"  TOTAL aretes: {total_edges}")

    # Write GMSH MSH 2.2 file
    with open(output_file, 'w') as f:
        f.write("$MeshFormat\n")
        f.write("2.2 0 8\n")
        f.write("$EndMeshFormat\n")

        f.write("$PhysicalNames\n")
        f.write("4\n")
        f.write('1 1 "Gamma_F"\n')  # Robin (heated base)
        f.write('1 2 "Gamma_D"\n')  # Dirichlet (unused but kept for compatibility)
        f.write('1 3 "Gamma_N"\n')  # Neumann (insulated flanks)
        f.write('2 10 "Omega"\n')   # Domain
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
    print("ARIANE 5 CORPS CENTRAL - RESUME")
    print("=" * 70)
    print(f"  Geometrie:")
    print(f"    - H_nose  = {H_nose} m (ogive)")
    print(f"    - H_body  = {H_body} m (reservoir)")
    print(f"    - H_skirt = {H_skirt} m (jupe Vulcain)")
    print(f"    - R_core  = {R_core} m (rayon corps)")
    print(f"    - R_skirt = {R_skirt} m (rayon jupe)")
    print(f"    - H_total = {H_total} m")
    print(f"  Maillage:")
    print(f"    - Noeuds:    {len(nodes)}")
    print(f"    - Triangles: {len(triangles)}")
    print(f"    - Aretes:    {total_edges}")
    print(f"    - Qualite Q_T moyenne: {Q_mean:.2f}")
    print(f"  Conditions aux limites (TAIL-FIRST):")
    print(f"    - Gamma_F (Robin): BASE (y=0) - chauffee")
    print(f"    - Gamma_N (Neumann): flancs + ogive - isoles")
    print("=" * 70)

    return {
        'n_nodes': len(nodes),
        'n_triangles': len(triangles),
        'n_edges': total_edges,
        'Q_mean': Q_mean,
        'Q_max': Q_max,
        'H_total': H_total
    }


if __name__ == '__main__':
    import sys

    output = "rocket_mesh.msh"
    if len(sys.argv) > 1:
        output = sys.argv[1]

    stats = generate_ariane5_core_mesh(output_file=output)
    print(f"\n[OK] Maillage Ariane 5 corps central genere!")
    print(f"Pour lancer la simulation: python scripts/demo_reentry.py")
