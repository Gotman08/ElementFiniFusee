"""
@file rocket_mesh.py
@brief Pure Python mesh generator for rocket thermal analysis
@author HPC-Code-Documenter
@date 2025

@details
This module generates structured triangular meshes for rocket geometry
without requiring external GMSH dependencies. It creates meshes suitable
for 2D axisymmetric thermal analysis of rocket nose cones and body sections.

Key features:
- Automatic mesh refinement for quasi-equilateral triangles ($Q_T \\approx 1$)
- Support for "nose-first" and "tail-first" reentry configurations
- Triangle quality metrics based on course criteria
- GMSH MSH 2.2 format output

Mesh quality criterion from course:
$Q_T = \\frac{\\sqrt{3}}{6} \\times \\frac{h_T}{r_T}$

where $h_T$ is the longest edge and $r_T$ is the inscribed circle radius.
$Q_T = 1$ for equilateral triangles.

Boundary conditions setup:
- Physical ID 1 (Gamma_F): Robin boundary (convective flux)
- Physical ID 2 (Gamma_D): Dirichlet boundary (fixed temperature)
- Physical ID 3 (Gamma_N): Neumann boundary (insulated/zero flux)
- Physical ID 4 (Gamma_axis): Axis/tip boundary
- Physical ID 10 (Omega): Domain interior
"""
import numpy as np


def compute_triangle_quality(coords):
    """
    @brief Compute triangle quality using the course-defined criterion.

    @details
    Calculates the quality factor $Q_T$ for a triangle:
    $Q_T = \\frac{\\sqrt{3}}{6} \\times \\frac{h_T}{r_T}$

    where:
    - $h_T$ is the longest edge length
    - $r_T$ is the inscribed circle radius: $r_T = A/s$
    - $A$ is the triangle area (Heron's formula)
    - $s$ is the semi-perimeter

    @param coords: NumPy array of shape (3, 2) with vertex coordinates

    @return Quality factor $Q_T$ (1.0 for equilateral, >1 for degraded triangles)
        Returns infinity for degenerate triangles
    """
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


def compute_aspect_ratio(coords):
    """
    @brief Compute the aspect ratio of a triangle.

    @details
    Calculates the ratio of the longest to shortest edge:
    $AR = \\frac{\\max(a, b, c)}{\\min(a, b, c)}$

    An equilateral triangle has $AR = 1$.

    @param coords: NumPy array of shape (3, 2) with vertex coordinates

    @return Aspect ratio (1.0 for equilateral triangles)
    """
    a = np.linalg.norm(coords[1] - coords[0])
    b = np.linalg.norm(coords[2] - coords[1])
    c = np.linalg.norm(coords[0] - coords[2])

    sides = [a, b, c]
    return max(sides) / max(min(sides), 1e-12)


def generate_rocket_mesh(output_file="rocket_mesh.msh",
                        L_nose=0.5,
                        L_body=2.0,
                        R_body=0.3,
                        thickness=0.05,
                        ny=8,
                        mode="tail-first",
                        auto_nx=True):
    """
    @brief Generate a high-quality triangular mesh for rocket wall geometry.

    @details
    Creates a structured triangular mesh representing the rocket wall cross-section.
    The mesh is optimized for thermal FEM analysis with:

    - Automatic calculation of $n_x$ to achieve $\\Delta x \\approx \\Delta y$
      (quasi-equilateral triangles with $Q_T \\approx 1$)
    - Proper boundary identification for different physical groups
    - Support for two reentry modes with different boundary condition setups

    Geometry description:
    - Conical nose section of length $L_{nose}$
    - Cylindrical body section of length $L_{body}$
    - Wall thickness from inner to outer radius

    MODE "nose-first" (classical reentry, nose forward):
    - Gamma_F (Robin): External surface (flanks + nose) receives aerodynamic heating
    - Gamma_D (Dirichlet): Base at fixed temperature (cooled)
    - Gamma_N (Neumann): Internal surface (insulated)

    MODE "tail-first" (retropropulsive reentry, base forward):
    - Gamma_F (Robin): Base receives aerodynamic heating (attack zone)
    - Gamma_N (Neumann): Flanks + nose + interior (in wake, insulated)

    @param output_file: Path for output .msh file
    @param L_nose: Nose cone length [m]
    @param L_body: Cylindrical body length [m]
    @param R_body: Body outer radius [m]
    @param thickness: Wall thickness [m]
    @param ny: Number of radial divisions (through thickness)
    @param mode: Reentry mode - "nose-first" or "tail-first"
    @param auto_nx: If True, compute $n_x$ automatically for optimal quality

    @return Dictionary with mesh statistics:
        - 'nx', 'ny': Grid dimensions
        - 'n_nodes', 'n_triangles', 'n_edges': Element counts
        - 'Q_mean', 'Q_max': Quality metrics
        - 'mode': Reentry mode used

    @example
    >>> stats = generate_rocket_mesh(
    ...     output_file="rocket.msh",
    ...     L_nose=0.5, L_body=2.0,
    ...     mode="tail-first"
    ... )
    >>> print(f"Generated mesh with {stats['n_nodes']} nodes")
    """
    print("=" * 70)
    print("GENERATION DU MAILLAGE PYTHON - QUALITE AMELIOREE")
    print("=" * 70)
    print(f"Mode de rentree: {mode.upper()}")

    L_total = L_nose + L_body

    delta_y = thickness / (ny - 1)

    if auto_nx:
        nx = int(np.ceil(L_total / delta_y)) + 1
        delta_x = L_total / (nx - 1)
        print(f"\n[QUALITE] Calcul automatique de nx:")
        print(f"  - Delta_y (radial) = {delta_y*1000:.2f} mm")
        print(f"  - Delta_x (longitudinal) = {delta_x*1000:.2f} mm")
        print(f"  - Ratio Delta_x/Delta_y = {delta_x/delta_y:.2f} (objectif: 1.0)")
        print(f"  - nx calcule = {nx}")
    else:
        nx = 25
        delta_x = L_total / (nx - 1)
        print(f"\n[WARNING] nx fixe a {nx} - triangles allonges!")
        print(f"  - Ratio Delta_x/Delta_y = {delta_x/delta_y:.1f}")

    nodes = {}
    node_id = 1

    x_coords = np.linspace(0, L_total, nx)

    R_inner = R_body - thickness
    y_coords = np.linspace(R_inner, R_body, ny)

    def rocket_profile(x, R_body, L_nose):
        """
        @brief Compute the outer radius at axial position x.

        @param x: Axial coordinate [m]
        @param R_body: Body radius [m]
        @param L_nose: Nose cone length [m]

        @return Outer radius at position x [m]
        """
        if x <= L_nose:
            return R_body * (x / L_nose)
        else:
            return R_body

    node_matrix = np.zeros((nx, ny), dtype=int)

    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            if j == ny - 1:
                y_actual = rocket_profile(x, R_body, L_nose)
            elif j == 0:
                y_actual = max(0.01, rocket_profile(x, R_body, L_nose) - thickness)
            else:
                ratio = j / (ny - 1)
                y_outer = rocket_profile(x, R_body, L_nose)
                y_inner = max(0.01, y_outer - thickness)
                y_actual = y_inner + ratio * (y_outer - y_inner)

            nodes[node_id] = [x, y_actual, 0.0]
            node_matrix[i, j] = node_id
            node_id += 1

    print(f"\n[OK] Noeuds generes: {len(nodes)}")

    triangles = {}
    elem_id = 1
    quality_list = []
    aspect_ratio_list = []

    for i in range(nx - 1):
        for j in range(ny - 1):
            n1 = node_matrix[i, j]
            n2 = node_matrix[i + 1, j]
            n3 = node_matrix[i + 1, j + 1]
            n4 = node_matrix[i, j + 1]

            triangles[elem_id] = {
                'type': 2,
                'tags': [10],
                'nodes': [n1, n2, n3]
            }

            coords1 = np.array([nodes[n1][:2], nodes[n2][:2], nodes[n3][:2]])
            quality_list.append(compute_triangle_quality(coords1))
            aspect_ratio_list.append(compute_aspect_ratio(coords1))
            elem_id += 1

            triangles[elem_id] = {
                'type': 2,
                'tags': [10],
                'nodes': [n1, n3, n4]
            }

            coords2 = np.array([nodes[n1][:2], nodes[n3][:2], nodes[n4][:2]])
            quality_list.append(compute_triangle_quality(coords2))
            aspect_ratio_list.append(compute_aspect_ratio(coords2))
            elem_id += 1

    print(f"[OK] Triangles generes: {len(triangles)}")

    Q_mean = np.mean(quality_list)
    Q_max = np.max(quality_list)
    Q_min = np.min(quality_list)
    AR_mean = np.mean(aspect_ratio_list)
    AR_max = np.max(aspect_ratio_list)

    print(f"\n[QUALITE DES TRIANGLES]")
    print(f"  Critere Q_T (cours): min={Q_min:.2f}, moy={Q_mean:.2f}, max={Q_max:.2f}")
    print(f"  (Q_T = 1 pour triangle equilateral)")
    print(f"  Ratio d'aspect: moy={AR_mean:.2f}, max={AR_max:.2f}")

    if Q_mean < 2.0:
        print(f"  QUALITE ACCEPTABLE (Q_moy < 2)")
    else:
        print(f"  QUALITE INSUFFISANTE (Q_moy >= 2)")

    edges = {}
    edge_id = elem_id

    if mode == "tail-first":
        print(f"\n[CONDITIONS AUX LIMITES - MODE TAIL-FIRST]")

        for j in range(ny - 1):
            n1 = node_matrix[nx - 1, j]
            n2 = node_matrix[nx - 1, j + 1]
            edges[edge_id] = {
                'type': 1,
                'tags': [1],
                'nodes': [n1, n2]
            }
            edge_id += 1
        print(f"  [Gamma_F] BASE (Robin, flux entrant): {ny - 1} aretes")

        for i in range(nx - 1):
            n1 = node_matrix[i, ny - 1]
            n2 = node_matrix[i + 1, ny - 1]
            edges[edge_id] = {
                'type': 1,
                'tags': [3],
                'nodes': [n1, n2]
            }
            edge_id += 1
        print(f"  [Gamma_N] FLANCS (Neumann, isoles): {nx - 1} aretes")

        for i in range(nx - 1):
            n1 = node_matrix[i, 0]
            n2 = node_matrix[i + 1, 0]
            edges[edge_id] = {
                'type': 1,
                'tags': [3],
                'nodes': [n1, n2]
            }
            edge_id += 1
        print(f"  [Gamma_N] INTERIEUR (Neumann, isole): {nx - 1} aretes")

        for j in range(ny - 1):
            n1 = node_matrix[0, j]
            n2 = node_matrix[0, j + 1]
            edges[edge_id] = {
                'type': 1,
                'tags': [4],
                'nodes': [n1, n2]
            }
            edge_id += 1
        print(f"  [Gamma_axis] OGIVE (Neumann, sillage): {ny - 1} aretes")

    else:
        print(f"\n[CONDITIONS AUX LIMITES - MODE NOSE-FIRST]")

        for i in range(nx - 1):
            n1 = node_matrix[i, ny - 1]
            n2 = node_matrix[i + 1, ny - 1]
            edges[edge_id] = {
                'type': 1,
                'tags': [1],
                'nodes': [n1, n2]
            }
            edge_id += 1
        print(f"  [Gamma_F] SURFACE EXT (Robin): {nx - 1} aretes")

        for j in range(ny - 1):
            n1 = node_matrix[nx - 1, j]
            n2 = node_matrix[nx - 1, j + 1]
            edges[edge_id] = {
                'type': 1,
                'tags': [2],
                'nodes': [n1, n2]
            }
            edge_id += 1
        print(f"  [Gamma_D] BASE (Dirichlet): {ny - 1} aretes")

        for i in range(nx - 1):
            n1 = node_matrix[i, 0]
            n2 = node_matrix[i + 1, 0]
            edges[edge_id] = {
                'type': 1,
                'tags': [3],
                'nodes': [n1, n2]
            }
            edge_id += 1
        print(f"  [Gamma_N] INTERIEUR (Neumann): {nx - 1} aretes")

        for j in range(ny - 1):
            n1 = node_matrix[0, j]
            n2 = node_matrix[0, j + 1]
            edges[edge_id] = {
                'type': 1,
                'tags': [4],
                'nodes': [n1, n2]
            }
            edge_id += 1
        print(f"  [Gamma_axis] OGIVE (Neumann): {ny - 1} aretes")

    with open(output_file, 'w') as f:
        f.write("$MeshFormat\n")
        f.write("2.2 0 8\n")
        f.write("$EndMeshFormat\n")

        f.write("$PhysicalNames\n")
        f.write("5\n")
        f.write('1 1 "Gamma_F"\n')
        f.write('1 2 "Gamma_D"\n')
        f.write('1 3 "Gamma_N"\n')
        f.write('1 4 "Gamma_axis"\n')
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

    print(f"\n[OK] Maillage ecrit dans: {output_file}")
    print("=" * 70)
    print(f"RESUME:")
    print(f"  - Mode:      {mode}")
    print(f"  - Noeuds:    {len(nodes)}")
    print(f"  - Triangles: {len(triangles)}")
    print(f"  - Aretes:    {len(edges)}")
    print(f"  - Qualite Q_T moyenne: {Q_mean:.2f}")
    print(f"  - nx x ny = {nx} x {ny}")
    print("=" * 70)

    return {
        'nx': nx,
        'ny': ny,
        'n_nodes': len(nodes),
        'n_triangles': len(triangles),
        'n_edges': len(edges),
        'Q_mean': Q_mean,
        'Q_max': Q_max,
        'mode': mode
    }


if __name__ == '__main__':
    import sys

    mode = "tail-first"
    if len(sys.argv) > 1:
        mode = sys.argv[1]

    print(f"\nGeneration du maillage en mode: {mode}")
    print("(Utilisation: python generate_mesh_python.py [nose-first|tail-first])\n")

    stats = generate_rocket_mesh(mode=mode)

    print("\n[OK] Maillage genere avec succes!")
    print("Vous pouvez maintenant lancer: python main.py")
