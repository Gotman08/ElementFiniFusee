"""
Module de lecture de maillage GMSH format .msh version 2.2
Lit les noeuds, éléments et groupes physiques
"""
import numpy as np
from typing import Dict, List, Tuple

class Mesh:
    """Classe représentant un maillage éléments finis"""

    def __init__(self):
        self.nodes = {}  # {node_id: [x, y, z]}
        self.elements = {}  # {elem_id: {'type': int, 'tags': [], 'nodes': []}}
        self.physical_groups = {}  # {physical_id: name}
        self.dimension = 2  # Par défaut 2D

    def get_triangles(self):
        """Retourne les éléments triangulaires (type 2 dans GMSH)"""
        return {eid: elem for eid, elem in self.elements.items()
                if elem['type'] == 2}

    def get_edges(self):
        """Retourne les éléments arêtes (type 1 dans GMSH)"""
        return {eid: elem for eid, elem in self.elements.items()
                if elem['type'] == 1}

    def get_boundary_edges_by_physical(self, physical_id):
        """Retourne les arêtes appartenant à un groupe physique donné"""
        edges = []
        for eid, elem in self.get_edges().items():
            if len(elem['tags']) > 0 and elem['tags'][0] == physical_id:
                edges.append((eid, elem))
        return edges

    def get_node_coords(self, node_ids):
        """Retourne les coordonnées d'une liste de noeuds"""
        coords = np.array([self.nodes[nid][:self.dimension] for nid in node_ids])
        return coords


def read_gmsh_mesh(filename: str) -> Mesh:
    """
    Lit un fichier maillage GMSH format .msh (version 2.2)

    Args:
        filename: Chemin vers le fichier .msh

    Returns:
        Objet Mesh contenant le maillage
    """
    mesh = Mesh()

    with open(filename, 'r') as f:
        line = f.readline()

        while line:
            line = line.strip()

            # Lecture des noeuds
            if line == '$Nodes':
                num_nodes = int(f.readline().strip())
                for _ in range(num_nodes):
                    parts = f.readline().strip().split()
                    node_id = int(parts[0])
                    coords = [float(parts[1]), float(parts[2]), float(parts[3])]
                    mesh.nodes[node_id] = coords
                line = f.readline()  # $EndNodes

            # Lecture des éléments
            elif line == '$Elements':
                num_elements = int(f.readline().strip())
                for _ in range(num_elements):
                    parts = f.readline().strip().split()
                    elem_id = int(parts[0])
                    elem_type = int(parts[1])
                    num_tags = int(parts[2])
                    tags = [int(parts[3 + i]) for i in range(num_tags)]
                    nodes = [int(parts[3 + num_tags + i])
                            for i in range(len(parts) - 3 - num_tags)]

                    mesh.elements[elem_id] = {
                        'type': elem_type,
                        'tags': tags,
                        'nodes': nodes
                    }
                line = f.readline()  # $EndElements

            # Lecture des groupes physiques (optionnel)
            elif line == '$PhysicalNames':
                num_groups = int(f.readline().strip())
                for _ in range(num_groups):
                    parts = f.readline().strip().split()
                    dimension = int(parts[0])
                    physical_id = int(parts[1])
                    name = parts[2].strip('"')
                    mesh.physical_groups[physical_id] = name
                    if dimension > mesh.dimension:
                        mesh.dimension = dimension
                line = f.readline()  # $EndPhysicalNames

            else:
                line = f.readline()

    return mesh


def create_node_mapping(mesh: Mesh) -> Tuple[Dict[int, int], int]:
    """
    Crée un mapping entre IDs de noeuds GMSH et indices DOF

    Args:
        mesh: Objet Mesh

    Returns:
        (node_to_dof, num_dofs): Dictionnaire de mapping et nombre total de DOFs
    """
    # Extraire tous les noeuds uniques des éléments triangulaires
    all_nodes = set()
    for elem in mesh.get_triangles().values():
        all_nodes.update(elem['nodes'])

    # Créer le mapping : node_id -> index DOF
    node_to_dof = {node_id: idx for idx, node_id in enumerate(sorted(all_nodes))}

    return node_to_dof, len(node_to_dof)


if __name__ == '__main__':
    # Test basique de lecture
    import sys
    if len(sys.argv) > 1:
        mesh = read_gmsh_mesh(sys.argv[1])
        print(f"Maillage lu avec succès:")
        print(f"  - {len(mesh.nodes)} noeuds")
        print(f"  - {len(mesh.get_triangles())} triangles")
        print(f"  - {len(mesh.get_edges())} arêtes")
        print(f"  - Dimension: {mesh.dimension}D")
        if mesh.physical_groups:
            print(f"  - Groupes physiques: {mesh.physical_groups}")
