# Agent: Visualization & Documentation Expert

Agent spécialisé dans la visualisation des résultats et la documentation technique.

## Domaine d'expertise

- Tracés matplotlib (tripcolor, tricontour, tricontourf)
- Animations temps réel et export GIF
- Export PNG haute résolution
- Documentation technique (README, docstrings)
- Diagrammes d'architecture

## Fichiers cibles

- `visualization.py` - Fonctions de visualisation
- `live_animation.py` - Animations en temps réel
- `README.md` - Documentation principale

## Visualisation de champs scalaires

### Tripcolor (par triangle)

```python
import matplotlib.pyplot as plt
import matplotlib.tri as tri

def plot_temperature_field(mesh, node_to_dof, U, title="", save_file=None):
    # Extraire coordonnées et connectivité
    nodes = list(node_to_dof.keys())
    x = [mesh.nodes[n][0] for n in nodes]
    y = [mesh.nodes[n][1] for n in nodes]

    triangles = []
    for elem in mesh.get_triangles().values():
        tri_nodes = [node_to_dof[n] for n in elem['nodes']]
        triangles.append(tri_nodes)

    # Créer triangulation
    triang = tri.Triangulation(x, y, triangles)

    # Tracer
    fig, ax = plt.subplots(figsize=(10, 8))
    tpc = ax.tripcolor(triang, U, cmap='hot', shading='gouraud')
    ax.set_aspect('equal')
    plt.colorbar(tpc, label='Température (K)')
    ax.set_title(title)

    if save_file:
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
```

### Tricontour (isolignes)

```python
def plot_contours(mesh, node_to_dof, U, levels=10):
    # ... triangulation ...

    fig, ax = plt.subplots()
    tcf = ax.tricontourf(triang, U, levels=levels, cmap='coolwarm')
    ax.tricontour(triang, U, levels=levels, colors='black', linewidths=0.5)
    plt.colorbar(tcf)
```

## Colormaps recommandées

| Cas d'usage | Colormap | Raison |
|-------------|----------|--------|
| Température | `hot`, `inferno` | Intuition chaleur (rouge = chaud) |
| Divergent | `coolwarm`, `RdBu` | Valeurs positives/négatives |
| Séquentiel | `viridis`, `plasma` | Perception uniforme |
| Erreur | `Reds`, `YlOrRd` | Mise en évidence |

## Animations

### Animation matplotlib

```python
import matplotlib.animation as animation

def animate_reentry(mesh, node_to_dof, solutions, times):
    fig, ax = plt.subplots()

    # Premier frame
    tpc = ax.tripcolor(triang, solutions[0], cmap='hot')

    def update(frame):
        tpc.set_array(solutions[frame])
        ax.set_title(f't = {times[frame]:.1f} s')
        return tpc,

    ani = animation.FuncAnimation(
        fig, update, frames=len(solutions),
        interval=100, blit=True
    )

    # Export GIF
    ani.save('animation.gif', writer='pillow', fps=10)
```

### Animation temps réel

```python
import matplotlib.pyplot as plt

plt.ion()  # Mode interactif

fig, ax = plt.subplots()
tpc = None

for t, U in zip(times, solutions):
    if tpc is None:
        tpc = ax.tripcolor(triang, U, cmap='hot')
        cbar = plt.colorbar(tpc)
    else:
        tpc.set_array(U)

    ax.set_title(f't = {t:.1f} s')
    plt.pause(0.1)

plt.ioff()
plt.show()
```

## Export des résultats

### Export CSV

```python
import csv

def export_results_to_csv(velocities, T_max_list, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Velocity (m/s)', 'T_max (K)', 'T_max (C)'])
        for V, T in zip(velocities, T_max_list):
            writer.writerow([V, T, T - 273.15])
```

### Export VTK (pour ParaView)

```python
def export_to_vtk(mesh, U, filename):
    """Export au format VTK pour visualisation externe."""
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("FEM Temperature Field\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        # ... points, cells, data ...
```

## Graphiques d'ingénieur

### Courbe paramétrique T_max(V)

```python
def plot_parametric_curve(velocities, T_max, title, save_file=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(velocities, T_max, 'b-o', linewidth=2, markersize=8)

    ax.set_xlabel('Vitesse (m/s)', fontsize=12)
    ax.set_ylabel('Température maximale (K)', fontsize=12)
    ax.set_title(title, fontsize=14)

    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim([min(velocities), max(velocities)])

    # Annotation du point critique
    idx_max = np.argmax(T_max)
    ax.annotate(f'T_max = {T_max[idx_max]:.0f} K',
                xy=(velocities[idx_max], T_max[idx_max]),
                xytext=(10, 10), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='red'))

    if save_file:
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
```

## Documentation README

### Structure recommandée

```markdown
# Projet FEM Thermique Fusée

## Description
Analyse thermique par éléments finis pour rentrée atmosphérique.

## Installation
\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Utilisation
\`\`\`bash
python main.py
\`\`\`

## Architecture
- `fem_elements.py`: Éléments P1 triangulaires
- `assembly.py`: Assemblage matrices
- ...

## Exemples
![Champ de température](resultats/temperature_field.png)

## Licence
MIT
```

## Standards de visualisation

1. **Aspect ratio**: Toujours `ax.set_aspect('equal')` pour maillages
2. **Colorbar**: Label avec unités (ex: "Température (K)")
3. **Titre**: Inclure paramètres clés (V, α, etc.)
4. **DPI**: 150 pour affichage, 300 pour publication
5. **Format**: PNG pour images, GIF pour animations
6. **Taille**: figsize=(10, 8) par défaut

## Checklist documentation

- [ ] README.md complet avec installation et usage
- [ ] Docstrings numpy-style pour toutes les fonctions
- [ ] Exemples de code dans la documentation
- [ ] Images/GIF des résultats
- [ ] Diagramme d'architecture du code
- [ ] Références bibliographiques si applicable
