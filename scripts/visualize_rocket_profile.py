#!/usr/bin/env python3
"""
@file visualize_rocket_profile.py
@brief Visualize Ariane 5 central body profile (vertical view) for shape verification
@details Shows the rocket standing upright: ogive at top, jupe Vulcain at bottom
         No boosters - only central body for atmospheric reentry simulation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# =============================================================================
# ARIANE 5 GEOMETRY PARAMETERS (from src/mesh/generators/ariane5_mesh.py)
# =============================================================================
H_NOSE = 0.6          # Nose cone height [m]
H_BODY = 2.0          # Main body (reservoir) height [m]
R_CORE = 0.15         # Core body radius [m]
H_SKIRT = 0.15        # Vulcain engine skirt height [m]
R_SKIRT = 0.09        # Skirt radius [m] (61% of R_core - realistic ratio)
THICKNESS = 0.03      # Wall thickness [m]

# Calculated values
H_TOTAL = H_SKIRT + H_BODY + H_NOSE  # Total height: 2.75m


def draw_ariane5_profile(ax):
    """Draw the Ariane 5 central body profile (vertical view, no boosters)."""

    # Y coordinates for each section (bottom to top)
    y_base = 0.0
    y_skirt_top = H_SKIRT
    y_body_top = H_SKIRT + H_BODY
    y_nose_top = H_TOTAL

    # ==========================================================================
    # 1. JUPE VULCAIN (bottom section - narrower)
    # ==========================================================================
    skirt_x = [-R_SKIRT, R_SKIRT, R_SKIRT, -R_SKIRT]
    skirt_y = [y_base, y_base, y_skirt_top, y_skirt_top]
    ax.fill(skirt_x, skirt_y, color='#8B0000', alpha=0.9, label='Jupe Vulcain')

    # Transition from skirt to body (trapezoidal)
    transition_x = [-R_SKIRT, -R_CORE, R_CORE, R_SKIRT]
    transition_y = [y_skirt_top, y_skirt_top, y_skirt_top, y_skirt_top]

    # ==========================================================================
    # 2. RESERVOIR / CORPS PRINCIPAL (middle section - wider)
    # ==========================================================================
    body_x = [-R_CORE, R_CORE, R_CORE, -R_CORE]
    body_y = [y_skirt_top, y_skirt_top, y_body_top, y_body_top]
    ax.fill(body_x, body_y, color='#2C3E50', alpha=0.9, label='Reservoir')

    # ==========================================================================
    # 3. OGIVE (top section - conical nose)
    # ==========================================================================
    # Left side of cone
    ogive_left_x = [-R_CORE, 0]
    ogive_left_y = [y_body_top, y_nose_top]

    # Right side of cone
    ogive_right_x = [R_CORE, 0]
    ogive_right_y = [y_body_top, y_nose_top]

    # Fill ogive triangle
    ogive_x = [-R_CORE, 0, R_CORE]
    ogive_y = [y_body_top, y_nose_top, y_body_top]
    ax.fill(ogive_x, ogive_y, color='#1A5276', alpha=0.9, label='Ogive')

    # ==========================================================================
    # 4. DRAW OUTLINE
    # ==========================================================================
    # Complete outline (counter-clockwise from bottom-left)
    outline_x = [
        -R_SKIRT,  # Bottom-left of skirt
        -R_SKIRT,  # Top-left of skirt
        -R_CORE,   # Bottom-left of body (step out)
        -R_CORE,   # Top-left of body
        0,         # Nose tip
        R_CORE,    # Top-right of body
        R_CORE,    # Bottom-right of body
        R_SKIRT,   # Top-right of skirt (step in)
        R_SKIRT,   # Bottom-right of skirt
        -R_SKIRT   # Close the path
    ]
    outline_y = [
        y_base,       # Bottom-left of skirt
        y_skirt_top,  # Top-left of skirt
        y_skirt_top,  # Bottom-left of body
        y_body_top,   # Top-left of body
        y_nose_top,   # Nose tip
        y_body_top,   # Top-right of body
        y_skirt_top,  # Bottom-right of body
        y_skirt_top,  # Top-right of skirt
        y_base,       # Bottom-right of skirt
        y_base        # Close
    ]
    ax.plot(outline_x, outline_y, 'k-', linewidth=2.5)

    # ==========================================================================
    # 5. ENGINE NOZZLE (at bottom)
    # ==========================================================================
    nozzle_r_inner = R_SKIRT * 0.5
    nozzle_r_outer = R_SKIRT * 0.7
    nozzle_length = 0.08
    nozzle_x = [-nozzle_r_inner, -nozzle_r_outer, nozzle_r_outer, nozzle_r_inner]
    nozzle_y = [y_base, y_base - nozzle_length, y_base - nozzle_length, y_base]
    ax.fill(nozzle_x, nozzle_y, color='#E74C3C', alpha=0.9, label='Tuyere Vulcain')

    return H_SKIRT, H_BODY, H_NOSE, R_CORE, R_SKIRT


def add_annotations(ax):
    """Add dimension annotations and labels."""

    y_base = 0.0
    y_skirt_top = H_SKIRT
    y_body_top = H_SKIRT + H_BODY
    y_nose_top = H_TOTAL

    # ==========================================================================
    # DIMENSION ARROWS
    # ==========================================================================

    # Total height (left side)
    arrow_x = -R_CORE - 0.12
    ax.annotate('', xy=(arrow_x, y_nose_top), xytext=(arrow_x, y_base),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
    ax.text(arrow_x - 0.03, H_TOTAL/2, f'H_total\n{H_TOTAL:.2f}m',
            ha='right', va='center', fontsize=9, color='blue', rotation=90)

    # Body height
    arrow_x2 = R_CORE + 0.08
    ax.annotate('', xy=(arrow_x2, y_body_top), xytext=(arrow_x2, y_skirt_top),
                arrowprops=dict(arrowstyle='<->', color='green', lw=1.2))
    ax.text(arrow_x2 + 0.02, (y_skirt_top + y_body_top)/2, f'H_body\n{H_BODY:.1f}m',
            ha='left', va='center', fontsize=8, color='green')

    # Nose height
    ax.annotate('', xy=(arrow_x2, y_nose_top), xytext=(arrow_x2, y_body_top),
                arrowprops=dict(arrowstyle='<->', color='green', lw=1.2))
    ax.text(arrow_x2 + 0.02, (y_body_top + y_nose_top)/2, f'H_nose\n{H_NOSE:.1f}m',
            ha='left', va='center', fontsize=8, color='green')

    # Skirt height
    ax.annotate('', xy=(arrow_x2, y_skirt_top), xytext=(arrow_x2, y_base),
                arrowprops=dict(arrowstyle='<->', color='green', lw=1.2))
    ax.text(arrow_x2 + 0.02, H_SKIRT/2, f'H_skirt\n{H_SKIRT:.2f}m',
            ha='left', va='center', fontsize=8, color='green')

    # Core radius (horizontal)
    ax.annotate('', xy=(R_CORE, y_skirt_top + 0.5), xytext=(0, y_skirt_top + 0.5),
                arrowprops=dict(arrowstyle='<->', color='orange', lw=1.2))
    ax.text(R_CORE/2, y_skirt_top + 0.55, f'R_core={R_CORE:.2f}m',
            ha='center', fontsize=8, color='orange')

    # Skirt radius
    ax.annotate('', xy=(R_SKIRT, y_base + 0.05), xytext=(0, y_base + 0.05),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=1.2))
    ax.text(R_SKIRT/2, y_base + 0.02, f'R_skirt={R_SKIRT:.2f}m',
            ha='center', fontsize=7, color='purple')

    # ==========================================================================
    # PART LABELS
    # ==========================================================================
    ax.text(0, y_base + H_SKIRT/2, 'JUPE\nVULCAIN', ha='center', va='center',
            fontsize=8, fontweight='bold', color='white')
    ax.text(0, y_skirt_top + H_BODY/2, 'RESERVOIR\n(corps principal)', ha='center', va='center',
            fontsize=9, fontweight='bold', color='white')
    ax.text(0, y_body_top + H_NOSE/3, 'OGIVE', ha='center', va='center',
            fontsize=9, fontweight='bold', color='white')

    # ==========================================================================
    # REENTRY DIRECTION (tail-first)
    # ==========================================================================
    ax.annotate('', xy=(0, y_base - 0.25), xytext=(0, y_base - 0.12),
                arrowprops=dict(arrowstyle='->', color='red', lw=3))
    ax.text(0, y_base - 0.32, 'Direction rentree\n(TAIL-FIRST)', ha='center',
            fontsize=10, color='red', fontweight='bold')

    # Heating zone indicator
    ax.annotate('Zone chauffee\n(Gamma_F - Robin)',
                xy=(R_SKIRT + 0.02, y_base), xytext=(R_CORE + 0.15, y_base + 0.3),
                fontsize=9, color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))


def main():
    """Generate Ariane 5 profile visualization."""

    # Create output directory
    output_dir = Path(__file__).parent.parent / "data" / "output" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create figure (taller than wide for vertical rocket)
    fig, ax = plt.subplots(1, 1, figsize=(8, 12))

    # Draw rocket
    draw_ariane5_profile(ax)

    # Add annotations
    add_annotations(ax)

    # Configure axes
    margin_x = 0.25
    margin_y = 0.5
    ax.set_xlim(-R_CORE - margin_x, R_CORE + margin_x)
    ax.set_ylim(-margin_y, H_TOTAL + 0.15)
    ax.set_aspect('equal')
    ax.set_xlabel('x [m] (rayon)', fontsize=11)
    ax.set_ylabel('y [m] (hauteur)', fontsize=11)
    ax.set_title('ARIANE 5 - Corps Central (sans boosters)\nRentree Atmospherique TAIL-FIRST',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    # Legend
    ax.legend(loc='upper right', fontsize=9)

    # Add info box
    textstr = (f'ARIANE 5 - Corps Central\n'
               f'(sans boosters lateraux)\n\n'
               f'MODE TAIL-FIRST:\n'
               f'- Base (jupe) en avant\n'
               f'- Ogive protegee (sillage)\n'
               f'- Freinage par reacteur')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    plt.tight_layout()

    # Save figure
    output_file = output_dir / "rocket_profile.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Image sauvegardee: {output_file}")

    # Also save in current directory for easy access
    plt.savefig("rocket_profile.png", dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Image aussi sauvegardee dans: rocket_profile.png")

    plt.close()

    return str(output_file)


if __name__ == '__main__':
    output_path = main()
    print(f"\n[OK] Visualisation ARIANE 5 generee avec succes!")
    print(f"Ouvrez l'image pour verifier la forme de la fusee.")
    print(f"\nParametres utilises:")
    print(f"  - H_nose  = {H_NOSE} m (ogive)")
    print(f"  - H_body  = {H_BODY} m (reservoir)")
    print(f"  - H_skirt = {H_SKIRT} m (jupe Vulcain)")
    print(f"  - R_core  = {R_CORE} m (rayon corps)")
    print(f"  - R_skirt = {R_SKIRT} m (rayon jupe)")
    print(f"  - H_total = {H_TOTAL} m")
