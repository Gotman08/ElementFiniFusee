"""
@file __init__.py
@brief Visualization module initialization
@author HPC-Code-Documenter
@date 2025

@details
The visualization module provides tools for displaying and exporting
FEM thermal analysis results.

Static plotting (plotting.py):
- plot_temperature_field: 2D contour plot of temperature distribution
- plot_parametric_curve: $T_{max}(V)$ engineering curves
- plot_multiple_temperature_fields: Side-by-side comparison figures
- export_results_to_csv: Data export for external analysis

Animation (animation.py):
- compute_all_frames: Pre-compute solutions for smooth playback
- create_animation: Interactive/exportable reentry animation

Output formats:
- PNG images at 300 DPI for publication quality
- GIF animations via Pillow
- MP4 videos via FFmpeg
- CSV data files for spreadsheet import

@example
>>> from src.visualization import (
...     plot_temperature_field,
...     plot_parametric_curve,
...     export_results_to_csv
... )
>>> plot_parametric_curve(velocities, T_max_list, save_file="results.png")
"""

from src.visualization.plotting import (
    plot_temperature_field,
    plot_parametric_curve,
    plot_multiple_temperature_fields,
    export_results_to_csv,
)

__all__ = [
    "plot_temperature_field",
    "plot_parametric_curve",
    "plot_multiple_temperature_fields",
    "export_results_to_csv",
]
