from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "validation"))

from openfield.apertures import TwoDimensionalArray
from python_helpers.io import write_geometry


CASE_NAME = "two_dimensional_array_geometry"

enabled = np.array(
    [
        [1, 0, 1],
        [0, 1, 0],
        [1, 1, 0],
        [0, 1, 1],
    ],
    dtype=bool,
)

aperture = TwoDimensionalArray(
    elements_x=4,
    elements_y=3,
    width=0.3e-3,
    height=0.4e-3,
    kerf_x=0.03e-3,
    kerf_y=0.04e-3,
    enabled=enabled,
    subdivisions=(2, 2),
).focused_at([0.0, 0.0, 40e-3])

write_geometry(CASE_NAME, "openfield", aperture)
