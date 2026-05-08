from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "validation"))

from openfield.apertures import RectangleAperture
from python_helpers.io import write_geometry


CASE_NAME = "rectangle_aperture_geometry"

rect = np.array(
    [
        [
            1,
            -0.5e-3,
            -0.25e-3,
            0.0,
            0.0,
            -0.25e-3,
            0.0,
            0.0,
            0.25e-3,
            0.0,
            -0.5e-3,
            0.25e-3,
            0.0,
            1.0,
            0.5e-3,
            0.5e-3,
            -0.25e-3,
            0.0,
            0.0,
        ],
        [
            2,
            0.1e-3,
            -0.25e-3,
            0.0,
            0.6e-3,
            -0.25e-3,
            0.0,
            0.6e-3,
            0.25e-3,
            0.0,
            0.1e-3,
            0.25e-3,
            0.0,
            1.0,
            0.5e-3,
            0.5e-3,
            0.35e-3,
            0.0,
            0.0,
        ],
    ],
    dtype=np.float64,
)

aperture = RectangleAperture.from_fieldii_rectangles(rect, focus=[0.0, 0.0, 30e-3])
write_geometry(CASE_NAME, "openfield", aperture)
