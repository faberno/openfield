from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "validation"))

from openfield.apertures import LinearArray
from python_helpers.io import write_geometry


CASE_NAME = "linear_array_geometry"

aperture = LinearArray(
    elements=8,
    width=0.3e-3,
    height=5e-3,
    kerf=0.03e-3,
    subdivisions=(2, 3),
).focused_at([0.0, 0.0, 40e-3])

write_geometry(CASE_NAME, "openfield", aperture)
