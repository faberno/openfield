from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "validation"))

from python_helpers.aperture_calc_h_cases import run_case


run_case("convex_focused_array_spatial_impulse")
