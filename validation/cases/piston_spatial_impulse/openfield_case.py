from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "validation"))

from openfield import Medium, Simulation
from openfield.apertures import Piston
from python_helpers.io import write_time_response


CASE_NAME = "piston_spatial_impulse"

sim = Simulation(sampling_frequency=100e6, medium=Medium(sound_speed=1540.0))
aperture = Piston(radius=5e-3, element_size=0.5e-3, tessellation="cartesian")
points = [
    [0.0, 0.0, 30e-3],
    [0.0, 0.0, 40e-3],
]

response = sim.spatial_impulse_response(aperture, points)
write_time_response(CASE_NAME, "openfield", response)
