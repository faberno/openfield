from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "validation"))

from openfield import Medium, Simulation
from openfield.apertures import LinearArray
from python_helpers.io import write_time_response


CASE_NAME = "linear_array_soft_baffle_spatial_impulse"

sim = Simulation(sampling_frequency=100e6, medium=Medium(sound_speed=1540.0))
aperture = (
    LinearArray(
        elements=4,
        width=0.3e-3,
        height=5e-3,
        kerf=0.03e-3,
        subdivisions=(1, 5),
    )
    .focused_at([0.0, 0.0, 40e-3])
    .with_baffle("soft")
)
points = [
    [4e-3, 0.0, 30e-3],
    [6e-3, 0.0, 40e-3],
]

response = sim.spatial_impulse_response(aperture, points)
write_time_response(CASE_NAME, "openfield", response)
