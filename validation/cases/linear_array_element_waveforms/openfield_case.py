from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "validation"))

from openfield import Medium, Simulation
from openfield.apertures import LinearArray
from openfield.waveforms import Waveform
from python_helpers.io import write_time_response


CASE_NAME = "linear_array_element_waveforms"
FS = 100e6

sim = Simulation(sampling_frequency=FS, medium=Medium(sound_speed=1540.0))
aperture = LinearArray(
    elements=4,
    width=0.3e-3,
    height=5e-3,
    kerf=0.03e-3,
    subdivisions=(1, 2),
).focused_at([0.0, 0.0, 40e-3])
aperture = aperture.with_impulse_response(Waveform([1.0], sampling_frequency=FS)).with_element_waveforms(
    [
        Waveform([1.0, 0.0, 0.0], sampling_frequency=FS),
        Waveform([0.5, 0.25, 0.0], sampling_frequency=FS),
        Waveform([0.0, 1.0, 0.0], sampling_frequency=FS),
        Waveform([0.0, 0.5, 1.0], sampling_frequency=FS),
    ]
)
points = [
    [0.0, 0.0, 40e-3],
    [2e-3, 0.0, 40e-3],
]

response = sim.emitted_pressure(aperture, points)
write_time_response(CASE_NAME, "openfield", response)
