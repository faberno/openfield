from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "validation"))

from openfield import Medium, Simulation
from openfield.apertures import LinearArray
from openfield.waveforms import ToneBurst
from python_helpers.io import write_time_response


CASE_NAME = "linear_array_scatterer_response"

f0 = 3e6
sim = Simulation(sampling_frequency=100e6, medium=Medium(sound_speed=1540.0))
base = LinearArray(
    elements=4,
    width=0.3e-3,
    height=5e-3,
    kerf=0.03e-3,
    subdivisions=(1, 5),
).focused_at([0.0, 0.0, 40e-3])
transmit = base.with_impulse_response(
    ToneBurst(center_frequency=f0, cycles=2, window="hann")
).with_excitation(
    ToneBurst(center_frequency=f0, cycles=2)
)
receive = base.with_impulse_response(ToneBurst(center_frequency=f0, cycles=2, window="hann"))
points = [
    [0.0, 0.0, 30e-3],
    [0.0, 0.0, 40e-3],
    [2e-3, 0.0, 40e-3],
]
amplitudes = [1.0, -0.5, 0.25]

response = sim.scatterer_response(transmit, receive, points, amplitudes)
write_time_response(CASE_NAME, "openfield", response)
