from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .medium import Medium


BackendName = Literal["numpy", "cuda"]


@dataclass(frozen=True)
class Simulation:
    """Container for simulation-wide numerical settings."""

    sampling_frequency: float
    medium: Medium = Medium()
    backend: BackendName = "numpy"

    def __post_init__(self) -> None:
        if self.sampling_frequency <= 0:
            raise ValueError("sampling_frequency must be positive")
        if self.backend not in {"numpy", "cuda"}:
            raise ValueError("backend must be 'numpy' or 'cuda'")

    def spatial_impulse_response(self, aperture, points):
        from .physics.spatial_impulse import spatial_impulse_response

        return spatial_impulse_response(self, aperture=aperture, points=points)

    def emitted_pressure(self, transmit, points):
        from .physics.pressure import emitted_pressure

        return emitted_pressure(self, transmit=transmit, points=points)
