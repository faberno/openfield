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

    def pulse_echo_response(self, transmit, receive, points):
        from .physics.scattering import pulse_echo_response

        return pulse_echo_response(self, transmit=transmit, receive=receive, points=points)

    def scatterer_response(self, transmit, receive, points, amplitudes):
        from .physics.scattering import scatterer_response

        return scatterer_response(
            self,
            transmit=transmit,
            receive=receive,
            points=points,
            amplitudes=amplitudes,
        )

    def receive_channel_responses(self, transmit, receive, points, amplitudes):
        from .physics.scattering import receive_channel_responses

        return receive_channel_responses(
            self,
            transmit=transmit,
            receive=receive,
            points=points,
            amplitudes=amplitudes,
        )

    def full_matrix_capture(self, transmit, receive, points, amplitudes, *, decimation_factor: int = 1):
        from .physics.scattering import full_matrix_capture

        return full_matrix_capture(
            self,
            transmit=transmit,
            receive=receive,
            points=points,
            amplitudes=amplitudes,
            decimation_factor=decimation_factor,
        )
