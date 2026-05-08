from __future__ import annotations

from openfield.physics.convolution import convolve_time_response
from openfield.waveforms import sampled_waveform


def emitted_pressure(simulation, *, transmit, points):
    """Calculate emitted pressure from a transmit aperture at field points."""

    if transmit.excitation is None:
        raise ValueError("transmit aperture must have an excitation waveform")
    if transmit.impulse_response is None:
        raise ValueError("transmit aperture must have an impulse response waveform")

    if simulation.backend != "numpy":
        raise NotImplementedError("only the numpy backend is available for emitted_pressure")

    excitation = sampled_waveform(transmit.excitation, simulation.sampling_frequency)
    impulse_response = sampled_waveform(transmit.impulse_response, simulation.sampling_frequency)
    spatial_response = simulation.spatial_impulse_response(transmit, points)

    pressure = convolve_time_response(spatial_response, excitation)
    return convolve_time_response(pressure, impulse_response)
