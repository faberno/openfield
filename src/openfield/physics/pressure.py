from __future__ import annotations

import numpy as np

from openfield.physics.convolution import convolve_time_response
from openfield.responses import TimeResponse
from openfield.waveforms import sampled_waveform


def emitted_pressure(simulation, *, transmit, points):
    """Calculate emitted pressure from a transmit aperture at field points."""

    if transmit.excitation is None and transmit.element_waveforms is None:
        raise ValueError("transmit aperture must have an excitation waveform")
    if transmit.impulse_response is None:
        raise ValueError("transmit aperture must have an impulse response waveform")

    if simulation.backend != "numpy":
        raise NotImplementedError("only the numpy backend is available for emitted_pressure")

    impulse_response = sampled_waveform(transmit.impulse_response, simulation.sampling_frequency)
    if transmit.element_waveforms is not None:
        pressure = _element_waveform_pressure(simulation, transmit=transmit, points=points)
    else:
        excitation = sampled_waveform(transmit.excitation, simulation.sampling_frequency)
        spatial_response = simulation.spatial_impulse_response(transmit, points)
        pressure = convolve_time_response(spatial_response, excitation)

    return convolve_time_response(pressure, impulse_response)


def _element_waveform_pressure(simulation, *, transmit, points) -> TimeResponse:
    responses = []
    for physical_index in range(transmit.physical_element_count):
        waveform = transmit.element_waveforms.waveform_for(physical_index, default=transmit.excitation)
        if waveform is None:
            continue
        waveform = sampled_waveform(waveform, simulation.sampling_frequency)
        if not np.any(waveform.samples):
            continue
        element_aperture = transmit.select_physical_elements([physical_index])._replace(
            excitation=waveform,
            element_waveforms=None,
        )
        spatial_response = simulation.spatial_impulse_response(element_aperture, points)
        responses.append(convolve_time_response(spatial_response, waveform))
    if not responses:
        raise ValueError("at least one element waveform or default excitation is required")
    return _sum_time_responses(responses)


def _sum_time_responses(responses: list[TimeResponse]) -> TimeResponse:
    sampling_frequency = responses[0].sampling_frequency
    for response in responses:
        if response.sampling_frequency != sampling_frequency:
            raise ValueError("responses must use the same sampling_frequency")
    start_time = min(response.start_time for response in responses)
    end_time = max(
        response.start_time + (response.sample_count - 1) / sampling_frequency
        for response in responses
    )
    sample_count = int(round((end_time - start_time) * sampling_frequency)) + 1
    output_shape = (sample_count,) + responses[0].samples.shape[1:]
    samples = np.zeros(output_shape, dtype=np.float64)
    for response in responses:
        offset = int(round((response.start_time - start_time) * sampling_frequency))
        samples[offset : offset + response.sample_count] += response.samples
    return TimeResponse(
        samples=samples,
        sampling_frequency=sampling_frequency,
        start_time=start_time,
    )
