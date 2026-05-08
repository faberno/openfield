from __future__ import annotations

import numpy as np

from openfield.physics.convolution import convolve_time_response
from openfield.responses import TimeResponse


def pulse_echo_response(simulation, *, transmit, receive, points) -> TimeResponse:
    """Calculate the pulse-echo voltage response at field points."""

    _require_pulse_echo_waveforms(transmit, receive)
    if simulation.backend != "numpy":
        raise NotImplementedError("only the numpy backend is available for pulse_echo_response")

    points = _points_array(points)
    transmit_response = simulation.spatial_impulse_response(transmit, points)
    receive_response = simulation.spatial_impulse_response(receive, points)

    response = _convolve_pointwise(transmit_response, receive_response)
    response = convolve_time_response(response, transmit.excitation)
    response = convolve_time_response(response, transmit.impulse_response)
    response = convolve_time_response(response, receive.impulse_response)
    # Field II drops the final pulse-echo convolution sample in calc_hhp and
    # calc_scat_multi outputs.
    return _trim_trailing_samples(response, 1)


def scatterer_response(simulation, *, transmit, receive, points, amplitudes) -> TimeResponse:
    """Calculate the summed RF response from point scatterers."""

    points = _points_array(points)
    amplitudes = _amplitudes_array(amplitudes, points.shape[0])
    response = _weighted_scatterer_response(
        simulation,
        transmit=transmit,
        receive=receive,
        points=points,
        amplitudes=amplitudes,
    )
    # calc_scat and calc_scat_all drop one more trailing sample than calc_hhp.
    return _trim_trailing_samples(response, 1)


def _weighted_scatterer_response(simulation, *, transmit, receive, points, amplitudes) -> TimeResponse:
    points = _points_array(points)
    amplitudes = _amplitudes_array(amplitudes, points.shape[0])
    response = pulse_echo_response(simulation, transmit=transmit, receive=receive, points=points)
    samples = response.samples @ amplitudes
    return TimeResponse(
        samples=samples[:, None],
        sampling_frequency=response.sampling_frequency,
        start_time=response.start_time,
    )


def receive_channel_responses(simulation, *, transmit, receive, points, amplitudes) -> TimeResponse:
    """Calculate one scatterer RF trace per receive physical element."""

    responses = [
        _weighted_scatterer_response(
            simulation,
            transmit=transmit,
            receive=receive.select_physical_elements([receive_index]),
            points=points,
            amplitudes=amplitudes,
        )
        for receive_index in range(receive.physical_element_count)
    ]
    return _stack_time_responses(responses)


def full_matrix_capture(
    simulation,
    *,
    transmit,
    receive,
    points,
    amplitudes,
    decimation_factor: int = 1,
) -> TimeResponse:
    """Calculate channel data for every transmit/receive element pair."""

    decimation_factor = int(round(decimation_factor))
    if decimation_factor < 1:
        raise ValueError("decimation_factor must be at least one")

    responses = []
    for transmit_index in range(transmit.physical_element_count):
        transmit_element = transmit.select_physical_elements([transmit_index])
        for receive_index in range(receive.physical_element_count):
            responses.append(
                scatterer_response(
                    simulation,
                    transmit=transmit_element,
                    receive=receive.select_physical_elements([receive_index]),
                    points=points,
                    amplitudes=amplitudes,
                )
            )
    response = _stack_time_responses(responses)
    # Field II's calc_scat_all starts two samples later than the equivalent
    # per-channel scatterer traces.
    response = _trim_leading_samples(response, 2)
    if decimation_factor == 1:
        return response
    return TimeResponse(
        samples=response.samples[::decimation_factor],
        sampling_frequency=response.sampling_frequency / decimation_factor,
        start_time=response.start_time,
    )


def _require_pulse_echo_waveforms(transmit, receive) -> None:
    if transmit.excitation is None:
        raise ValueError("transmit aperture must have an excitation waveform")
    if transmit.impulse_response is None:
        raise ValueError("transmit aperture must have an impulse response waveform")
    if receive.impulse_response is None:
        raise ValueError("receive aperture must have an impulse response waveform")


def _points_array(points) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        points = points[None, :]
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    return points


def _amplitudes_array(amplitudes, count: int) -> np.ndarray:
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    if amplitudes.ndim == 0:
        amplitudes = amplitudes[None]
    amplitudes = amplitudes.reshape(-1)
    if amplitudes.shape != (count,):
        raise ValueError("amplitudes must contain one value per point")
    return amplitudes


def _convolve_pointwise(first: TimeResponse, second: TimeResponse) -> TimeResponse:
    if first.sampling_frequency != second.sampling_frequency:
        raise ValueError("responses must use the same sampling_frequency")
    if first.samples.ndim != 2 or second.samples.ndim != 2:
        raise ValueError("pointwise responses must be two-dimensional")
    if first.samples.shape[1] != second.samples.shape[1]:
        raise ValueError("responses must contain the same number of points")

    samples = np.empty(
        (first.sample_count + second.sample_count - 1, first.samples.shape[1]),
        dtype=np.float64,
    )
    for point_index in range(first.samples.shape[1]):
        samples[:, point_index] = np.convolve(
            first.samples[:, point_index],
            second.samples[:, point_index],
            mode="full",
        )
    samples = samples / first.sampling_frequency
    return TimeResponse(
        samples=samples,
        sampling_frequency=first.sampling_frequency,
        start_time=first.start_time + second.start_time,
    )


def _stack_time_responses(responses: list[TimeResponse]) -> TimeResponse:
    if not responses:
        raise ValueError("at least one response is required")
    sampling_frequency = responses[0].sampling_frequency
    for response in responses:
        if response.sampling_frequency != sampling_frequency:
            raise ValueError("responses must use the same sampling_frequency")
        if response.samples.ndim != 2 or response.samples.shape[1] != 1:
            raise ValueError("responses must be single-column time responses")

    start_time = min(response.start_time for response in responses)
    end_time = max(
        response.start_time + (response.sample_count - 1) / sampling_frequency
        for response in responses
    )
    sample_count = int(round((end_time - start_time) * sampling_frequency)) + 1
    samples = np.zeros((sample_count, len(responses)), dtype=np.float64)
    for column, response in enumerate(responses):
        offset = int(round((response.start_time - start_time) * sampling_frequency))
        samples[offset : offset + response.sample_count, column] = response.samples[:, 0]

    return TimeResponse(
        samples=samples,
        sampling_frequency=sampling_frequency,
        start_time=start_time,
    )


def _trim_trailing_samples(response: TimeResponse, count: int) -> TimeResponse:
    if count <= 0:
        return response
    if response.sample_count <= count:
        raise ValueError("cannot trim all response samples")
    return TimeResponse(
        samples=response.samples[:-count],
        sampling_frequency=response.sampling_frequency,
        start_time=response.start_time,
    )


def _trim_leading_samples(response: TimeResponse, count: int) -> TimeResponse:
    if count <= 0:
        return response
    if response.sample_count <= count:
        raise ValueError("cannot trim all response samples")
    return TimeResponse(
        samples=response.samples[count:],
        sampling_frequency=response.sampling_frequency,
        start_time=response.start_time + count / response.sampling_frequency,
    )
