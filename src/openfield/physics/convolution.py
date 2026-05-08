from __future__ import annotations

import numpy as np

from openfield.responses import TimeResponse
from openfield.waveforms import Waveform, sampled_waveform


def convolve_time_response(
    response: TimeResponse,
    waveform,
    *,
    scale_by_dt: bool = True,
) -> TimeResponse:
    """Convolve a sampled time response with a waveform along the time axis."""

    waveform = sampled_waveform(waveform, response.sampling_frequency)
    samples = _convolve_along_time(response.samples, waveform.samples)
    if scale_by_dt:
        samples = samples / response.sampling_frequency
    return TimeResponse(
        samples=samples,
        sampling_frequency=response.sampling_frequency,
        start_time=response.start_time + waveform.start_time,
    )


def convolve_waveforms(first, second, sampling_frequency: float, *, scale_by_dt: bool = True) -> Waveform:
    """Sample two waveform-like objects and return their convolution."""

    first = sampled_waveform(first, sampling_frequency)
    second = sampled_waveform(second, sampling_frequency)
    samples = np.convolve(first.samples, second.samples, mode="full")
    if scale_by_dt:
        samples = samples / sampling_frequency
    return Waveform(
        samples=samples,
        sampling_frequency=sampling_frequency,
        start_time=first.start_time + second.start_time,
    )


def _convolve_along_time(samples: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim != 1:
        raise ValueError("kernel must be one-dimensional")
    if samples.ndim == 1:
        return np.convolve(samples, kernel, mode="full")

    output_shape = (samples.shape[0] + kernel.size - 1,) + samples.shape[1:]
    output = np.empty(output_shape, dtype=np.float64)
    for index in np.ndindex(samples.shape[1:]):
        output[(slice(None),) + index] = np.convolve(samples[(slice(None),) + index], kernel, mode="full")
    return output
