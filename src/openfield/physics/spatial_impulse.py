from __future__ import annotations

import numpy as np

from openfield.responses import TimeResponse


def _points_array(points) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        points = points[None, :]
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    return points


def spatial_impulse_response(simulation, *, aperture, points) -> TimeResponse:
    """Approximate spatial impulse response using far-field element weights.

    This is a correctness-oriented CPU reference path. It establishes response
    shapes and time-axis semantics before optimized kernels are added.
    """

    if simulation.backend != "numpy":
        raise NotImplementedError("only the numpy backend is available in the scaffold")

    points = _points_array(points)
    centers = aperture.centers
    areas = aperture.areas
    physical_indices = aperture.physical_indices
    sound_speed = simulation.medium.sound_speed
    fs = simulation.sampling_frequency

    delays = np.zeros(aperture.physical_element_count, dtype=np.float64)
    if aperture.focus is not None:
        delays = aperture.focus.delays(aperture, sound_speed)

    diff = points[:, None, :] - centers[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    if np.any(distances == 0):
        raise ValueError("field points must not coincide with aperture elements")

    arrival_times = distances / sound_speed + delays[physical_indices][None, :]
    start_time = float(np.floor(np.min(arrival_times) * fs) / fs)
    end_time = float(np.ceil(np.max(arrival_times) * fs) / fs)
    sample_count = int(round((end_time - start_time) * fs)) + 2
    samples = np.zeros((sample_count, points.shape[0]), dtype=np.float64)

    weights = areas[None, :] / (2.0 * np.pi * distances)
    positions = (arrival_times - start_time) * fs
    lower = np.floor(positions).astype(np.int64)
    fraction = positions - lower

    for point_index in range(points.shape[0]):
        np.add.at(samples[:, point_index], lower[point_index], weights[point_index] * (1.0 - fraction[point_index]))
        np.add.at(samples[:, point_index], lower[point_index] + 1, weights[point_index] * fraction[point_index])

    return TimeResponse(
        samples=samples,
        sampling_frequency=fs,
        start_time=start_time,
    )
