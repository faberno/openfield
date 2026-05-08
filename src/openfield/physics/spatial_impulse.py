from __future__ import annotations

import numpy as np

from openfield.responses import TimeResponse

FIELDII_LEADING_PAD_SAMPLES = 1
FIELDII_TRAILING_PAD_SAMPLES = 5


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
    start_time, sample_count = _fieldii_like_time_axis(
        aperture=aperture,
        points=points,
        delays=delays,
        sound_speed=sound_speed,
        sampling_frequency=fs,
    )
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


def _fieldii_like_time_axis(
    *,
    aperture,
    points: np.ndarray,
    delays: np.ndarray,
    sound_speed: float,
    sampling_frequency: float,
) -> tuple[float, int]:
    support_times = []
    for point in points:
        for element in aperture.elements:
            support_points = element.vertices
            if support_points is None:
                support_points = element.center[None, :]
            support_times.extend(
                (
                    np.linalg.norm(point - support_points, axis=1) / sound_speed
                    + delays[element.physical_index]
                ).tolist()
            )

    support_times = np.asarray(support_times, dtype=np.float64)
    support_start = np.floor(np.min(support_times) * sampling_frequency) / sampling_frequency
    support_end = np.ceil(np.max(support_times) * sampling_frequency) / sampling_frequency
    start_time = support_start - FIELDII_LEADING_PAD_SAMPLES / sampling_frequency
    end_time = support_end + FIELDII_TRAILING_PAD_SAMPLES / sampling_frequency
    sample_count = int(round((end_time - start_time) * sampling_frequency)) + 1
    return float(start_time), sample_count
