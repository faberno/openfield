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
    """Calculate a CPU reference spatial impulse response.

    The default Field II rectangle path uses a far-field approximation for
    each subelement: one weighted contribution at the subelement center,
    deposited on the sampled time axis.
    """

    if simulation.backend != "numpy":
        raise NotImplementedError("only the numpy backend is available in the scaffold")

    points = _points_array(points)
    sound_speed = simulation.medium.sound_speed
    fs = simulation.sampling_frequency

    delays = np.zeros(aperture.physical_element_count, dtype=np.float64)
    if aperture.focus is not None:
        delays = aperture.focus.delays(aperture, sound_speed, time=0.0)
    apodization = np.ones(aperture.physical_element_count, dtype=np.float64)
    if aperture.apodization is not None:
        apodization = np.asarray(aperture.apodization.at_time(0.0), dtype=np.float64)
        if apodization.shape != (aperture.physical_element_count,):
            raise ValueError("apodization length must match physical element count")
    subelement_apodization = np.ones(len(aperture.elements), dtype=np.float64)
    if aperture.subelement_apodization is not None:
        subelement_apodization = aperture.subelement_apodization.values_for(aperture)
    subelement_delays = np.zeros(len(aperture.elements), dtype=np.float64)
    if aperture.subelement_delays is not None:
        subelement_delays = aperture.subelement_delays.values_for(aperture)

    start_time, sample_count = _fieldii_like_time_axis(
        aperture=aperture,
        points=points,
        delays=delays,
        subelement_delays=subelement_delays,
        sound_speed=sound_speed,
        sampling_frequency=fs,
    )
    samples = np.zeros((sample_count, points.shape[0]), dtype=np.float64)

    for point_index, point in enumerate(points):
        for element_index, element in enumerate(aperture.elements):
            delay = delays[element.physical_index] + subelement_delays[element_index]
            weight_scale = (
                float(apodization[element.physical_index])
                * float(subelement_apodization[element_index])
                * _baffle_weight(aperture.baffle, point=point, element=element)
            )
            _add_center_delta_response(
                samples[:, point_index],
                point=point,
                element=element,
                start_time=start_time,
                delay=delay,
                weight_scale=weight_scale,
                sound_speed=sound_speed,
                sampling_frequency=fs,
            )

    return TimeResponse(
        samples=samples,
        sampling_frequency=fs,
        start_time=start_time,
    )


def _rectangle_spatial_impulse(*, point, element, time: np.ndarray, sound_speed: float) -> np.ndarray:
    vertices = element.vertices
    tangent_x = vertices[1] - vertices[0]
    width = float(np.linalg.norm(tangent_x))
    if width == 0:
        raise ValueError("rectangle width must be positive")
    tangent_x = tangent_x / width

    tangent_y = vertices[3] - vertices[0]
    height = float(np.linalg.norm(tangent_y))
    if height == 0:
        raise ValueError("rectangle height must be positive")
    tangent_y = tangent_y / height

    normal = np.cross(tangent_x, tangent_y)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm == 0:
        raise ValueError("rectangle normal must be non-zero")
    normal = normal / normal_norm
    if np.dot(normal, element.normal) < 0:
        normal = -normal

    relative = point - element.center
    projected_x = float(np.dot(relative, tangent_x))
    projected_y = float(np.dot(relative, tangent_y))
    distance_to_plane = abs(float(np.dot(relative, normal)))

    # Field II samples the integrated response, which behaves like a
    # sample-bin average of the analytic angular response.
    dt = float(time[1] - time[0]) if len(time) > 1 else 0.0
    offsets = 0.5 * dt * np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)], dtype=np.float64)
    weights = 0.5 * np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=np.float64)
    response = np.zeros_like(time, dtype=np.float64)

    for offset, weight in zip(offsets, weights):
        shifted_time = time + offset
        valid = shifted_time >= distance_to_plane / sound_speed
        if not np.any(valid):
            continue

        radii_sq = (sound_speed * shifted_time[valid]) ** 2 - distance_to_plane * distance_to_plane
        radii = np.sqrt(np.maximum(radii_sq, 0.0))
        angles = np.array(
            [
                _angular_measure_in_rectangle(
                    radius=radius,
                    center_x=projected_x,
                    center_y=projected_y,
                    half_width=width / 2,
                    half_height=height / 2,
                )
                for radius in radii
            ],
            dtype=np.float64,
        )
        response[valid] += weight * sound_speed * angles / (2.0 * np.pi)
    return response


def _angular_measure_in_rectangle(
    *,
    radius: float,
    center_x: float,
    center_y: float,
    half_width: float,
    half_height: float,
) -> float:
    if radius <= 1e-15:
        inside = abs(center_x) <= half_width and abs(center_y) <= half_height
        return 2.0 * np.pi if inside else 0.0

    angles = [0.0, 2.0 * np.pi]
    for x_edge in (-half_width, half_width):
        value = (x_edge - center_x) / radius
        if -1.0 <= value <= 1.0:
            angle = float(np.arccos(value))
            angles.extend([angle, (2.0 * np.pi - angle) % (2.0 * np.pi)])
    for y_edge in (-half_height, half_height):
        value = (y_edge - center_y) / radius
        if -1.0 <= value <= 1.0:
            angle = float(np.arcsin(value))
            angles.extend([angle % (2.0 * np.pi), (np.pi - angle) % (2.0 * np.pi)])

    angles = np.array(sorted(set(_round_angle(angle) for angle in angles)), dtype=np.float64)
    measure = 0.0
    for left, right in zip(angles[:-1], angles[1:]):
        if right <= left:
            continue
        midpoint = 0.5 * (left + right)
        x = center_x + radius * np.cos(midpoint)
        y = center_y + radius * np.sin(midpoint)
        if -half_width <= x <= half_width and -half_height <= y <= half_height:
            measure += right - left
    return float(measure)


def _round_angle(angle: float) -> float:
    angle = angle % (2.0 * np.pi)
    if np.isclose(angle, 2.0 * np.pi):
        angle = 0.0
    return round(float(angle), 15)


def _add_center_delta_response(
    output: np.ndarray,
    *,
    point,
    element,
    start_time: float,
    delay: float,
    weight_scale: float,
    sound_speed: float,
    sampling_frequency: float,
) -> None:
    distance = float(np.linalg.norm(point - element.center))
    if distance == 0:
        raise ValueError("field points must not coincide with aperture elements")
    weight = weight_scale * element.area / (2.0 * np.pi * distance) * sampling_frequency

    arrival_time = distance / sound_speed + delay
    if element.vertices is None:
        index = int(np.floor((arrival_time - start_time) * sampling_frequency))
        if 0 <= index < output.shape[0]:
            output[index] += weight
        return

    support_times = np.linalg.norm(point - element.vertices, axis=1) / sound_speed + delay
    support_start = float(np.min(support_times))
    support_end = float(np.max(support_times))
    duration = support_end - support_start
    if duration <= 1e-15:
        index = int(np.floor((arrival_time - start_time) * sampling_frequency))
        if 0 <= index < output.shape[0]:
            output[index] += weight
        return

    first = int(np.floor((support_start - start_time) * sampling_frequency))
    last = int(np.floor((support_end - start_time) * sampling_frequency))
    dt = 1.0 / sampling_frequency
    for index in range(first, last + 1):
        if not 0 <= index < output.shape[0]:
            continue
        bin_start = start_time + index * dt
        bin_end = bin_start + dt
        overlap = min(support_end, bin_end) - max(support_start, bin_start)
        if overlap > 0:
            output[index] += weight * overlap / duration


def _fieldii_like_time_axis(
    *,
    aperture,
    points: np.ndarray,
    delays: np.ndarray,
    subelement_delays: np.ndarray,
    sound_speed: float,
    sampling_frequency: float,
) -> tuple[float, int]:
    support_times = []
    for point in points:
        for element_index, element in enumerate(aperture.elements):
            support_points = element.vertices
            if support_points is None:
                support_points = element.center[None, :]
            support_times.extend(
                (
                    np.linalg.norm(point - support_points, axis=1) / sound_speed
                    + delays[element.physical_index]
                    + subelement_delays[element_index]
                ).tolist()
            )

    support_times = np.asarray(support_times, dtype=np.float64)
    support_start = np.floor(np.min(support_times) * sampling_frequency) / sampling_frequency
    support_end = np.ceil(np.max(support_times) * sampling_frequency) / sampling_frequency
    start_time = support_start - FIELDII_LEADING_PAD_SAMPLES / sampling_frequency
    end_time = support_end + FIELDII_TRAILING_PAD_SAMPLES / sampling_frequency
    sample_count = int(round((end_time - start_time) * sampling_frequency)) + 1
    return float(start_time), sample_count


def _baffle_weight(baffle, *, point, element) -> float:
    if baffle is None or baffle.kind == "rigid":
        return 1.0
    direction = point - element.center
    distance = float(np.linalg.norm(direction))
    if distance == 0:
        return 0.0
    cosine = float(np.dot(element.normal, direction / distance))
    return max(cosine, 0.0)
