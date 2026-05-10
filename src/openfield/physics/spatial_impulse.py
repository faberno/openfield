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

    Rectangular subelements use Field II's far-field rectangle convention:
    the rectangle is projected onto the propagation direction and integrated
    over sample-centered time bins.
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
        leading_pad_samples=_leading_pad_samples(aperture),
        trailing_pad_samples=_trailing_pad_samples(aperture, subelement_delays),
    )
    samples = np.zeros((sample_count, points.shape[0]), dtype=np.float64)

    for point_index, point in enumerate(points):
        for element_index, element in enumerate(aperture.elements):
            physical_delay = delays[element.physical_index]
            delay = physical_delay + subelement_delays[element_index]
            weight_scale = (
                float(apodization[element.physical_index])
                * float(subelement_apodization[element_index])
            )
            baffle_distance = _soft_baffle_normal_distance(aperture.baffle, point=point, element=element)
            _add_center_delta_response(
                samples[:, point_index],
                point=point,
                element=element,
                exact_flat_polygon=aperture.name in {"triangle_aperture", "line_bounded_aperture"},
                start_time=start_time,
                delay=delay,
                propagation_delay=physical_delay,
                weight_scale=weight_scale,
                baffle_distance=baffle_distance,
                sound_speed=sound_speed,
                sampling_frequency=fs,
            )

    return TimeResponse(
        samples=samples,
        sampling_frequency=fs,
        start_time=start_time,
    )


def _add_center_delta_response(
    output: np.ndarray,
    *,
    point,
    element,
    exact_flat_polygon: bool,
    start_time: float,
    delay: float,
    propagation_delay: float,
    weight_scale: float,
    baffle_distance: float | None,
    sound_speed: float,
    sampling_frequency: float,
) -> None:
    distance = float(np.linalg.norm(point - element.center))
    if distance == 0:
        raise ValueError("field points must not coincide with aperture elements")

    arrival_time = distance / sound_speed + delay
    if exact_flat_polygon and element.vertices is not None:
        _add_exact_flat_polygon_response(
            output,
            point=point,
            element=element,
            start_time=start_time,
            delay=delay,
            weight_scale=weight_scale,
            baffle_distance=baffle_distance,
            sound_speed=sound_speed,
            sampling_frequency=sampling_frequency,
        )
        return

    if element.vertices is not None and element.vertices.shape == (4, 3):
        _add_projected_rectangle_response(
            output,
            point=point,
            element=element,
            start_time=start_time,
            delay=delay,
            propagation_delay=propagation_delay,
            weight_scale=weight_scale,
            baffle_distance=baffle_distance,
            sound_speed=sound_speed,
            sampling_frequency=sampling_frequency,
        )
        return

    if element.vertices is None:
        index = int(np.floor((arrival_time - start_time) * sampling_frequency + 0.5))
        if 0 <= index < output.shape[0]:
            sample_time = start_time + index / sampling_frequency
            output[index] += _sample_weight(
                element=element,
                sample_time=sample_time,
                propagation_delay=propagation_delay,
                weight_scale=weight_scale,
                baffle_distance=baffle_distance,
                sound_speed=sound_speed,
                sampling_frequency=sampling_frequency,
            )
        return

    _add_projected_polygon_response(
        output,
        point=point,
        element=element,
        start_time=start_time,
        delay=delay,
        propagation_delay=propagation_delay,
        weight_scale=weight_scale,
        baffle_distance=baffle_distance,
        sound_speed=sound_speed,
        sampling_frequency=sampling_frequency,
    )


def _add_projected_rectangle_response(
    output: np.ndarray,
    *,
    point,
    element,
    start_time: float,
    delay: float,
    propagation_delay: float,
    weight_scale: float,
    baffle_distance: float | None,
    sound_speed: float,
    sampling_frequency: float,
) -> None:
    distance_vector = point - element.center
    distance = float(np.linalg.norm(distance_vector))
    if distance == 0:
        raise ValueError("field points must not coincide with aperture elements")

    direction = distance_vector / distance
    half_x, half_y = _fieldii_rectangle_projection_half_widths(
        element,
        direction=direction,
        sound_speed=sound_speed,
    )
    arrival_time = distance / sound_speed + delay
    dt = 1.0 / sampling_frequency

    if half_x <= 1e-18 and half_y <= 1e-18:
        index = int(np.floor((arrival_time - start_time) * sampling_frequency + 0.5))
        if 0 <= index < output.shape[0]:
            sample_time = start_time + index * dt
            output[index] += _sample_weight(
                element=element,
                sample_time=sample_time,
                propagation_delay=propagation_delay,
                weight_scale=weight_scale,
                baffle_distance=baffle_distance,
                sound_speed=sound_speed,
                sampling_frequency=sampling_frequency,
            )
        return

    support_radius = half_x + half_y
    support_start = arrival_time - support_radius
    support_end = arrival_time + support_radius
    first = int(np.floor((support_start - start_time) * sampling_frequency - 0.5)) - 1
    last = int(np.ceil((support_end - start_time) * sampling_frequency + 0.5)) + 1

    for index in range(first, last + 1):
        if not 0 <= index < output.shape[0]:
            continue
        sample_time = start_time + index * dt
        fraction = _projected_rectangle_fraction(
            sample_time - 0.5 * dt - arrival_time,
            sample_time + 0.5 * dt - arrival_time,
            half_x,
            half_y,
        )
        if fraction <= 0.0:
            continue
        output[index] += _sample_weight(
            element=element,
            sample_time=sample_time,
            propagation_delay=propagation_delay,
            weight_scale=weight_scale,
            baffle_distance=baffle_distance,
            sound_speed=sound_speed,
            sampling_frequency=sampling_frequency,
        ) * fraction


def _add_exact_flat_polygon_response(
    output: np.ndarray,
    *,
    point,
    element,
    start_time: float,
    delay: float,
    weight_scale: float,
    baffle_distance: float | None,
    sound_speed: float,
    sampling_frequency: float,
) -> None:
    del baffle_distance
    vertices = np.asarray(element.vertices, dtype=np.float64)
    origin = vertices[0]
    normal = np.asarray(element.normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    tangent_x = vertices[1] - vertices[0]
    tangent_x = tangent_x - np.dot(tangent_x, normal) * normal
    tangent_x_norm = float(np.linalg.norm(tangent_x))
    if tangent_x_norm <= 1e-18:
        _add_projected_polygon_response(
            output,
            point=point,
            element=element,
            start_time=start_time,
            delay=delay,
            propagation_delay=0.0,
            weight_scale=weight_scale,
            baffle_distance=None,
            sound_speed=sound_speed,
            sampling_frequency=sampling_frequency,
        )
        return
    tangent_x = tangent_x / tangent_x_norm
    tangent_y = np.cross(normal, tangent_x)

    signed_distance = float(np.dot(point - origin, normal))
    projected_point = point - signed_distance * normal
    distance_to_plane = abs(signed_distance)
    vertices_2d = np.column_stack(
        [
            (vertices - projected_point) @ tangent_x,
            (vertices - projected_point) @ tangent_y,
        ]
    )

    critical_radii = _flat_polygon_critical_radii(vertices_2d)
    support_start = (
        np.sqrt(distance_to_plane * distance_to_plane + critical_radii[0] * critical_radii[0])
        / sound_speed
        + delay
    )
    support_end = (
        np.sqrt(distance_to_plane * distance_to_plane + critical_radii[-1] * critical_radii[-1])
        / sound_speed
        + delay
    )

    dt = 1.0 / sampling_frequency
    first = int(np.floor((support_start - start_time) * sampling_frequency - 0.5)) - 1
    last = int(np.ceil((support_end - start_time) * sampling_frequency + 0.5)) + 1
    nodes, weights = _gauss_legendre_nodes_weights()

    for index in range(first, last + 1):
        if not 0 <= index < output.shape[0]:
            continue
        sample_time = start_time + index * dt
        bin_start = sample_time - 0.5 * dt - delay
        bin_end = sample_time + 0.5 * dt - delay
        if bin_end <= 0.0:
            continue
        integral = 0.0
        event_times = _flat_polygon_event_times(
            critical_radii,
            distance_to_plane=distance_to_plane,
            sound_speed=sound_speed,
            left=bin_start,
            right=bin_end,
        )
        for left, right in zip(event_times[:-1], event_times[1:], strict=True):
            width = right - left
            if width <= 1e-18:
                continue
            midpoint = 0.5 * (left + right)
            half_width = 0.5 * width
            for node, weight in zip(nodes, weights, strict=True):
                tau = midpoint + half_width * float(node)
                value = _flat_polygon_response_at_tau(
                    tau,
                    vertices_2d=vertices_2d,
                    distance_to_plane=distance_to_plane,
                    sound_speed=sound_speed,
                )
                if value:
                    integral += float(weight) * half_width * value
        if integral:
            output[index] += weight_scale * sampling_frequency * integral


def _add_projected_polygon_response(
    output: np.ndarray,
    *,
    point,
    element,
    start_time: float,
    delay: float,
    propagation_delay: float,
    weight_scale: float,
    baffle_distance: float | None,
    sound_speed: float,
    sampling_frequency: float,
) -> None:
    distance_vector = point - element.center
    distance = float(np.linalg.norm(distance_vector))
    if distance == 0:
        raise ValueError("field points must not coincide with aperture elements")

    direction = distance_vector / distance
    vertices = np.asarray(element.vertices, dtype=np.float64)
    offsets = -((vertices - element.center) @ direction) / sound_speed
    support_start = float(np.min(offsets)) + distance / sound_speed + delay
    support_end = float(np.max(offsets)) + distance / sound_speed + delay
    dt = 1.0 / sampling_frequency
    first = int(np.floor((support_start - start_time) * sampling_frequency - 0.5)) - 1
    last = int(np.ceil((support_end - start_time) * sampling_frequency + 0.5)) + 1

    arrival_time = distance / sound_speed + delay
    total_area = _polygon_area(vertices)
    if total_area <= 1e-30 or support_end - support_start <= 1e-15:
        index = int(np.floor((arrival_time - start_time) * sampling_frequency + 0.5))
        if 0 <= index < output.shape[0]:
            sample_time = start_time + index * dt
            output[index] += _sample_weight(
                element=element,
                sample_time=sample_time,
                propagation_delay=propagation_delay,
                weight_scale=weight_scale,
                baffle_distance=baffle_distance,
                sound_speed=sound_speed,
                sampling_frequency=sampling_frequency,
            )
        return

    for index in range(first, last + 1):
        if not 0 <= index < output.shape[0]:
            continue
        sample_time = start_time + index * dt
        fraction = _projected_polygon_fraction(
            sample_time - 0.5 * dt - arrival_time,
            sample_time + 0.5 * dt - arrival_time,
            vertices,
            offsets,
            total_area,
        )
        if fraction <= 0.0:
            continue
        output[index] += _sample_weight(
            element=element,
            sample_time=sample_time,
            propagation_delay=propagation_delay,
            weight_scale=weight_scale,
            baffle_distance=baffle_distance,
            sound_speed=sound_speed,
            sampling_frequency=sampling_frequency,
        ) * fraction


def _rectangle_tangents(element) -> tuple[np.ndarray, np.ndarray, float, float]:
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
    return tangent_x, tangent_y, width, height


def _fieldii_rectangle_projection_half_widths(
    element,
    *,
    direction: np.ndarray,
    sound_speed: float,
) -> tuple[float, float]:
    vertices = element.vertices
    edge_x = vertices[1] - vertices[0]
    edge_y = vertices[3] - vertices[0]
    if abs(float(edge_x[2])) > 1e-15 and abs(float(edge_y[2])) > 1e-15:
        return (
            abs(float(np.dot(direction, edge_x))) / (2.0 * sound_speed),
            abs(float(np.dot(direction, edge_y))) / (2.0 * sound_speed),
        )
    if (
        abs(float(edge_y[0])) > 1e-15
        and abs(float(edge_y[1])) > 1e-15
        and abs(float(edge_y[2])) > 1e-15
        and abs(float(element.normal[0])) > 1e-12
        and abs(float(element.normal[1])) > 1e-12
    ):
        half_y = _fieldii_edge_projection_half_width_with_length(
            edge_y,
            edge_length=float(np.linalg.norm(edge_y[[1, 2]])),
            direction=direction,
            sound_speed=sound_speed,
        )
    else:
        half_y = _fieldii_edge_projection_half_width(
            edge_y,
            direction=direction,
            sound_speed=sound_speed,
        )
    return (
        _fieldii_edge_projection_half_width(edge_x, direction=direction, sound_speed=sound_speed),
        half_y,
    )


def _fieldii_edge_projection_half_width(
    edge: np.ndarray,
    *,
    direction: np.ndarray,
    sound_speed: float,
) -> float:
    projected = abs(float(np.dot(direction, edge))) / (2.0 * sound_speed)
    planar_length = float(np.max(np.abs(edge[:2])))
    edge_length = float(np.linalg.norm(edge))
    if planar_length <= 1e-18:
        return projected
    return projected * edge_length / planar_length


def _fieldii_edge_projection_half_width_with_length(
    edge: np.ndarray,
    *,
    edge_length: float,
    direction: np.ndarray,
    sound_speed: float,
) -> float:
    projected = abs(float(np.dot(direction, edge))) / (2.0 * sound_speed)
    planar_length = float(np.max(np.abs(edge[:2])))
    if planar_length <= 1e-18:
        return projected
    return projected * edge_length / planar_length


def _projected_rectangle_fraction(left: float, right: float, half_x: float, half_y: float) -> float:
    return _projected_rectangle_cdf(right, half_x, half_y) - _projected_rectangle_cdf(
        left,
        half_x,
        half_y,
    )


def _projected_rectangle_cdf(value: float, half_x: float, half_y: float) -> float:
    large = max(float(half_x), float(half_y))
    small = min(float(half_x), float(half_y))

    if small <= 1e-18:
        if large <= 1e-18:
            return 0.0 if value < 0.0 else 1.0
        if value <= -large:
            return 0.0
        if value >= large:
            return 1.0
        return (value + large) / (2.0 * large)

    outer = large + small
    plateau = large - small
    if value <= -outer:
        return 0.0
    if value >= outer:
        return 1.0
    if value < -plateau:
        return (value + outer) ** 2 / (8.0 * large * small)
    if value <= plateau:
        return small / (2.0 * large) + (value + plateau) / (2.0 * large)
    return 1.0 - (outer - value) ** 2 / (8.0 * large * small)


def _projected_polygon_fraction(
    left: float,
    right: float,
    vertices: np.ndarray,
    offsets: np.ndarray,
    total_area: float,
) -> float:
    return (
        _projected_polygon_cdf(right, vertices, offsets, total_area)
        - _projected_polygon_cdf(left, vertices, offsets, total_area)
    )


def _projected_polygon_cdf(
    value: float,
    vertices: np.ndarray,
    offsets: np.ndarray,
    total_area: float,
) -> float:
    if value <= float(np.min(offsets)):
        return 0.0
    if value >= float(np.max(offsets)):
        return 1.0

    clipped_vertices, _ = _clip_polygon_by_offset(vertices, offsets, value)
    if len(clipped_vertices) < 3:
        return 0.0
    return min(max(_polygon_area(clipped_vertices) / total_area, 0.0), 1.0)


def _clip_polygon_by_offset(
    vertices: np.ndarray,
    offsets: np.ndarray,
    limit: float,
) -> tuple[np.ndarray, np.ndarray]:
    clipped_vertices = []
    clipped_offsets = []
    previous_vertex = vertices[-1]
    previous_offset = float(offsets[-1])
    previous_inside = previous_offset <= limit

    for current_vertex, current_offset_value in zip(vertices, offsets, strict=True):
        current_offset = float(current_offset_value)
        current_inside = current_offset <= limit
        if current_inside != previous_inside:
            fraction = (limit - previous_offset) / (current_offset - previous_offset)
            clipped_vertices.append(previous_vertex + fraction * (current_vertex - previous_vertex))
            clipped_offsets.append(limit)
        if current_inside:
            clipped_vertices.append(current_vertex)
            clipped_offsets.append(current_offset)
        previous_vertex = current_vertex
        previous_offset = current_offset
        previous_inside = current_inside

    return np.asarray(clipped_vertices, dtype=np.float64), np.asarray(clipped_offsets, dtype=np.float64)


def _polygon_area(vertices: np.ndarray) -> float:
    if len(vertices) < 3:
        return 0.0
    origin = vertices[0]
    area = 0.0
    for index in range(1, len(vertices) - 1):
        area += 0.5 * float(
            np.linalg.norm(np.cross(vertices[index] - origin, vertices[index + 1] - origin))
        )
    return area


def _flat_polygon_critical_radii(vertices_2d: np.ndarray) -> np.ndarray:
    radii = []
    origin = np.array([0.0, 0.0], dtype=np.float64)
    if _point_in_polygon_2d(origin, vertices_2d):
        radii.append(0.0)

    for vertex in vertices_2d:
        radii.append(float(np.linalg.norm(vertex)))

    for index in range(len(vertices_2d)):
        start = vertices_2d[index]
        end = vertices_2d[(index + 1) % len(vertices_2d)]
        segment = end - start
        length_squared = float(np.dot(segment, segment))
        if length_squared <= 1e-30:
            continue
        fraction = -float(np.dot(start, segment)) / length_squared
        if -1e-12 <= fraction <= 1.0 + 1e-12:
            closest = start + min(max(fraction, 0.0), 1.0) * segment
            radii.append(float(np.linalg.norm(closest)))

    if not radii:
        return np.array([0.0], dtype=np.float64)

    radii = sorted(max(radius, 0.0) for radius in radii)
    unique_radii = [radii[0]]
    for radius in radii[1:]:
        if radius - unique_radii[-1] > 1e-15:
            unique_radii.append(radius)
    return np.asarray(unique_radii, dtype=np.float64)


def _flat_polygon_event_times(
    critical_radii: np.ndarray,
    *,
    distance_to_plane: float,
    sound_speed: float,
    left: float,
    right: float,
) -> np.ndarray:
    if right <= left:
        return np.asarray([left, right], dtype=np.float64)

    times = [float(left), float(right)]
    for radius in critical_radii:
        time = np.sqrt(distance_to_plane * distance_to_plane + float(radius) * float(radius))
        time /= sound_speed
        if left + 1e-15 < time < right - 1e-15:
            times.append(float(time))

    times = sorted(times)
    unique_times = [times[0]]
    for time in times[1:]:
        if time - unique_times[-1] > 1e-15:
            unique_times.append(time)
    if unique_times[-1] < right:
        unique_times.append(float(right))
    else:
        unique_times[-1] = float(right)
    return np.asarray(unique_times, dtype=np.float64)


def _flat_polygon_propagation_event_times(*, point, element, sound_speed: float) -> np.ndarray:
    vertices = np.asarray(element.vertices, dtype=np.float64)
    normal = np.asarray(element.normal, dtype=np.float64)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-18:
        return np.linalg.norm(point - vertices, axis=1) / sound_speed
    normal = normal / normal_norm

    tangent_x = vertices[1] - vertices[0]
    tangent_x = tangent_x - np.dot(tangent_x, normal) * normal
    tangent_x_norm = float(np.linalg.norm(tangent_x))
    if tangent_x_norm <= 1e-18:
        return np.linalg.norm(point - vertices, axis=1) / sound_speed
    tangent_x = tangent_x / tangent_x_norm
    tangent_y = np.cross(normal, tangent_x)

    signed_distance = float(np.dot(point - vertices[0], normal))
    projected_point = point - signed_distance * normal
    distance_to_plane = abs(signed_distance)
    vertices_2d = np.column_stack(
        [
            (vertices - projected_point) @ tangent_x,
            (vertices - projected_point) @ tangent_y,
        ]
    )
    critical_radii = _flat_polygon_critical_radii(vertices_2d)
    return (
        np.sqrt(distance_to_plane * distance_to_plane + critical_radii * critical_radii)
        / sound_speed
    )


def _flat_polygon_response_at_tau(
    tau: float,
    *,
    vertices_2d: np.ndarray,
    distance_to_plane: float,
    sound_speed: float,
) -> float:
    if tau <= 0.0:
        return 0.0
    radius = sound_speed * tau
    if radius < distance_to_plane:
        return 0.0
    in_plane_radius = np.sqrt(max(radius * radius - distance_to_plane * distance_to_plane, 0.0))
    angle = _circle_polygon_angular_measure(vertices_2d, in_plane_radius)
    if not angle:
        return 0.0
    return float(sound_speed * angle / (2.0 * np.pi))


def _circle_polygon_angular_measure(vertices_2d: np.ndarray, radius: float) -> float:
    if radius <= 1e-18:
        return 2.0 * np.pi if _point_in_polygon_2d(np.array([0.0, 0.0]), vertices_2d) else 0.0

    angles = [0.0, 2.0 * np.pi]
    for vertex in vertices_2d:
        if float(np.dot(vertex, vertex)) > 1e-30:
            angles.append(_positive_angle(np.arctan2(vertex[1], vertex[0])))

    for index in range(len(vertices_2d)):
        start = vertices_2d[index]
        end = vertices_2d[(index + 1) % len(vertices_2d)]
        segment = end - start
        a = float(np.dot(segment, segment))
        if a <= 1e-30:
            continue
        b = 2.0 * float(np.dot(start, segment))
        c = float(np.dot(start, start)) - radius * radius
        discriminant = b * b - 4.0 * a * c
        if discriminant < -1e-24:
            continue
        discriminant = max(discriminant, 0.0)
        root = np.sqrt(discriminant)
        for fraction in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)):
            if -1e-12 <= fraction <= 1.0 + 1e-12:
                point = start + min(max(fraction, 0.0), 1.0) * segment
                angles.append(_positive_angle(np.arctan2(point[1], point[0])))

    angles = np.asarray(sorted(angles), dtype=np.float64)
    unique_angles = [float(angles[0])]
    for angle in angles[1:]:
        if float(angle) - unique_angles[-1] > 1e-12:
            unique_angles.append(float(angle))
    if 2.0 * np.pi - unique_angles[-1] <= 1e-12:
        unique_angles[-1] = 2.0 * np.pi
    elif unique_angles[-1] < 2.0 * np.pi:
        unique_angles.append(2.0 * np.pi)

    measure = 0.0
    for left, right in zip(unique_angles[:-1], unique_angles[1:], strict=True):
        if right <= left:
            continue
        midpoint = 0.5 * (left + right)
        point = radius * np.array([np.cos(midpoint), np.sin(midpoint)], dtype=np.float64)
        if _point_in_polygon_2d(point, vertices_2d):
            measure += right - left
    return float(measure)


def _point_in_polygon_2d(point: np.ndarray, vertices_2d: np.ndarray) -> bool:
    x, y = float(point[0]), float(point[1])
    inside = False
    previous = vertices_2d[-1]
    for current in vertices_2d:
        x0, y0 = float(previous[0]), float(previous[1])
        x1, y1 = float(current[0]), float(current[1])
        edge = current - previous
        rel = point - previous
        cross = float(edge[0] * rel[1] - edge[1] * rel[0])
        if abs(cross) <= 1e-12 and min(x0, x1) - 1e-12 <= x <= max(x0, x1) + 1e-12 and min(
            y0,
            y1,
        ) - 1e-12 <= y <= max(y0, y1) + 1e-12:
            return True
        if (y0 > y) != (y1 > y):
            intersection_x = (x1 - x0) * (y - y0) / (y1 - y0) + x0
            if x < intersection_x:
                inside = not inside
        previous = current
    return inside


def _positive_angle(angle: float) -> float:
    if angle < 0.0:
        angle += 2.0 * np.pi
    if angle >= 2.0 * np.pi:
        angle -= 2.0 * np.pi
    return float(angle)


def _gauss_legendre_nodes_weights() -> tuple[np.ndarray, np.ndarray]:
    return np.polynomial.legendre.leggauss(32)


def _sample_weight(
    *,
    element,
    sample_time: float,
    propagation_delay: float,
    weight_scale: float,
    baffle_distance: float | None,
    sound_speed: float,
    sampling_frequency: float,
) -> float:
    propagation_time = sample_time - propagation_delay
    if propagation_time <= 0:
        return 0.0
    distance = sound_speed * propagation_time
    if baffle_distance is not None:
        weight_scale *= max(baffle_distance / distance, 0.0)
    return weight_scale * element.area / (2.0 * np.pi * distance) * sampling_frequency


def _fieldii_like_time_axis(
    *,
    aperture,
    points: np.ndarray,
    delays: np.ndarray,
    subelement_delays: np.ndarray,
    sound_speed: float,
    sampling_frequency: float,
    leading_pad_samples: int = FIELDII_LEADING_PAD_SAMPLES,
    trailing_pad_samples: int = FIELDII_TRAILING_PAD_SAMPLES,
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
            if aperture.name in {"triangle_aperture", "line_bounded_aperture"} and element.vertices is not None:
                support_times.extend(
                    (
                        _flat_polygon_propagation_event_times(
                            point=point,
                            element=element,
                            sound_speed=sound_speed,
                        )
                        + delays[element.physical_index]
                        + subelement_delays[element_index]
                    ).tolist()
                )

    support_times = np.asarray(support_times, dtype=np.float64)
    support_start = np.floor(np.min(support_times) * sampling_frequency) / sampling_frequency
    support_end = np.ceil(np.max(support_times) * sampling_frequency) / sampling_frequency
    start_time = support_start - leading_pad_samples / sampling_frequency
    end_time = support_end + trailing_pad_samples / sampling_frequency
    sample_count = int(round((end_time - start_time) * sampling_frequency)) + 1
    return float(start_time), sample_count


def _leading_pad_samples(aperture) -> int:
    if "fieldii_leading_pad_samples" in aperture.metadata:
        return int(aperture.metadata["fieldii_leading_pad_samples"])
    if aperture.focus is not None and aperture.focus.__class__.__name__ == "DelayTimeline":
        return 0
    return FIELDII_LEADING_PAD_SAMPLES


def _trailing_pad_samples(aperture, subelement_delays: np.ndarray) -> int:
    if "fieldii_trailing_pad_samples" in aperture.metadata:
        return int(aperture.metadata["fieldii_trailing_pad_samples"])
    if aperture.focus is not None and aperture.focus.__class__.__name__ == "DelayTimeline":
        return FIELDII_TRAILING_PAD_SAMPLES - 1
    if np.any(subelement_delays != 0.0):
        return FIELDII_TRAILING_PAD_SAMPLES - 1
    return FIELDII_TRAILING_PAD_SAMPLES


def _soft_baffle_normal_distance(baffle, *, point, element) -> float | None:
    if baffle is None or baffle.kind == "rigid":
        return None
    return max(float(np.dot(element.normal, point - element.center)), 0.0)
