from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from openfield.responses import TimeResponse

try:  # Optional CPU acceleration; the NumPy path remains the fallback.
    from numba import njit
except Exception:  # pragma: no cover - depends on optional runtime dependency
    njit = None

FIELDII_LEADING_PAD_SAMPLES = 1
FIELDII_TRAILING_PAD_SAMPLES = 5
GAUSS_LEGENDRE_ORDER = 32
_GAUSS_LEGENDRE_NODES_WEIGHTS: tuple[np.ndarray, np.ndarray] | None = None


@dataclass(frozen=True)
class _PreparedFacet:
    vertices: np.ndarray
    normal: np.ndarray
    centroid: np.ndarray
    distance_to_plane: float
    vertices_2d: np.ndarray
    critical_radii: np.ndarray
    angular_data: tuple[np.ndarray, np.ndarray, bool] | None
    support_start: float
    support_end: float


def _points_array(points) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        points = points[None, :]
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    return points


def spatial_impulse_response(simulation, *, aperture, points) -> TimeResponse:
    """Calculate a CPU reference spatial impulse response.

    Polygonal subelements are integrated as exact flat facets. Non-planar
    polygons are triangulated, so curved apertures converge by mesh refinement
    instead of relying on Field II's far-field rectangle approximation.
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
    facets_by_element = [
        _polygon_facets(element) if element.vertices is not None else None
        for element in aperture.elements
    ]
    prepared_facets_by_point_element = [
        [
            _prepare_facets_for_point(
                point=point,
                element=element,
                facets=facets_by_element[element_index],
                sound_speed=sound_speed,
            )
            if facets_by_element[element_index] is not None
            else None
            for element_index, element in enumerate(aperture.elements)
        ]
        for point in points
    ]

    start_time, sample_count = _fieldii_like_time_axis(
        aperture=aperture,
        points=points,
        delays=delays,
        subelement_delays=subelement_delays,
        facets_by_element=facets_by_element,
        prepared_facets_by_point_element=prepared_facets_by_point_element,
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
                start_time=start_time,
                delay=delay,
                propagation_delay=physical_delay,
                weight_scale=weight_scale,
                baffle_distance=baffle_distance,
                facets=facets_by_element[element_index],
                prepared_facets=prepared_facets_by_point_element[point_index][element_index],
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
    start_time: float,
    delay: float,
    propagation_delay: float,
    weight_scale: float,
    baffle_distance: float | None,
    facets: list[tuple[np.ndarray, np.ndarray]] | None,
    prepared_facets: list[_PreparedFacet] | None,
    sound_speed: float,
    sampling_frequency: float,
) -> None:
    distance = float(np.linalg.norm(point - element.center))
    if distance == 0:
        raise ValueError("field points must not coincide with aperture elements")

    arrival_time = distance / sound_speed + delay
    if element.vertices is not None:
        _add_faceted_polygon_response(
            output,
            point=point,
            element=element,
            start_time=start_time,
            delay=delay,
            propagation_delay=propagation_delay,
            weight_scale=weight_scale,
            baffle_distance=baffle_distance,
            facets=facets,
            prepared_facets=prepared_facets,
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


def _add_faceted_polygon_response(
    output: np.ndarray,
    *,
    point,
    element,
    start_time: float,
    delay: float,
    propagation_delay: float,
    weight_scale: float,
    baffle_distance: float | None,
    facets: list[tuple[np.ndarray, np.ndarray]] | None,
    prepared_facets: list[_PreparedFacet] | None = None,
    sound_speed: float,
    sampling_frequency: float,
) -> None:
    if prepared_facets is not None:
        for prepared in prepared_facets:
            facet_baffle_distance = baffle_distance
            if baffle_distance is not None:
                facet_baffle_distance = max(float(np.dot(prepared.normal, point - prepared.centroid)), 0.0)
            _add_prepared_exact_flat_polygon_response(
                output,
                prepared=prepared,
                start_time=start_time,
                delay=delay,
                propagation_delay=propagation_delay,
                weight_scale=weight_scale,
                baffle_distance=facet_baffle_distance,
                sound_speed=sound_speed,
                sampling_frequency=sampling_frequency,
            )
        return

    if facets is None:
        facets = _polygon_facets(element)
    for vertices, normal in facets:
        facet_baffle_distance = baffle_distance
        if baffle_distance is not None:
            facet_baffle_distance = max(float(np.dot(normal, point - np.mean(vertices, axis=0))), 0.0)
        _add_exact_flat_polygon_response(
            output,
            point=point,
            element=element,
            vertices=vertices,
            normal=normal,
            start_time=start_time,
            delay=delay,
            propagation_delay=propagation_delay,
            weight_scale=weight_scale,
            baffle_distance=facet_baffle_distance,
            sound_speed=sound_speed,
            sampling_frequency=sampling_frequency,
        )


def _prepare_facets_for_point(
    *,
    point,
    element,
    facets: list[tuple[np.ndarray, np.ndarray]] | None,
    sound_speed: float,
) -> list[_PreparedFacet] | None:
    if facets is None:
        return None
    prepared = []
    for vertices, normal in facets:
        facet = _prepare_flat_facet(point=point, vertices=vertices, normal=normal, sound_speed=sound_speed)
        if facet is None:
            return None
        prepared.append(facet)
    return prepared


def _dot3(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[0] + first[1] * second[1] + first[2] * second[2])


def _norm3(vector: np.ndarray) -> float:
    return math.sqrt(_dot3(vector, vector))


def _cross3(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.array(
        [
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        ],
        dtype=np.float64,
    )


def _project_vertices_2d(
    vertices: np.ndarray,
    origin: np.ndarray,
    tangent_x: np.ndarray,
    tangent_y: np.ndarray,
) -> np.ndarray:
    projected = np.empty((vertices.shape[0], 2), dtype=np.float64)
    for index in range(vertices.shape[0]):
        rel = vertices[index] - origin
        projected[index, 0] = _dot3(rel, tangent_x)
        projected[index, 1] = _dot3(rel, tangent_y)
    return projected


def _centroid(vertices: np.ndarray) -> np.ndarray:
    center = np.zeros(3, dtype=np.float64)
    for vertex in vertices:
        center += vertex
    return center / vertices.shape[0]


def _prepare_flat_facet(
    *,
    point,
    vertices: np.ndarray,
    normal: np.ndarray,
    sound_speed: float,
) -> _PreparedFacet | None:
    vertices = np.asarray(vertices, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    if _prepare_flat_facet_numba is not None:
        (
            ok,
            prepared_normal,
            centroid,
            distance_to_plane,
            vertices_2d,
            critical_radii,
            critical_count,
            normal_angles,
            thresholds,
            angular_count,
            origin_inside,
            support_start,
            support_end,
        ) = _prepare_flat_facet_numba(point, vertices, normal, sound_speed)
        if not ok:
            return None
        angular_data = None
        if angular_count > 0:
            angular_data = (
                normal_angles[:angular_count].copy(),
                thresholds[:angular_count].copy(),
                bool(origin_inside),
            )
        return _PreparedFacet(
            vertices=vertices,
            normal=prepared_normal,
            centroid=centroid,
            distance_to_plane=float(distance_to_plane),
            vertices_2d=vertices_2d,
            critical_radii=critical_radii[:critical_count].copy(),
            angular_data=angular_data,
            support_start=float(support_start),
            support_end=float(support_end),
        )

    normal_norm = _norm3(normal)
    if normal_norm <= 1e-18:
        return None
    normal = normal / normal_norm
    tangent_x = vertices[1] - vertices[0]
    tangent_x = tangent_x - _dot3(tangent_x, normal) * normal
    tangent_x_norm = _norm3(tangent_x)
    if tangent_x_norm <= 1e-18:
        return None
    tangent_x = tangent_x / tangent_x_norm
    tangent_y = _cross3(normal, tangent_x)

    signed_distance = _dot3(point - vertices[0], normal)
    projected_point = point - signed_distance * normal
    distance_to_plane = abs(signed_distance)
    vertices_2d = _project_vertices_2d(vertices, projected_point, tangent_x, tangent_y)
    critical_radii = _flat_polygon_critical_radii(vertices_2d)
    support_start = (
        math.sqrt(distance_to_plane * distance_to_plane + critical_radii[0] * critical_radii[0])
        / sound_speed
    )
    support_end = (
        math.sqrt(distance_to_plane * distance_to_plane + critical_radii[-1] * critical_radii[-1])
        / sound_speed
    )
    return _PreparedFacet(
        vertices=vertices,
        normal=normal,
        centroid=_centroid(vertices),
        distance_to_plane=distance_to_plane,
        vertices_2d=vertices_2d,
        critical_radii=critical_radii,
        angular_data=_convex_polygon_angular_data(vertices_2d),
        support_start=float(support_start),
        support_end=float(support_end),
    )


def _add_prepared_exact_flat_polygon_response(
    output: np.ndarray,
    *,
    prepared: _PreparedFacet,
    start_time: float,
    delay: float,
    propagation_delay: float,
    weight_scale: float,
    baffle_distance: float | None,
    sound_speed: float,
    sampling_frequency: float,
) -> None:
    support_start = prepared.support_start + delay
    support_end = prepared.support_end + delay
    dt = 1.0 / sampling_frequency
    first = int(np.floor((support_start - start_time) * sampling_frequency - 0.5)) - 1
    last = int(np.ceil((support_end - start_time) * sampling_frequency + 0.5)) + 1
    nodes, weights = _gauss_legendre_nodes_weights()

    if prepared.angular_data is not None and _add_exact_flat_polygon_response_numba is not None:
        normal_angles, thresholds, origin_inside = prepared.angular_data
        _add_exact_flat_polygon_response_numba(
            output,
            normal_angles,
            thresholds,
            bool(origin_inside),
            prepared.critical_radii,
            prepared.distance_to_plane,
            first,
            last,
            start_time,
            delay,
            propagation_delay,
            weight_scale,
            0.0 if baffle_distance is None else float(baffle_distance),
            baffle_distance is not None,
            sound_speed,
            sampling_frequency,
            nodes,
            weights,
        )
        return

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
            prepared.critical_radii,
            distance_to_plane=prepared.distance_to_plane,
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
                    vertices_2d=prepared.vertices_2d,
                    angular_data=prepared.angular_data,
                    distance_to_plane=prepared.distance_to_plane,
                    sound_speed=sound_speed,
                )
                if value:
                    integral += float(weight) * half_width * value
        if integral:
            output[index] += (
                weight_scale
                * _baffle_scale(
                    baffle_distance,
                    sample_time=sample_time,
                    propagation_delay=propagation_delay,
                    sound_speed=sound_speed,
                )
                * sampling_frequency
                * integral
            )


def _add_exact_flat_polygon_response(
    output: np.ndarray,
    *,
    point,
    element,
    vertices: np.ndarray | None = None,
    normal: np.ndarray | None = None,
    start_time: float,
    delay: float,
    propagation_delay: float,
    weight_scale: float,
    baffle_distance: float | None,
    sound_speed: float,
    sampling_frequency: float,
) -> None:
    if vertices is None:
        vertices = np.asarray(element.vertices, dtype=np.float64)
    else:
        vertices = np.asarray(vertices, dtype=np.float64)
    origin = vertices[0]
    if normal is None:
        normal = np.asarray(element.normal, dtype=np.float64)
    else:
        normal = np.asarray(normal, dtype=np.float64)
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
            propagation_delay=propagation_delay,
            weight_scale=weight_scale,
            baffle_distance=baffle_distance,
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
    angular_data = _convex_polygon_angular_data(vertices_2d)

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

    if angular_data is not None and _add_exact_flat_polygon_response_numba is not None:
        normal_angles, thresholds, origin_inside = angular_data
        _add_exact_flat_polygon_response_numba(
            output,
            normal_angles,
            thresholds,
            bool(origin_inside),
            critical_radii,
            distance_to_plane,
            first,
            last,
            start_time,
            delay,
            propagation_delay,
            weight_scale,
            0.0 if baffle_distance is None else float(baffle_distance),
            baffle_distance is not None,
            sound_speed,
            sampling_frequency,
            nodes,
            weights,
        )
        return

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
                    angular_data=angular_data,
                    distance_to_plane=distance_to_plane,
                    sound_speed=sound_speed,
                )
                if value:
                    integral += float(weight) * half_width * value
        if integral:
            output[index] += (
                weight_scale
                * _baffle_scale(
                    baffle_distance,
                    sample_time=sample_time,
                    propagation_delay=propagation_delay,
                    sound_speed=sound_speed,
                )
                * sampling_frequency
                * integral
            )


def _polygon_facets(element) -> list[tuple[np.ndarray, np.ndarray]]:
    vertices = np.asarray(element.vertices, dtype=np.float64)
    if vertices.shape[0] == 3 or _is_coplanar(vertices):
        return [(vertices, _facet_normal(vertices, reference_normal=element.normal))]

    facets = []
    for index in range(1, vertices.shape[0] - 1):
        triangle = vertices[[0, index, index + 1]]
        facets.append((triangle, _facet_normal(triangle, reference_normal=element.normal)))
    return facets


def _is_coplanar(vertices: np.ndarray) -> bool:
    origin = vertices[0]
    normal = np.zeros(3, dtype=np.float64)
    for index in range(1, vertices.shape[0] - 1):
        normal += _cross3(vertices[index] - origin, vertices[index + 1] - origin)
    norm = _norm3(normal)
    if norm <= 1e-30:
        return True
    normal /= norm
    max_distance = 0.0
    for vertex in vertices:
        max_distance = max(max_distance, abs(_dot3(vertex - origin, normal)))
    return bool(max_distance <= 1e-12)


def _facet_normal(vertices: np.ndarray, *, reference_normal: np.ndarray) -> np.ndarray:
    origin = vertices[0]
    normal = np.zeros(3, dtype=np.float64)
    for index in range(1, vertices.shape[0] - 1):
        normal += _cross3(vertices[index] - origin, vertices[index + 1] - origin)
    norm = _norm3(normal)
    if norm <= 1e-30:
        raise ValueError("facet area must be positive")
    normal /= norm
    reference_normal = np.asarray(reference_normal, dtype=np.float64)
    if _dot3(normal, reference_normal) < 0.0:
        normal = -normal
    return normal


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
        radii.append(math.hypot(float(vertex[0]), float(vertex[1])))

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
            radii.append(math.hypot(float(closest[0]), float(closest[1])))

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


def _faceted_polygon_propagation_event_times(
    *,
    point,
    element,
    sound_speed: float,
    facets: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> np.ndarray:
    times = []
    if facets is None:
        facets = _polygon_facets(element)
    for vertices, normal in facets:
        times.extend(
            _flat_polygon_propagation_event_times(
                point=point,
                vertices=vertices,
                normal=normal,
                sound_speed=sound_speed,
            ).tolist()
        )
    return np.asarray(times, dtype=np.float64)


def _faceted_polygon_propagation_bounds(
    *,
    point,
    element,
    sound_speed: float,
    facets: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[float, float]:
    if facets is None:
        facets = _polygon_facets(element)
    lower = math.inf
    upper = 0.0
    for vertices, normal in facets:
        start, end = _flat_polygon_propagation_bounds(
            point=point,
            vertices=vertices,
            normal=normal,
            sound_speed=sound_speed,
        )
        lower = min(lower, start)
        upper = max(upper, end)
    return lower, upper


def _flat_polygon_propagation_bounds(
    *,
    point,
    vertices: np.ndarray,
    normal: np.ndarray,
    sound_speed: float,
) -> tuple[float, float]:
    vertices = np.asarray(vertices, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-18:
        distances = np.linalg.norm(point - vertices, axis=1) / sound_speed
        return float(np.min(distances)), float(np.max(distances))
    normal = normal / normal_norm

    tangent_x = vertices[1] - vertices[0]
    tangent_x = tangent_x - np.dot(tangent_x, normal) * normal
    tangent_x_norm = float(np.linalg.norm(tangent_x))
    if tangent_x_norm <= 1e-18:
        distances = np.linalg.norm(point - vertices, axis=1) / sound_speed
        return float(np.min(distances)), float(np.max(distances))
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
    min_radius = _flat_polygon_min_radius(vertices_2d)
    max_radius = float(np.max(np.linalg.norm(vertices_2d, axis=1)))
    return (
        math.sqrt(distance_to_plane * distance_to_plane + min_radius * min_radius)
        / sound_speed,
        math.sqrt(distance_to_plane * distance_to_plane + max_radius * max_radius)
        / sound_speed,
    )


def _flat_polygon_min_radius(vertices_2d: np.ndarray) -> float:
    origin = np.array([0.0, 0.0], dtype=np.float64)
    if _point_in_polygon_2d(origin, vertices_2d):
        return 0.0

    minimum = math.inf
    for index in range(len(vertices_2d)):
        start = vertices_2d[index]
        end = vertices_2d[(index + 1) % len(vertices_2d)]
        segment = end - start
        length_squared = float(np.dot(segment, segment))
        if length_squared <= 1e-30:
            candidate = float(np.linalg.norm(start))
        else:
            fraction = -float(np.dot(start, segment)) / length_squared
            fraction = min(max(fraction, 0.0), 1.0)
            closest = start + fraction * segment
            candidate = math.hypot(float(closest[0]), float(closest[1]))
        minimum = min(minimum, candidate)
    return float(minimum)


def _flat_polygon_propagation_event_times(
    *,
    point,
    vertices: np.ndarray,
    normal: np.ndarray,
    sound_speed: float,
) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
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
    angular_data: tuple[np.ndarray, np.ndarray, bool] | None = None,
    distance_to_plane: float,
    sound_speed: float,
) -> float:
    if tau <= 0.0:
        return 0.0
    radius = sound_speed * tau
    if radius < distance_to_plane:
        return 0.0
    in_plane_radius = math.sqrt(max(radius * radius - distance_to_plane * distance_to_plane, 0.0))
    if angular_data is None:
        angle = _circle_polygon_angular_measure(vertices_2d, in_plane_radius)
    else:
        angle = _convex_polygon_angular_measure_from_data(angular_data, in_plane_radius)
    if not angle:
        return 0.0
    return float(sound_speed * angle / (2.0 * np.pi))


def _circle_polygon_angular_measure(vertices_2d: np.ndarray, radius: float) -> float:
    angular_data = _convex_polygon_angular_data(vertices_2d)
    if angular_data is not None:
        return _convex_polygon_angular_measure_from_data(angular_data, radius)
    return _circle_polygon_angular_measure_by_events(vertices_2d, radius)


def _convex_polygon_angular_data(vertices_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool] | None:
    area2 = _signed_area2(vertices_2d)
    if abs(area2) <= 1e-30:
        return None
    orientation = 1.0 if area2 > 0.0 else -1.0

    normal_angles = []
    thresholds = []
    for index in range(len(vertices_2d)):
        previous = vertices_2d[index - 1]
        current = vertices_2d[index]
        following = vertices_2d[(index + 1) % len(vertices_2d)]
        first = current - previous
        second = following - current
        cross = float(first[0] * second[1] - first[1] * second[0])
        if orientation * cross < -1e-12:
            return None

        edge = following - current
        edge_norm = math.hypot(float(edge[0]), float(edge[1]))
        if edge_norm <= 1e-30:
            continue
        normal_x = -orientation * float(edge[1])
        normal_y = orientation * float(edge[0])
        threshold = orientation * float(edge[0] * current[1] - edge[1] * current[0])
        normal_angles.append(_positive_angle(math.atan2(normal_y, normal_x)))
        thresholds.append(threshold / edge_norm)

    origin_inside = _point_in_polygon_2d(np.array([0.0, 0.0]), vertices_2d)
    return (
        np.asarray(normal_angles, dtype=np.float64),
        np.asarray(thresholds, dtype=np.float64),
        bool(origin_inside),
    )


def _convex_polygon_angular_measure_from_data(
    angular_data: tuple[np.ndarray, np.ndarray, bool],
    radius: float,
) -> float:
    normal_angles, thresholds, origin_inside = angular_data
    if radius <= 1e-18:
        return 2.0 * np.pi if origin_inside else 0.0
    intervals = [(0.0, 2.0 * np.pi)]

    for center, threshold in zip(normal_angles, thresholds, strict=True):
        q = float(threshold) / radius
        if q >= 1.0:
            if q > 1.0 + 1e-12:
                return 0.0
            edge_intervals = [(float(center), float(center))]
        elif q <= -1.0:
            continue
        else:
            delta = math.acos(q)
            left = _positive_angle(center - delta)
            right = _positive_angle(center + delta)
            if left <= right:
                edge_intervals = [(left, right)]
            else:
                edge_intervals = [(left, 2.0 * np.pi), (0.0, right)]

        intervals = _intersect_angle_intervals(intervals, edge_intervals)
        if not intervals:
            return 0.0

    return float(sum(right - left for left, right in intervals))


def _intersect_angle_intervals(
    first: list[tuple[float, float]],
    second: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    intersections = []
    for left_a, right_a in first:
        for left_b, right_b in second:
            left = max(left_a, left_b)
            right = min(right_a, right_b)
            if right - left > 1e-14:
                intersections.append((left, right))
    return intersections


def _is_convex_polygon_2d(vertices_2d: np.ndarray) -> bool:
    area2 = _signed_area2(vertices_2d)
    if abs(area2) <= 1e-30:
        return False
    orientation = 1.0 if area2 > 0.0 else -1.0
    for index in range(len(vertices_2d)):
        previous = vertices_2d[index - 1]
        current = vertices_2d[index]
        following = vertices_2d[(index + 1) % len(vertices_2d)]
        first = current - previous
        second = following - current
        cross = float(first[0] * second[1] - first[1] * second[0])
        if orientation * cross < -1e-12:
            return False
    return True


def _signed_area2(vertices_2d: np.ndarray) -> float:
    area2 = 0.0
    for index in range(len(vertices_2d)):
        start = vertices_2d[index]
        end = vertices_2d[(index + 1) % len(vertices_2d)]
        area2 += float(start[0] * end[1] - start[1] * end[0])
    return area2


def _circle_polygon_angular_measure_by_events(vertices_2d: np.ndarray, radius: float) -> float:
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


if njit is not None:

    @njit(cache=True)
    def _positive_angle_numba(angle):
        two_pi = 2.0 * math.pi
        if angle < 0.0:
            angle += two_pi
        if angle >= two_pi:
            angle -= two_pi
        return angle


    @njit(cache=True)
    def _point_in_polygon_2d_numba(vertices_2d, count, x, y):
        inside = False
        previous_x = vertices_2d[count - 1, 0]
        previous_y = vertices_2d[count - 1, 1]
        for index in range(count):
            current_x = vertices_2d[index, 0]
            current_y = vertices_2d[index, 1]
            edge_x = current_x - previous_x
            edge_y = current_y - previous_y
            rel_x = x - previous_x
            rel_y = y - previous_y
            cross = edge_x * rel_y - edge_y * rel_x
            if (
                abs(cross) <= 1e-12
                and min(previous_x, current_x) - 1e-12 <= x <= max(previous_x, current_x) + 1e-12
                and min(previous_y, current_y) - 1e-12 <= y <= max(previous_y, current_y) + 1e-12
            ):
                return True
            if (previous_y > y) != (current_y > y):
                intersection_x = (current_x - previous_x) * (y - previous_y) / (current_y - previous_y) + previous_x
                if x < intersection_x:
                    inside = not inside
            previous_x = current_x
            previous_y = current_y
        return inside


    @njit(cache=True)
    def _sort_unique_positive_numba(values, count):
        for index in range(1, count):
            value = values[index]
            position = index - 1
            while position >= 0 and values[position] > value:
                values[position + 1] = values[position]
                position -= 1
            values[position + 1] = value

        unique_count = 0
        for index in range(count):
            value = values[index]
            if value < 0.0:
                value = 0.0
            if unique_count == 0 or value - values[unique_count - 1] > 1e-15:
                values[unique_count] = value
                unique_count += 1
        return unique_count


    @njit(cache=True)
    def _prepare_flat_facet_numba(point, vertices, normal, sound_speed):
        vertex_count = vertices.shape[0]
        prepared_normal = np.empty(3, dtype=np.float64)
        centroid = np.zeros(3, dtype=np.float64)
        vertices_2d = np.empty((vertex_count, 2), dtype=np.float64)
        critical_radii = np.empty(2 * vertex_count + 1, dtype=np.float64)
        normal_angles = np.empty(vertex_count, dtype=np.float64)
        thresholds = np.empty(vertex_count, dtype=np.float64)

        normal_norm = math.sqrt(
            normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]
        )
        if normal_norm <= 1e-18:
            return (
                False,
                prepared_normal,
                centroid,
                0.0,
                vertices_2d,
                critical_radii,
                0,
                normal_angles,
                thresholds,
                0,
                False,
                0.0,
                0.0,
            )
        for axis in range(3):
            prepared_normal[axis] = normal[axis] / normal_norm

        tangent_x = np.empty(3, dtype=np.float64)
        for axis in range(3):
            tangent_x[axis] = vertices[1, axis] - vertices[0, axis]
        tangent_dot = (
            tangent_x[0] * prepared_normal[0]
            + tangent_x[1] * prepared_normal[1]
            + tangent_x[2] * prepared_normal[2]
        )
        for axis in range(3):
            tangent_x[axis] -= tangent_dot * prepared_normal[axis]
        tangent_x_norm = math.sqrt(
            tangent_x[0] * tangent_x[0] + tangent_x[1] * tangent_x[1] + tangent_x[2] * tangent_x[2]
        )
        if tangent_x_norm <= 1e-18:
            return (
                False,
                prepared_normal,
                centroid,
                0.0,
                vertices_2d,
                critical_radii,
                0,
                normal_angles,
                thresholds,
                0,
                False,
                0.0,
                0.0,
            )
        for axis in range(3):
            tangent_x[axis] /= tangent_x_norm

        tangent_y = np.empty(3, dtype=np.float64)
        tangent_y[0] = prepared_normal[1] * tangent_x[2] - prepared_normal[2] * tangent_x[1]
        tangent_y[1] = prepared_normal[2] * tangent_x[0] - prepared_normal[0] * tangent_x[2]
        tangent_y[2] = prepared_normal[0] * tangent_x[1] - prepared_normal[1] * tangent_x[0]

        rel0 = point[0] - vertices[0, 0]
        rel1 = point[1] - vertices[0, 1]
        rel2 = point[2] - vertices[0, 2]
        signed_distance = rel0 * prepared_normal[0] + rel1 * prepared_normal[1] + rel2 * prepared_normal[2]
        distance_to_plane = abs(signed_distance)
        projected_point = np.empty(3, dtype=np.float64)
        for axis in range(3):
            projected_point[axis] = point[axis] - signed_distance * prepared_normal[axis]

        for vertex_index in range(vertex_count):
            rel_x = vertices[vertex_index, 0] - projected_point[0]
            rel_y = vertices[vertex_index, 1] - projected_point[1]
            rel_z = vertices[vertex_index, 2] - projected_point[2]
            vertices_2d[vertex_index, 0] = rel_x * tangent_x[0] + rel_y * tangent_x[1] + rel_z * tangent_x[2]
            vertices_2d[vertex_index, 1] = rel_x * tangent_y[0] + rel_y * tangent_y[1] + rel_z * tangent_y[2]
            centroid[0] += vertices[vertex_index, 0]
            centroid[1] += vertices[vertex_index, 1]
            centroid[2] += vertices[vertex_index, 2]
        centroid[0] /= vertex_count
        centroid[1] /= vertex_count
        centroid[2] /= vertex_count

        critical_count = 0
        origin_inside = _point_in_polygon_2d_numba(vertices_2d, vertex_count, 0.0, 0.0)
        if origin_inside:
            critical_radii[critical_count] = 0.0
            critical_count += 1

        for vertex_index in range(vertex_count):
            x = vertices_2d[vertex_index, 0]
            y = vertices_2d[vertex_index, 1]
            critical_radii[critical_count] = math.sqrt(x * x + y * y)
            critical_count += 1

        for vertex_index in range(vertex_count):
            start_x = vertices_2d[vertex_index, 0]
            start_y = vertices_2d[vertex_index, 1]
            next_index = (vertex_index + 1) % vertex_count
            segment_x = vertices_2d[next_index, 0] - start_x
            segment_y = vertices_2d[next_index, 1] - start_y
            length_squared = segment_x * segment_x + segment_y * segment_y
            if length_squared <= 1e-30:
                continue
            fraction = -(start_x * segment_x + start_y * segment_y) / length_squared
            if -1e-12 <= fraction <= 1.0 + 1e-12:
                if fraction < 0.0:
                    fraction = 0.0
                elif fraction > 1.0:
                    fraction = 1.0
                closest_x = start_x + fraction * segment_x
                closest_y = start_y + fraction * segment_y
                critical_radii[critical_count] = math.sqrt(closest_x * closest_x + closest_y * closest_y)
                critical_count += 1

        critical_count = _sort_unique_positive_numba(critical_radii, critical_count)

        area2 = 0.0
        for vertex_index in range(vertex_count):
            next_index = (vertex_index + 1) % vertex_count
            area2 += (
                vertices_2d[vertex_index, 0] * vertices_2d[next_index, 1]
                - vertices_2d[vertex_index, 1] * vertices_2d[next_index, 0]
            )
        angular_count = 0
        if abs(area2) > 1e-30:
            orientation = 1.0 if area2 > 0.0 else -1.0
            convex = True
            for vertex_index in range(vertex_count):
                previous_index = (vertex_index + vertex_count - 1) % vertex_count
                next_index = (vertex_index + 1) % vertex_count
                first_x = vertices_2d[vertex_index, 0] - vertices_2d[previous_index, 0]
                first_y = vertices_2d[vertex_index, 1] - vertices_2d[previous_index, 1]
                second_x = vertices_2d[next_index, 0] - vertices_2d[vertex_index, 0]
                second_y = vertices_2d[next_index, 1] - vertices_2d[vertex_index, 1]
                cross = first_x * second_y - first_y * second_x
                if orientation * cross < -1e-12:
                    convex = False
                    break
            if convex:
                for vertex_index in range(vertex_count):
                    next_index = (vertex_index + 1) % vertex_count
                    edge_x = vertices_2d[next_index, 0] - vertices_2d[vertex_index, 0]
                    edge_y = vertices_2d[next_index, 1] - vertices_2d[vertex_index, 1]
                    edge_norm = math.sqrt(edge_x * edge_x + edge_y * edge_y)
                    if edge_norm <= 1e-30:
                        continue
                    normal_x = -orientation * edge_y
                    normal_y = orientation * edge_x
                    threshold = orientation * (
                        edge_x * vertices_2d[vertex_index, 1]
                        - edge_y * vertices_2d[vertex_index, 0]
                    )
                    normal_angles[angular_count] = _positive_angle_numba(math.atan2(normal_y, normal_x))
                    thresholds[angular_count] = threshold / edge_norm
                    angular_count += 1

        support_start = math.sqrt(
            distance_to_plane * distance_to_plane + critical_radii[0] * critical_radii[0]
        ) / sound_speed
        support_end = math.sqrt(
            distance_to_plane * distance_to_plane
            + critical_radii[critical_count - 1] * critical_radii[critical_count - 1]
        ) / sound_speed
        return (
            True,
            prepared_normal,
            centroid,
            distance_to_plane,
            vertices_2d,
            critical_radii,
            critical_count,
            normal_angles,
            thresholds,
            angular_count,
            origin_inside,
            support_start,
            support_end,
        )


    @njit(cache=True)
    def _convex_polygon_angular_measure_numba(
        normal_angles,
        thresholds,
        origin_inside,
        radius,
        lefts,
        rights,
        new_lefts,
        new_rights,
        edge_lefts,
        edge_rights,
    ):
        two_pi = 2.0 * math.pi
        if radius <= 1e-18:
            return two_pi if origin_inside else 0.0

        interval_count = 1
        lefts[0] = 0.0
        rights[0] = two_pi

        for edge_index in range(normal_angles.shape[0]):
            q = thresholds[edge_index] / radius
            if q >= 1.0:
                if q > 1.0 + 1e-12:
                    return 0.0
                edge_count = 1
                edge_lefts[0] = normal_angles[edge_index]
                edge_rights[0] = normal_angles[edge_index]
            elif q <= -1.0:
                continue
            else:
                center = normal_angles[edge_index]
                delta = math.acos(q)
                left = _positive_angle_numba(center - delta)
                right = _positive_angle_numba(center + delta)
                if left <= right:
                    edge_count = 1
                    edge_lefts[0] = left
                    edge_rights[0] = right
                else:
                    edge_count = 2
                    edge_lefts[0] = left
                    edge_rights[0] = two_pi
                    edge_lefts[1] = 0.0
                    edge_rights[1] = right

            new_count = 0
            for interval_index in range(interval_count):
                for edge_interval_index in range(edge_count):
                    left = lefts[interval_index]
                    if edge_lefts[edge_interval_index] > left:
                        left = edge_lefts[edge_interval_index]
                    right = rights[interval_index]
                    if edge_rights[edge_interval_index] < right:
                        right = edge_rights[edge_interval_index]
                    if right - left > 1e-14 and new_count < 8:
                        new_lefts[new_count] = left
                        new_rights[new_count] = right
                        new_count += 1

            if new_count == 0:
                return 0.0
            interval_count = new_count
            for interval_index in range(interval_count):
                lefts[interval_index] = new_lefts[interval_index]
                rights[interval_index] = new_rights[interval_index]

        measure = 0.0
        for interval_index in range(interval_count):
            measure += rights[interval_index] - lefts[interval_index]
        return measure


    @njit(cache=True)
    def _add_exact_flat_polygon_response_numba(
        output,
        normal_angles,
        thresholds,
        origin_inside,
        critical_radii,
        distance_to_plane,
        first,
        last,
        start_time,
        delay,
        propagation_delay,
        weight_scale,
        baffle_distance,
        has_baffle,
        sound_speed,
        sampling_frequency,
        nodes,
        weights,
    ):
        dt = 1.0 / sampling_frequency
        event_times = np.empty(critical_radii.shape[0] + 2, dtype=np.float64)
        lefts = np.empty(8, dtype=np.float64)
        rights = np.empty(8, dtype=np.float64)
        new_lefts = np.empty(8, dtype=np.float64)
        new_rights = np.empty(8, dtype=np.float64)
        edge_lefts = np.empty(2, dtype=np.float64)
        edge_rights = np.empty(2, dtype=np.float64)
        for index in range(first, last + 1):
            if index < 0 or index >= output.shape[0]:
                continue

            sample_time = start_time + index * dt
            bin_start = sample_time - 0.5 * dt - delay
            bin_end = sample_time + 0.5 * dt - delay
            if bin_end <= 0.0:
                continue

            event_count = 1
            event_times[0] = bin_start
            for radius_index in range(critical_radii.shape[0]):
                radius = critical_radii[radius_index]
                event_time = math.sqrt(
                    distance_to_plane * distance_to_plane + radius * radius
                ) / sound_speed
                if bin_start + 1e-15 < event_time < bin_end - 1e-15:
                    event_times[event_count] = event_time
                    event_count += 1
            event_times[event_count] = bin_end
            event_count += 1

            integral = 0.0
            for event_index in range(event_count - 1):
                left = event_times[event_index]
                right = event_times[event_index + 1]
                width = right - left
                if width <= 1e-18:
                    continue
                midpoint = 0.5 * (left + right)
                half_width = 0.5 * width
                for node_index in range(nodes.shape[0]):
                    tau = midpoint + half_width * nodes[node_index]
                    if tau <= 0.0:
                        continue
                    radius = sound_speed * tau
                    if radius < distance_to_plane:
                        continue
                    in_plane = math.sqrt(
                        max(radius * radius - distance_to_plane * distance_to_plane, 0.0)
                    )
                    angle = _convex_polygon_angular_measure_numba(
                        normal_angles,
                        thresholds,
                        origin_inside,
                        in_plane,
                        lefts,
                        rights,
                        new_lefts,
                        new_rights,
                        edge_lefts,
                        edge_rights,
                    )
                    if angle != 0.0:
                        value = sound_speed * angle / (2.0 * math.pi)
                        integral += weights[node_index] * half_width * value

            if integral != 0.0:
                baffle_scale = 1.0
                if has_baffle:
                    propagation_time = sample_time - propagation_delay
                    if propagation_time <= 0.0:
                        baffle_scale = 0.0
                    else:
                        baffle_scale = baffle_distance / (sound_speed * propagation_time)
                        if baffle_scale < 0.0:
                            baffle_scale = 0.0
                output[index] += weight_scale * baffle_scale * sampling_frequency * integral

else:
    _prepare_flat_facet_numba = None
    _add_exact_flat_polygon_response_numba = None


def _gauss_legendre_nodes_weights() -> tuple[np.ndarray, np.ndarray]:
    global _GAUSS_LEGENDRE_NODES_WEIGHTS
    if _GAUSS_LEGENDRE_NODES_WEIGHTS is None:
        _GAUSS_LEGENDRE_NODES_WEIGHTS = np.polynomial.legendre.leggauss(GAUSS_LEGENDRE_ORDER)
    return _GAUSS_LEGENDRE_NODES_WEIGHTS


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
    weight_scale *= _baffle_scale(
        baffle_distance,
        sample_time=sample_time,
        propagation_delay=propagation_delay,
        sound_speed=sound_speed,
    )
    return weight_scale * element.area / (2.0 * np.pi * distance) * sampling_frequency


def _baffle_scale(
    baffle_distance: float | None,
    *,
    sample_time: float,
    propagation_delay: float,
    sound_speed: float,
) -> float:
    if baffle_distance is None:
        return 1.0
    propagation_time = sample_time - propagation_delay
    if propagation_time <= 0:
        return 0.0
    distance = sound_speed * propagation_time
    return max(float(baffle_distance) / distance, 0.0)


def _fieldii_like_time_axis(
    *,
    aperture,
    points: np.ndarray,
    delays: np.ndarray,
    subelement_delays: np.ndarray,
    facets_by_element: list[list[tuple[np.ndarray, np.ndarray]] | None],
    prepared_facets_by_point_element: list[list[list[_PreparedFacet] | None]],
    sound_speed: float,
    sampling_frequency: float,
    leading_pad_samples: int = FIELDII_LEADING_PAD_SAMPLES,
    trailing_pad_samples: int = FIELDII_TRAILING_PAD_SAMPLES,
) -> tuple[float, int]:
    support_times = []
    for point_index, point in enumerate(points):
        for element_index, element in enumerate(aperture.elements):
            offset = delays[element.physical_index] + subelement_delays[element_index]
            if element.vertices is not None:
                prepared_facets = prepared_facets_by_point_element[point_index][element_index]
                if prepared_facets is not None:
                    for prepared in prepared_facets:
                        support_times.append(prepared.support_start + offset)
                        support_times.append(prepared.support_end + offset)
                else:
                    start, end = _faceted_polygon_propagation_bounds(
                        point=point,
                        element=element,
                        sound_speed=sound_speed,
                        facets=facets_by_element[element_index],
                    )
                    support_times.append(start + offset)
                    support_times.append(end + offset)
            else:
                support_times.append(float(np.linalg.norm(point - element.center)) / sound_speed + offset)

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
