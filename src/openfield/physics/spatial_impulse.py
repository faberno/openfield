from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from openfield.physics.flat_facet_kernel import (
    add_spatial_impulse_response as _add_spatial_impulse_response_numba,
)
from openfield.physics.flat_facet_kernel import prepare_flat_facets as _prepare_flat_facets_numba
from openfield.responses import TimeResponse

FIELDII_LEADING_PAD_SAMPLES = 1
FIELDII_TRAILING_PAD_SAMPLES = 5
GAUSS_LEGENDRE_ORDER = 32
_GAUSS_LEGENDRE_NODES_WEIGHTS: tuple[np.ndarray, np.ndarray] | None = None


@dataclass(frozen=True)
class _PackedAperture:
    centers: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    physical_indices: np.ndarray
    element_has_vertices: np.ndarray
    element_facet_start: np.ndarray
    element_facet_count: np.ndarray
    facet_vertices: np.ndarray
    facet_vertex_counts: np.ndarray
    facet_normals: np.ndarray


@dataclass(frozen=True)
class _PreparedFacetArrays:
    ok: np.ndarray
    normals: np.ndarray
    centroids: np.ndarray
    distance_to_plane: np.ndarray
    critical_radii: np.ndarray
    critical_counts: np.ndarray
    normal_angles: np.ndarray
    thresholds: np.ndarray
    angular_counts: np.ndarray
    origin_inside: np.ndarray
    support_start: np.ndarray
    support_end: np.ndarray


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

    packed = _pack_aperture(aperture)
    prepared = _prepare_facets(points, packed, sound_speed)

    start_time, sample_count = _fieldii_like_time_axis(
        aperture=aperture,
        points=points,
        delays=delays,
        subelement_delays=subelement_delays,
        packed=packed,
        prepared=prepared,
        sound_speed=sound_speed,
        sampling_frequency=fs,
        leading_pad_samples=_leading_pad_samples(aperture),
        trailing_pad_samples=_trailing_pad_samples(aperture, subelement_delays),
    )
    samples = np.zeros((sample_count, points.shape[0]), dtype=np.float64)
    nodes, weights = _gauss_legendre_nodes_weights()

    _add_spatial_impulse_response_numba(
        samples,
        points,
        packed.centers,
        packed.normals,
        packed.areas,
        packed.physical_indices,
        packed.element_has_vertices,
        packed.element_facet_start,
        packed.element_facet_count,
        delays,
        subelement_delays,
        apodization,
        subelement_apodization,
        prepared.normals,
        prepared.centroids,
        prepared.distance_to_plane,
        prepared.critical_radii,
        prepared.critical_counts,
        prepared.normal_angles,
        prepared.thresholds,
        prepared.angular_counts,
        prepared.origin_inside,
        prepared.support_start,
        prepared.support_end,
        start_time,
        sound_speed,
        fs,
        nodes,
        weights,
        _has_soft_baffle(aperture.baffle),
    )

    return TimeResponse(
        samples=samples,
        sampling_frequency=fs,
        start_time=start_time,
    )


def _pack_aperture(aperture) -> _PackedAperture:
    element_count = len(aperture.elements)
    centers = aperture.centers
    normals = aperture.normals
    areas = aperture.areas
    physical_indices = aperture.physical_indices
    element_has_vertices = np.zeros(element_count, dtype=np.bool_)
    element_facet_start = np.zeros(element_count, dtype=np.int64)
    element_facet_count = np.zeros(element_count, dtype=np.int64)
    facet_vertices = []
    facet_vertex_counts = []
    facet_normals = []

    for element_index, element in enumerate(aperture.elements):
        if element.vertices is None:
            continue
        element_has_vertices[element_index] = True
        element_facet_start[element_index] = len(facet_vertices)
        facets = _polygon_facets(element)
        element_facet_count[element_index] = len(facets)
        for vertices, normal in facets:
            facet_vertices.append(np.asarray(vertices, dtype=np.float64))
            facet_vertex_counts.append(len(vertices))
            facet_normals.append(np.asarray(normal, dtype=np.float64))

    max_vertices = max(facet_vertex_counts, default=3)
    packed_vertices = np.zeros((len(facet_vertices), max_vertices, 3), dtype=np.float64)
    for facet_index, vertices in enumerate(facet_vertices):
        packed_vertices[facet_index, : len(vertices), :] = vertices

    return _PackedAperture(
        centers=centers,
        normals=normals,
        areas=areas,
        physical_indices=physical_indices,
        element_has_vertices=element_has_vertices,
        element_facet_start=element_facet_start,
        element_facet_count=element_facet_count,
        facet_vertices=packed_vertices,
        facet_vertex_counts=np.asarray(facet_vertex_counts, dtype=np.int64),
        facet_normals=np.asarray(facet_normals, dtype=np.float64).reshape((-1, 3)),
    )


def _prepare_facets(points: np.ndarray, packed: _PackedAperture, sound_speed: float) -> _PreparedFacetArrays:
    if packed.facet_vertices.shape[0] == 0:
        point_count = points.shape[0]
        return _PreparedFacetArrays(
            ok=np.empty((point_count, 0), dtype=np.bool_),
            normals=np.empty((point_count, 0, 3), dtype=np.float64),
            centroids=np.empty((point_count, 0, 3), dtype=np.float64),
            distance_to_plane=np.empty((point_count, 0), dtype=np.float64),
            critical_radii=np.empty((point_count, 0, 7), dtype=np.float64),
            critical_counts=np.empty((point_count, 0), dtype=np.int64),
            normal_angles=np.empty((point_count, 0, 3), dtype=np.float64),
            thresholds=np.empty((point_count, 0, 3), dtype=np.float64),
            angular_counts=np.empty((point_count, 0), dtype=np.int64),
            origin_inside=np.empty((point_count, 0), dtype=np.bool_),
            support_start=np.empty((point_count, 0), dtype=np.float64),
            support_end=np.empty((point_count, 0), dtype=np.float64),
        )

    (
        ok,
        normals,
        centroids,
        distance_to_plane,
        critical_radii,
        critical_counts,
        normal_angles,
        thresholds,
        angular_counts,
        origin_inside,
        support_start,
        support_end,
    ) = _prepare_flat_facets_numba(
        points,
        packed.facet_vertices,
        packed.facet_vertex_counts,
        packed.facet_normals,
        sound_speed,
    )
    if not bool(np.all(ok)):
        raise ValueError("flat-facet preparation failed for a degenerate polygon")
    if bool(np.any(angular_counts <= 0)):
        raise ValueError("flat-facet integration requires convex, non-degenerate polygon facets")

    return _PreparedFacetArrays(
        ok=ok,
        normals=normals,
        centroids=centroids,
        distance_to_plane=distance_to_plane,
        critical_radii=critical_radii,
        critical_counts=critical_counts,
        normal_angles=normal_angles,
        thresholds=thresholds,
        angular_counts=angular_counts,
        origin_inside=origin_inside,
        support_start=support_start,
        support_end=support_end,
    )


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


def _gauss_legendre_nodes_weights() -> tuple[np.ndarray, np.ndarray]:
    global _GAUSS_LEGENDRE_NODES_WEIGHTS
    if _GAUSS_LEGENDRE_NODES_WEIGHTS is None:
        _GAUSS_LEGENDRE_NODES_WEIGHTS = np.polynomial.legendre.leggauss(GAUSS_LEGENDRE_ORDER)
    return _GAUSS_LEGENDRE_NODES_WEIGHTS


def _fieldii_like_time_axis(
    *,
    aperture,
    points: np.ndarray,
    delays: np.ndarray,
    subelement_delays: np.ndarray,
    packed: _PackedAperture,
    prepared: _PreparedFacetArrays,
    sound_speed: float,
    sampling_frequency: float,
    leading_pad_samples: int = FIELDII_LEADING_PAD_SAMPLES,
    trailing_pad_samples: int = FIELDII_TRAILING_PAD_SAMPLES,
) -> tuple[float, int]:
    support_times = []
    for point_index, point in enumerate(points):
        for element_index, element in enumerate(aperture.elements):
            offset = delays[element.physical_index] + subelement_delays[element_index]
            if packed.element_has_vertices[element_index]:
                start = packed.element_facet_start[element_index]
                end = start + packed.element_facet_count[element_index]
                for facet_index in range(start, end):
                    support_times.append(prepared.support_start[point_index, facet_index] + offset)
                    support_times.append(prepared.support_end[point_index, facet_index] + offset)
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


def _has_soft_baffle(baffle) -> bool:
    return bool(baffle is not None and baffle.kind != "rigid")
