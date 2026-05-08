from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from .base import Aperture
from .elements import SubElement
from .tessellation import (
    circular_planar_grid,
    polygon_area_centroid_normal,
    quadrilateral_area_normal,
    triangle_area_normal,
)


class Piston(Aperture):
    """Flat circular piston aperture."""

    def __init__(self, radius: float, element_size: float):
        elements = circular_planar_grid(radius=radius, element_size=element_size)
        super().__init__(
            elements=elements,
            physical_element_count=1,
            name="piston",
            metadata={"radius": radius, "element_size": element_size},
        )


class ConcavePiston(Aperture):
    """Spherically focused circular piston aperture."""

    def __init__(self, radius: float, focal_radius: float, element_size: float):
        if radius <= 0:
            raise ValueError("radius must be positive")
        if focal_radius <= 0:
            raise ValueError("focal_radius must be positive")
        if radius >= focal_radius:
            raise ValueError("radius must be smaller than focal_radius")
        if element_size <= 0:
            raise ValueError("element_size must be positive")

        planar = circular_planar_grid(radius=radius, element_size=element_size)
        focus = np.array([0.0, 0.0, focal_radius], dtype=np.float64)
        elements: list[SubElement] = []

        for index, element in enumerate(planar):
            x, y, _ = element.center
            radial_sq = x * x + y * y
            z = focal_radius - np.sqrt(focal_radius * focal_radius - radial_sq)
            center = np.array([x, y, z], dtype=np.float64)
            normal = focus - center
            normal /= np.linalg.norm(normal)
            projected_area_factor = max(float(normal[2]), np.finfo(float).eps)
            vertices = []
            for vertex in element.vertices:
                vx, vy, _ = vertex
                vertex_radial_sq = vx * vx + vy * vy
                if vertex_radial_sq >= focal_radius * focal_radius:
                    continue
                vz = focal_radius - np.sqrt(focal_radius * focal_radius - vertex_radial_sq)
                vertices.append([vx, vy, vz])
            elements.append(
                SubElement(
                    center=center,
                    normal=normal,
                    area=element.area / projected_area_factor,
                    physical_index=0,
                    subelement_index=index,
                    vertices=np.asarray(vertices, dtype=np.float64) if len(vertices) >= 3 else None,
                )
            )

        super().__init__(
            elements=tuple(elements),
            physical_element_count=1,
            name="concave_piston",
            metadata={
                "radius": radius,
                "focal_radius": focal_radius,
                "element_size": element_size,
            },
        )


class RectangleAperture(Aperture):
    """Aperture defined directly by rectangular surface patches."""

    def __init__(self, vertices, physical_indices: Sequence[int] | None = None):
        vertices = np.asarray(vertices, dtype=np.float64)
        if vertices.ndim != 3 or vertices.shape[1:] != (4, 3):
            raise ValueError("vertices must have shape (n_rectangles, 4, 3)")
        physical_indices = _physical_indices(physical_indices, len(vertices))
        elements = []
        for index, rectangle in enumerate(vertices):
            area, normal = quadrilateral_area_normal(rectangle)
            center = np.mean(rectangle, axis=0)
            elements.append(
                SubElement(
                    center=center,
                    normal=normal,
                    area=area,
                    physical_index=int(physical_indices[index]),
                    subelement_index=index,
                    vertices=rectangle,
                )
            )

        super().__init__(
            elements=tuple(elements),
            physical_element_count=int(np.max(physical_indices)) + 1,
            name="rectangle_aperture",
            metadata={"vertices": vertices.copy(), "physical_indices": physical_indices.copy()},
        )


class TriangleAperture(Aperture):
    """Aperture defined directly by triangular surface patches."""

    def __init__(self, vertices, physical_indices: Sequence[int] | None = None):
        vertices = np.asarray(vertices, dtype=np.float64)
        if vertices.ndim != 3 or vertices.shape[1:] != (3, 3):
            raise ValueError("vertices must have shape (n_triangles, 3, 3)")
        physical_indices = _physical_indices(physical_indices, len(vertices))
        elements = []
        for index, triangle in enumerate(vertices):
            area, normal = triangle_area_normal(triangle)
            center = np.mean(triangle, axis=0)
            elements.append(
                SubElement(
                    center=center,
                    normal=normal,
                    area=area,
                    physical_index=int(physical_indices[index]),
                    subelement_index=index,
                    vertices=triangle,
                )
            )

        super().__init__(
            elements=tuple(elements),
            physical_element_count=int(np.max(physical_indices)) + 1,
            name="triangle_aperture",
            metadata={"vertices": vertices.copy(), "physical_indices": physical_indices.copy()},
        )


class LineBoundedAperture(Aperture):
    """Flat aperture defined by half-plane bounded polygons.

    The ``lines`` input follows Field II's line-bounded representation:
    ``physical, subelement, slope, infinity, intersect, above``. Indices may
    be zero- or one-based; one-based indices are converted to zero-based.
    """

    def __init__(self, lines, centers=None, bounding_extent: float | None = None):
        lines = np.asarray(lines, dtype=np.float64)
        if lines.ndim != 2 or lines.shape[1] != 6:
            raise ValueError("lines must have shape (n_lines, 6)")
        if bounding_extent is not None and bounding_extent <= 0:
            raise ValueError("bounding_extent must be positive")

        centers_array = None if centers is None else np.asarray(centers, dtype=np.float64)
        if centers_array is not None and (centers_array.ndim != 2 or centers_array.shape[1] != 3):
            raise ValueError("centers must have shape (n_physical, 3)")

        physical_column = _normalize_index_column(lines[:, 0])
        subelement_column = _normalize_index_column(lines[:, 1])
        line_data = lines.copy()
        line_data[:, 0] = physical_column
        line_data[:, 1] = subelement_column
        extent = _line_bounded_extent(line_data, centers_array, bounding_extent)

        grouped_lines: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
        for row in line_data:
            grouped_lines[(int(row[0]), int(row[1]))].append(row)

        elements = []
        for subelement_index, ((physical_index, _), group) in enumerate(sorted(grouped_lines.items())):
            polygon_xy = _clip_unit_box_with_lines(group, extent)
            if len(polygon_xy) < 3:
                raise ValueError("line constraints produced an empty polygon")
            vertices = np.column_stack(
                [polygon_xy[:, 0], polygon_xy[:, 1], np.zeros(len(polygon_xy), dtype=np.float64)]
            )
            area, center, normal = polygon_area_centroid_normal(vertices)
            elements.append(
                SubElement(
                    center=center,
                    normal=normal,
                    area=area,
                    physical_index=physical_index,
                    subelement_index=subelement_index,
                    vertices=vertices,
                )
            )

        super().__init__(
            elements=tuple(elements),
            physical_element_count=max(element.physical_index for element in elements) + 1,
            name="line_bounded_aperture",
            metadata={"lines": line_data.copy(), "centers": centers_array, "bounding_extent": extent},
        )


def _physical_indices(physical_indices: Sequence[int] | None, count: int) -> np.ndarray:
    if physical_indices is None:
        return np.arange(count, dtype=np.int64)
    physical_indices = np.asarray(physical_indices, dtype=np.int64)
    if physical_indices.shape != (count,):
        raise ValueError("physical_indices must contain one index per patch")
    if np.any(physical_indices < 0):
        raise ValueError("physical_indices must be non-negative")
    return physical_indices


def _normalize_index_column(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.allclose(values, np.round(values)):
        raise ValueError("line indices must be integers")
    values = np.round(values).astype(np.int64)
    if np.min(values) >= 1:
        values = values - 1
    if np.any(values < 0):
        raise ValueError("line indices must be non-negative")
    return values


def _line_bounded_extent(lines: np.ndarray, centers: np.ndarray | None, bounding_extent: float | None) -> float:
    if bounding_extent is not None:
        return float(bounding_extent)
    candidates = [1.0]
    finite_values = lines[:, 4][np.isfinite(lines[:, 4])]
    if finite_values.size:
        candidates.append(float(np.max(np.abs(finite_values))) * 2 + 1.0)
    if centers is not None:
        candidates.append(float(np.max(np.abs(centers[:, :2]))) * 2 + 1.0)
    return max(candidates)


def _clip_unit_box_with_lines(lines: Sequence[np.ndarray], extent: float) -> np.ndarray:
    polygon = np.array(
        [
            [-extent, -extent],
            [extent, -extent],
            [extent, extent],
            [-extent, extent],
        ],
        dtype=np.float64,
    )
    for line in lines:
        polygon = _clip_polygon_with_half_plane(polygon, line)
        if len(polygon) == 0:
            break
    return polygon


def _clip_polygon_with_half_plane(polygon: np.ndarray, line: np.ndarray) -> np.ndarray:
    if len(polygon) == 0:
        return polygon
    clipped = []
    previous = polygon[-1]
    previous_value = _half_plane_value(previous, line)
    previous_inside = previous_value >= -1e-12
    for current in polygon:
        current_value = _half_plane_value(current, line)
        current_inside = current_value >= -1e-12
        if current_inside != previous_inside:
            clipped.append(_line_segment_intersection(previous, current, previous_value, current_value))
        if current_inside:
            clipped.append(current)
        previous = current
        previous_value = current_value
        previous_inside = current_inside
    return np.asarray(clipped, dtype=np.float64)


def _half_plane_value(point: np.ndarray, line: np.ndarray) -> float:
    _, _, slope, infinity, intersect, above = line
    x, y = point
    above = bool(above)
    if bool(infinity):
        value = intersect - x
    else:
        value = y - (slope * x + intersect)
    return value if above else -value


def _line_segment_intersection(
    start: np.ndarray,
    end: np.ndarray,
    start_value: float,
    end_value: float,
) -> np.ndarray:
    denominator = start_value - end_value
    if abs(denominator) < 1e-15:
        return end
    fraction = start_value / denominator
    return start + fraction * (end - start)
