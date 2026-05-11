from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
import math

import numpy as np

from .base import Aperture
from .elements import SubElement
from .tessellation import (
    circular_planar_grid,
    circular_polar_grid,
    polygon_area_centroid_normal,
    quadrilateral_area_normal,
    triangle_area_normal,
)


class Piston(Aperture):
    """Flat circular piston aperture."""

    def __init__(
        self,
        radius: float,
        element_size: float,
        *,
        tessellation: str = "cartesian",
        min_angular_segments: int = 8,
    ):
        if min_angular_segments < 3:
            raise ValueError("min_angular_segments must be at least 3")
        tessellation = tessellation.lower()
        if tessellation == "adaptive":
            tessellation = "polar"
        if tessellation == "cartesian":
            elements = circular_planar_grid(radius=radius, element_size=element_size)
        elif tessellation == "polar":
            elements = circular_polar_grid(
                radius=radius,
                element_size=element_size,
                min_angular_segments=min_angular_segments,
            )
        else:
            raise ValueError("tessellation must be 'cartesian', 'polar', or 'adaptive'")
        super().__init__(
            elements=elements,
            physical_element_count=1,
            name="piston",
            metadata={
                "radius": radius,
                "element_size": element_size,
                "tessellation": tessellation,
                "min_angular_segments": min_angular_segments,
            },
        )

    @classmethod
    def adaptive(
        cls,
        radius: float,
        element_size: float,
        *,
        min_angular_segments: int = 8,
    ) -> "Piston":
        """Build a boundary-fitted polar mesh for a flat circular piston."""

        return cls(
            radius=radius,
            element_size=element_size,
            tessellation="polar",
            min_angular_segments=min_angular_segments,
        )


class ConcavePiston(Aperture):
    """Spherically focused circular piston aperture.

    ``tessellation="cartesian"`` reproduces Field II's square subelement
    layout. ``tessellation="polar"`` fits the circular boundary and uses fewer
    facets near the center of the aperture.
    """

    def __init__(
        self,
        radius: float,
        focal_radius: float,
        element_size: float,
        *,
        tessellation: str = "cartesian",
        min_angular_segments: int = 8,
    ):
        if radius <= 0:
            raise ValueError("radius must be positive")
        if focal_radius <= 0:
            raise ValueError("focal_radius must be positive")
        if radius >= focal_radius:
            raise ValueError("radius must be smaller than focal_radius")
        if element_size <= 0:
            raise ValueError("element_size must be positive")
        if min_angular_segments < 3:
            raise ValueError("min_angular_segments must be at least 3")

        tessellation = tessellation.lower()
        if tessellation == "adaptive":
            tessellation = "polar"
        if tessellation == "cartesian":
            elements = _concave_cartesian_grid(radius, focal_radius, element_size)
        elif tessellation == "polar":
            elements = _concave_polar_grid(
                radius,
                focal_radius,
                element_size,
                min_angular_segments=min_angular_segments,
            )
        else:
            raise ValueError("tessellation must be 'cartesian', 'polar', or 'adaptive'")

        super().__init__(
            elements=elements,
            physical_element_count=1,
            name="concave_piston",
            metadata={
                "radius": radius,
                "focal_radius": focal_radius,
                "element_size": element_size,
                "tessellation": tessellation,
                "min_angular_segments": min_angular_segments,
                "fieldii_leading_pad_samples": 0,
                "fieldii_trailing_pad_samples": 4,
            },
        )

    @classmethod
    def adaptive(
        cls,
        radius: float,
        focal_radius: float,
        element_size: float,
        *,
        min_angular_segments: int = 8,
    ) -> "ConcavePiston":
        """Build a boundary-fitted polar mesh for a concave piston."""

        return cls(
            radius=radius,
            focal_radius=focal_radius,
            element_size=element_size,
            tessellation="polar",
            min_angular_segments=min_angular_segments,
        )


def _concave_surface_point(x: float, y: float, focal_radius: float) -> np.ndarray:
    radicand = focal_radius * focal_radius - x * x - y * y
    if radicand < 0.0:
        if radicand < -1e-12 * focal_radius * focal_radius:
            raise ValueError("surface point lies outside the concave sphere")
        radicand = 0.0
    z = focal_radius - np.sqrt(radicand)
    return np.array([x, y, z], dtype=np.float64)


def _concave_cartesian_grid(
    radius: float,
    focal_radius: float,
    element_size: float,
) -> tuple[SubElement, ...]:
    elements: list[SubElement] = []
    steps = int(np.ceil(2 * radius / element_size))
    first = -steps * element_size / 2 + element_size / 2

    index = 0
    for iy in range(steps):
        y = first + iy * element_size
        for ix in range(steps):
            x = first + ix * element_size
            if x * x + y * y > radius * radius:
                continue
            center = _concave_surface_point(x, y, focal_radius)
            x0 = x - element_size / 2.0
            x1 = x + element_size / 2.0
            y0 = y - element_size / 2.0
            y1 = y + element_size / 2.0
            vertices = np.array(
                [
                    _concave_surface_point(x0, y0, focal_radius),
                    _concave_surface_point(x1, y0, focal_radius),
                    _concave_surface_point(x1, y1, focal_radius),
                    _concave_surface_point(x0, y1, focal_radius),
                ],
                dtype=np.float64,
            )
            slope_xz = (vertices[1, 2] - vertices[0, 2]) / (vertices[1, 0] - vertices[0, 0])
            slope_yz = (vertices[3, 2] - vertices[0, 2]) / (vertices[3, 1] - vertices[0, 1])
            normal = np.array([-slope_xz, slope_yz, 1.0], dtype=np.float64)
            normal /= np.linalg.norm(normal)
            area = float(np.linalg.norm(vertices[1] - vertices[0])) * float(
                np.linalg.norm(vertices[3] - vertices[0])
            )
            elements.append(
                SubElement(
                    center=center,
                    normal=normal,
                    area=area,
                    physical_index=0,
                    subelement_index=index,
                    vertices=vertices,
                )
            )
            index += 1
    return tuple(elements)


def _concave_polar_grid(
    radius: float,
    focal_radius: float,
    element_size: float,
    *,
    min_angular_segments: int,
) -> tuple[SubElement, ...]:
    elements: list[SubElement] = []
    radial_segments = int(math.ceil(radius / element_size))
    radial_edges = np.linspace(0.0, radius, radial_segments + 1, dtype=np.float64)
    index = 0

    for radial_index in range(radial_segments):
        inner_radius = float(radial_edges[radial_index])
        outer_radius = float(radial_edges[radial_index + 1])
        angular_segments = max(
            min_angular_segments,
            int(math.ceil(2.0 * math.pi * outer_radius / element_size)),
        )
        for angular_index in range(angular_segments):
            theta0 = 2.0 * math.pi * angular_index / angular_segments
            theta1 = 2.0 * math.pi * (angular_index + 1) / angular_segments
            vertices = _concave_polar_vertices(inner_radius, outer_radius, theta0, theta1, focal_radius)
            area, center, normal = polygon_area_centroid_normal(vertices)
            if normal[2] < 0.0:
                vertices = vertices[::-1].copy()
                normal = -normal
            elements.append(
                SubElement(
                    center=center,
                    normal=normal,
                    area=area,
                    physical_index=0,
                    subelement_index=index,
                    vertices=vertices,
                )
            )
            index += 1

    return tuple(elements)


def _concave_polar_vertices(
    inner_radius: float,
    outer_radius: float,
    theta0: float,
    theta1: float,
    focal_radius: float,
) -> np.ndarray:
    if inner_radius == 0.0:
        points = [
            _concave_surface_point(0.0, 0.0, focal_radius),
            _concave_surface_point(
                outer_radius * math.cos(theta0),
                outer_radius * math.sin(theta0),
                focal_radius,
            ),
            _concave_surface_point(
                outer_radius * math.cos(theta1),
                outer_radius * math.sin(theta1),
                focal_radius,
            ),
        ]
    else:
        points = [
            _concave_surface_point(
                inner_radius * math.cos(theta0),
                inner_radius * math.sin(theta0),
                focal_radius,
            ),
            _concave_surface_point(
                outer_radius * math.cos(theta0),
                outer_radius * math.sin(theta0),
                focal_radius,
            ),
            _concave_surface_point(
                outer_radius * math.cos(theta1),
                outer_radius * math.sin(theta1),
                focal_radius,
            ),
            _concave_surface_point(
                inner_radius * math.cos(theta1),
                inner_radius * math.sin(theta1),
                focal_radius,
            ),
        ]
    return np.asarray(points, dtype=np.float64)


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

    @classmethod
    def from_fieldii_rectangles(cls, rect, *, focus=None) -> "RectangleAperture":
        """Build from Field II's ``xdc_rectangles`` matrix layout."""

        rect = np.asarray(rect, dtype=np.float64)
        if rect.ndim != 2 or rect.shape[1] != 19:
            raise ValueError("rect must have shape (n_rectangles, 19)")
        physical_indices = _normalize_index_column(rect[:, 0])
        vertices = rect[:, 1:13].reshape((-1, 4, 3))
        aperture = cls(vertices=vertices, physical_indices=physical_indices)
        elements = []
        for index, rectangle in enumerate(vertices):
            _, normal = quadrilateral_area_normal(rectangle)
            elements.append(
                SubElement(
                    center=rect[index, 16:19],
                    normal=normal,
                    area=float(rect[index, 14] * rect[index, 15]),
                    physical_index=int(physical_indices[index]),
                    subelement_index=index,
                    vertices=rectangle,
                )
            )
        metadata = dict(aperture.metadata)
        metadata["fieldii_rectangles"] = rect.copy()
        metadata["fieldii_trailing_pad_samples"] = 4
        aperture = aperture._replace(elements=tuple(elements), metadata=metadata)
        if focus is not None:
            aperture = aperture.focused_at(focus)
        return aperture

    def to_fieldii_rectangles(self) -> np.ndarray:
        """Export a Field II-compatible ``xdc_rectangles`` matrix."""

        rows = []
        apodization = _apodization_values(self)
        for element in self.elements:
            if element.vertices is None or element.vertices.shape != (4, 3):
                raise ValueError("all subelements must be quadrilaterals")
            vertices = np.asarray(element.vertices, dtype=np.float64)
            width = float(np.linalg.norm(vertices[1] - vertices[0]))
            height = float(np.linalg.norm(vertices[3] - vertices[0]))
            rows.append(
                [
                    element.physical_index + 1,
                    *vertices.reshape(-1).tolist(),
                    float(apodization[element.physical_index]),
                    width,
                    height,
                    *element.center.tolist(),
                ]
            )
        return np.asarray(rows, dtype=np.float64)


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

    @classmethod
    def from_fieldii_triangles(cls, data, *, focus=None) -> "TriangleAperture":
        """Build from Field II's ``xdc_triangles`` matrix layout."""

        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != 11:
            raise ValueError("data must have shape (n_triangles, 11)")
        physical_indices = _normalize_index_column(data[:, 0])
        vertices = data[:, 1:10].reshape((-1, 3, 3))
        aperture = cls(vertices=vertices, physical_indices=physical_indices)
        metadata = dict(aperture.metadata)
        metadata["fieldii_triangles"] = data.copy()
        metadata["fieldii_trailing_pad_samples"] = 2
        aperture = aperture._replace(metadata=metadata)
        if focus is not None:
            aperture = aperture.focused_at(focus)
        return aperture

    def to_fieldii_triangles(self) -> np.ndarray:
        """Export a Field II-compatible ``xdc_triangles`` matrix."""

        rows = []
        apodization = _apodization_values(self)
        for element in self.elements:
            if element.vertices is None or element.vertices.shape != (3, 3):
                raise ValueError("all subelements must be triangles")
            rows.append(
                [
                    element.physical_index + 1,
                    *np.asarray(element.vertices, dtype=np.float64).reshape(-1).tolist(),
                    float(apodization[element.physical_index]),
                ]
            )
        return np.asarray(rows, dtype=np.float64)


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

    @classmethod
    def from_fieldii_lines(
        cls,
        lines,
        *,
        centers=None,
        focus=None,
        bounding_extent: float | None = None,
    ) -> "LineBoundedAperture":
        """Build from Field II's ``xdc_lines`` matrix layout."""

        aperture = cls(lines=lines, centers=centers, bounding_extent=bounding_extent)
        metadata = dict(aperture.metadata)
        metadata["fieldii_trailing_pad_samples"] = 3
        aperture = aperture._replace(metadata=metadata)
        if focus is not None:
            aperture = aperture.focused_at(focus)
        return aperture

    def to_fieldii_lines(self) -> np.ndarray:
        """Export a Field II-compatible ``xdc_lines`` matrix."""

        lines = np.asarray(self.metadata["lines"], dtype=np.float64).copy()
        lines[:, 0:2] += 1
        return lines


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


def _apodization_values(aperture: Aperture) -> np.ndarray:
    if aperture.apodization is None:
        return np.ones(aperture.physical_element_count, dtype=np.float64)
    values = np.asarray(aperture.apodization.at_time(0.0), dtype=np.float64)
    if values.shape != (aperture.physical_element_count,):
        raise ValueError("apodization length must match physical element count")
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
