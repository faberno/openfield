from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .elements import SubElement


def _unit(vector, name: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError(f"{name} must be non-zero")
    return vector / norm


def rectangular_grid(
    *,
    center,
    width: float,
    height: float,
    subdivisions: tuple[int, int],
    physical_index: int,
    start_subelement_index: int = 0,
    normal=(0.0, 0.0, 1.0),
    tangent_x=(1.0, 0.0, 0.0),
) -> tuple[SubElement, ...]:
    """Tessellate a rectangular physical element into smaller rectangles."""

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    sub_x, sub_y = subdivisions
    if sub_x < 1 or sub_y < 1:
        raise ValueError("subdivisions must be positive")

    center = np.asarray(center, dtype=np.float64)
    if center.shape != (3,):
        raise ValueError("center must be a 3-vector")

    normal = _unit(normal, "normal")
    tangent_x = _unit(tangent_x, "tangent_x")
    tangent_x = tangent_x - np.dot(tangent_x, normal) * normal
    tangent_x = _unit(tangent_x, "tangent_x")
    tangent_y = np.cross(normal, tangent_x)

    dx = width / sub_x
    dy = height / sub_y
    area = dx * dy
    elements: list[SubElement] = []

    index = start_subelement_index
    for ix in range(sub_x):
        local_x = -width / 2 + (ix + 0.5) * dx
        for iy in range(sub_y):
            local_y = -height / 2 + (iy + 0.5) * dy
            element_center = center + local_x * tangent_x + local_y * tangent_y
            vertices = np.array(
                [
                    element_center - dx / 2 * tangent_x - dy / 2 * tangent_y,
                    element_center + dx / 2 * tangent_x - dy / 2 * tangent_y,
                    element_center + dx / 2 * tangent_x + dy / 2 * tangent_y,
                    element_center - dx / 2 * tangent_x + dy / 2 * tangent_y,
                ],
                dtype=np.float64,
            )
            elements.append(
                SubElement(
                    center=element_center,
                    normal=normal,
                    area=area,
                    physical_index=physical_index,
                    subelement_index=index,
                    vertices=vertices,
                )
            )
            index += 1

    return tuple(elements)


def mapped_rectangular_grid(
    *,
    surface,
    width: float,
    height: float,
    subdivisions: tuple[int, int],
    physical_index: int,
    start_subelement_index: int = 0,
    expected_normal=(0.0, 0.0, 1.0),
) -> tuple[SubElement, ...]:
    """Tessellate a rectangular parameter domain mapped onto a surface.

    ``surface`` is called as ``surface(local_x, local_y)`` and must return a
    3-vector. Local coordinates are centered on the physical element.
    """

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    sub_x, sub_y = subdivisions
    if sub_x < 1 or sub_y < 1:
        raise ValueError("subdivisions must be positive")

    expected_normal = _unit(expected_normal, "expected_normal")
    dx = width / sub_x
    dy = height / sub_y
    elements: list[SubElement] = []
    index = start_subelement_index

    for ix in range(sub_x):
        local_x = -width / 2 + (ix + 0.5) * dx
        for iy in range(sub_y):
            local_y = -height / 2 + (iy + 0.5) * dy
            center = np.asarray(surface(local_x, local_y), dtype=np.float64)
            vertices = np.asarray(
                [
                    surface(local_x - dx / 2, local_y - dy / 2),
                    surface(local_x + dx / 2, local_y - dy / 2),
                    surface(local_x + dx / 2, local_y + dy / 2),
                    surface(local_x - dx / 2, local_y + dy / 2),
                ],
                dtype=np.float64,
            )
            area, normal = quadrilateral_area_normal(vertices)
            if np.dot(normal, expected_normal) < 0:
                vertices = vertices[::-1].copy()
                normal = -normal
            elements.append(
                SubElement(
                    center=center,
                    normal=normal,
                    area=area,
                    physical_index=physical_index,
                    subelement_index=index,
                    vertices=vertices,
                )
            )
            index += 1

    return tuple(elements)


def circular_planar_grid(
    *,
    radius: float,
    element_size: float,
    physical_index: int = 0,
    start_subelement_index: int = 0,
    z: float = 0.0,
) -> tuple[SubElement, ...]:
    """Approximate a flat circular aperture with square subelements."""

    if radius <= 0:
        raise ValueError("radius must be positive")
    if element_size <= 0:
        raise ValueError("element_size must be positive")

    steps = int(np.ceil(2 * radius / element_size))
    first = -steps * element_size / 2 + element_size / 2
    elements: list[SubElement] = []
    index = start_subelement_index

    for ix in range(steps):
        x = first + ix * element_size
        for iy in range(steps):
            y = first + iy * element_size
            if x * x + y * y <= radius * radius:
                vertices = np.array(
                    [
                        [x - element_size / 2, y - element_size / 2, z],
                        [x + element_size / 2, y - element_size / 2, z],
                        [x + element_size / 2, y + element_size / 2, z],
                        [x - element_size / 2, y + element_size / 2, z],
                    ],
                    dtype=np.float64,
                )
                elements.append(
                    SubElement(
                        center=np.array([x, y, z], dtype=np.float64),
                        normal=np.array([0.0, 0.0, 1.0], dtype=np.float64),
                        area=element_size * element_size,
                        physical_index=physical_index,
                        subelement_index=index,
                        vertices=vertices,
                    )
                )
                index += 1

    return tuple(elements)


def renumber(elements: Iterable[SubElement]) -> tuple[SubElement, ...]:
    """Return elements with contiguous subelement indices."""

    return tuple(
        SubElement(
            center=element.center,
            normal=element.normal,
            area=element.area,
            physical_index=element.physical_index,
            subelement_index=index,
            vertices=element.vertices,
        )
        for index, element in enumerate(elements)
    )


def triangle_area_normal(vertices: np.ndarray) -> tuple[float, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.shape != (3, 3):
        raise ValueError("triangle vertices must have shape (3, 3)")
    normal_area = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
    norm = float(np.linalg.norm(normal_area))
    if norm == 0:
        raise ValueError("triangle area must be positive")
    return 0.5 * norm, normal_area / norm


def quadrilateral_area_normal(vertices: np.ndarray) -> tuple[float, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.shape != (4, 3):
        raise ValueError("quadrilateral vertices must have shape (4, 3)")
    area_1, normal_1 = triangle_area_normal(vertices[[0, 1, 2]])
    area_2, normal_2 = triangle_area_normal(vertices[[0, 2, 3]])
    normal = normal_1 * area_1 + normal_2 * area_2
    norm = float(np.linalg.norm(normal))
    if norm == 0:
        raise ValueError("quadrilateral area must be positive")
    return area_1 + area_2, normal / norm


def polygon_area_centroid_normal(vertices: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] < 3:
        raise ValueError("polygon vertices must have shape (n, 3) with n >= 3")

    origin = vertices[0]
    weighted_normal = np.zeros(3, dtype=np.float64)
    weighted_centroid = np.zeros(3, dtype=np.float64)
    area = 0.0
    for index in range(1, len(vertices) - 1):
        triangle = np.array([origin, vertices[index], vertices[index + 1]], dtype=np.float64)
        triangle_area, triangle_normal = triangle_area_normal(triangle)
        triangle_centroid = np.mean(triangle, axis=0)
        weighted_normal += triangle_normal * triangle_area
        weighted_centroid += triangle_centroid * triangle_area
        area += triangle_area

    if area == 0:
        raise ValueError("polygon area must be positive")
    normal = weighted_normal / np.linalg.norm(weighted_normal)
    centroid = weighted_centroid / area
    return area, centroid, normal
