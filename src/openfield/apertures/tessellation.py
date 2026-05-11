from __future__ import annotations

from collections.abc import Iterable
import math

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
    for iy in range(sub_y):
        local_y = -height / 2 + (iy + 0.5) * dy
        for ix in range(sub_x):
            local_x = -width / 2 + (ix + 0.5) * dx
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

    for iy in range(sub_y):
        local_y = -height / 2 + (iy + 0.5) * dy
        for ix in range(sub_x):
            local_x = -width / 2 + (ix + 0.5) * dx
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


def adaptive_rectangular_grid(
    *,
    surface,
    u_range: tuple[float, float],
    v_range: tuple[float, float],
    physical_index: int,
    start_subelement_index: int = 0,
    min_subdivisions: tuple[int, int] = (1, 1),
    max_edge_length: float | None = None,
    max_sagitta: float | None = None,
    max_normal_angle: float | None = None,
    max_depth: int = 8,
    expected_normal=(0.0, 0.0, 1.0),
) -> tuple[SubElement, ...]:
    """Adaptively tessellate a rectangular parameter domain on a surface."""

    if u_range[1] <= u_range[0] or v_range[1] <= v_range[0]:
        raise ValueError("parameter ranges must be increasing")
    sub_u, sub_v = min_subdivisions
    if sub_u < 1 or sub_v < 1:
        raise ValueError("min_subdivisions must be positive")
    if max_edge_length is not None and max_edge_length <= 0:
        raise ValueError("max_edge_length must be positive")
    if max_sagitta is not None and max_sagitta < 0:
        raise ValueError("max_sagitta must be non-negative")
    if max_normal_angle is not None and max_normal_angle < 0:
        raise ValueError("max_normal_angle must be non-negative")
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")

    expected_normal = _unit(expected_normal, "expected_normal")
    u_edges = np.linspace(u_range[0], u_range[1], sub_u + 1, dtype=np.float64)
    v_edges = np.linspace(v_range[0], v_range[1], sub_v + 1, dtype=np.float64)
    elements: list[SubElement] = []

    for iv in range(sub_v):
        for iu in range(sub_u):
            _refine_rectangular_patch(
                surface=surface,
                u0=float(u_edges[iu]),
                u1=float(u_edges[iu + 1]),
                v0=float(v_edges[iv]),
                v1=float(v_edges[iv + 1]),
                depth=0,
                max_depth=max_depth,
                max_edge_length=max_edge_length,
                max_sagitta=max_sagitta,
                max_normal_angle=max_normal_angle,
                expected_normal=expected_normal,
                physical_index=physical_index,
                elements=elements,
            )

    return tuple(
        SubElement(
            center=element.center,
            normal=element.normal,
            area=element.area,
            physical_index=element.physical_index,
            subelement_index=start_subelement_index + index,
            vertices=element.vertices,
        )
        for index, element in enumerate(elements)
    )


def _refine_rectangular_patch(
    *,
    surface,
    u0: float,
    u1: float,
    v0: float,
    v1: float,
    depth: int,
    max_depth: int,
    max_edge_length: float | None,
    max_sagitta: float | None,
    max_normal_angle: float | None,
    expected_normal: np.ndarray,
    physical_index: int,
    elements: list[SubElement],
) -> None:
    vertices = np.asarray(
        [
            surface(u0, v0),
            surface(u1, v0),
            surface(u1, v1),
            surface(u0, v1),
        ],
        dtype=np.float64,
    )
    center = np.asarray(surface(0.5 * (u0 + u1), 0.5 * (v0 + v1)), dtype=np.float64)
    split_u, split_v = _rectangular_patch_split_axes(
        surface,
        u0,
        u1,
        v0,
        v1,
        vertices,
        center,
        max_edge_length=max_edge_length,
        max_sagitta=max_sagitta,
        max_normal_angle=max_normal_angle,
        expected_normal=expected_normal,
    )
    if depth < max_depth and (split_u or split_v):
        um = 0.5 * (u0 + u1)
        vm = 0.5 * (v0 + v1)
        if split_u and split_v:
            children = (
                (u0, um, v0, vm),
                (um, u1, v0, vm),
                (um, u1, vm, v1),
                (u0, um, vm, v1),
            )
        elif split_u:
            children = (
                (u0, um, v0, v1),
                (um, u1, v0, v1),
            )
        else:
            children = (
                (u0, u1, v0, vm),
                (u0, u1, vm, v1),
            )
        for child_u0, child_u1, child_v0, child_v1 in children:
            _refine_rectangular_patch(
                surface=surface,
                u0=child_u0,
                u1=child_u1,
                v0=child_v0,
                v1=child_v1,
                depth=depth + 1,
                max_depth=max_depth,
                max_edge_length=max_edge_length,
                max_sagitta=max_sagitta,
                max_normal_angle=max_normal_angle,
                expected_normal=expected_normal,
                physical_index=physical_index,
                elements=elements,
            )
        return

    area, centroid, normal = polygon_area_centroid_normal(vertices)
    if np.dot(normal, expected_normal) < 0.0:
        vertices = vertices[::-1].copy()
        normal = -normal
    elements.append(
        SubElement(
            center=centroid,
            normal=normal,
            area=area,
            physical_index=physical_index,
            subelement_index=len(elements),
            vertices=vertices,
        )
    )


def _rectangular_patch_split_axes(
    surface,
    u0: float,
    u1: float,
    v0: float,
    v1: float,
    vertices: np.ndarray,
    center: np.ndarray,
    *,
    max_edge_length: float | None,
    max_sagitta: float | None,
    max_normal_angle: float | None,
    expected_normal: np.ndarray,
) -> tuple[bool, bool]:
    split_u = False
    split_v = False
    u_edge_length = 0.5 * (
        float(np.linalg.norm(vertices[1] - vertices[0]))
        + float(np.linalg.norm(vertices[2] - vertices[3]))
    )
    v_edge_length = 0.5 * (
        float(np.linalg.norm(vertices[3] - vertices[0]))
        + float(np.linalg.norm(vertices[2] - vertices[1]))
    )

    if max_edge_length is not None:
        split_u = split_u or u_edge_length > max_edge_length
        split_v = split_v or v_edge_length > max_edge_length

    um = 0.5 * (u0 + u1)
    vm = 0.5 * (v0 + v1)
    bottom_mid = np.asarray(surface(um, v0), dtype=np.float64)
    top_mid = np.asarray(surface(um, v1), dtype=np.float64)
    left_mid = np.asarray(surface(u0, vm), dtype=np.float64)
    right_mid = np.asarray(surface(u1, vm), dtype=np.float64)
    u_sagitta = max(
        float(np.linalg.norm(bottom_mid - 0.5 * (vertices[0] + vertices[1]))),
        float(np.linalg.norm(top_mid - 0.5 * (vertices[3] + vertices[2]))),
    )
    v_sagitta = max(
        float(np.linalg.norm(left_mid - 0.5 * (vertices[0] + vertices[3]))),
        float(np.linalg.norm(right_mid - 0.5 * (vertices[1] + vertices[2]))),
    )

    if max_sagitta is not None:
        bilinear_center = 0.25 * np.sum(vertices, axis=0)
        center_sagitta = float(np.linalg.norm(center - bilinear_center))
        split_u = split_u or u_sagitta > max_sagitta
        split_v = split_v or v_sagitta > max_sagitta
        if center_sagitta > max_sagitta and u_sagitta <= max_sagitta and v_sagitta <= max_sagitta:
            if u_edge_length >= v_edge_length:
                split_u = True
            else:
                split_v = True

    if max_normal_angle is not None:
        _, patch_normal = quadrilateral_area_normal(vertices)
        if np.dot(patch_normal, expected_normal) < 0.0:
            patch_normal = -patch_normal
        cos_limit = math.cos(max_normal_angle)
        normal_exceeded = False
        for index in range(4):
            triangle = np.asarray(
                [
                    vertices[index],
                    vertices[(index + 1) % 4],
                    center,
                ],
                dtype=np.float64,
            )
            _, local_normal = triangle_area_normal(triangle)
            if np.dot(local_normal, patch_normal) < 0.0:
                local_normal = -local_normal
            if float(np.dot(local_normal, patch_normal)) < cos_limit:
                normal_exceeded = True
                break
        if normal_exceeded:
            if u_sagitta > v_sagitta:
                split_u = True
            elif v_sagitta > u_sagitta:
                split_v = True
            elif u_edge_length >= v_edge_length:
                split_u = True
            else:
                split_v = True

    return split_u, split_v


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


def circular_polar_grid(
    *,
    radius: float,
    element_size: float,
    physical_index: int = 0,
    start_subelement_index: int = 0,
    z: float = 0.0,
    min_angular_segments: int = 8,
) -> tuple[SubElement, ...]:
    """Approximate a flat circular aperture with boundary-fitted polar sectors."""

    if radius <= 0:
        raise ValueError("radius must be positive")
    if element_size <= 0:
        raise ValueError("element_size must be positive")
    if min_angular_segments < 3:
        raise ValueError("min_angular_segments must be at least 3")

    radial_segments = int(np.ceil(radius / element_size))
    radial_edges = np.linspace(0.0, radius, radial_segments + 1, dtype=np.float64)
    elements: list[SubElement] = []
    index = start_subelement_index
    normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    for radial_index in range(radial_segments):
        inner_radius = float(radial_edges[radial_index])
        outer_radius = float(radial_edges[radial_index + 1])
        angular_segments = max(
            min_angular_segments,
            int(np.ceil(2.0 * np.pi * outer_radius / element_size)),
        )
        for angular_index in range(angular_segments):
            theta0 = 2.0 * np.pi * angular_index / angular_segments
            theta1 = 2.0 * np.pi * (angular_index + 1) / angular_segments
            vertices = _circular_polar_vertices(inner_radius, outer_radius, theta0, theta1, z)
            area, center, element_normal = polygon_area_centroid_normal(vertices)
            if np.dot(element_normal, normal) < 0.0:
                vertices = vertices[::-1].copy()
                element_normal = -element_normal
            elements.append(
                SubElement(
                    center=center,
                    normal=element_normal,
                    area=area,
                    physical_index=physical_index,
                    subelement_index=index,
                    vertices=vertices,
                )
            )
            index += 1

    return tuple(elements)


def _circular_polar_vertices(
    inner_radius: float,
    outer_radius: float,
    theta0: float,
    theta1: float,
    z: float,
) -> np.ndarray:
    if inner_radius == 0.0:
        vertices = [
            [0.0, 0.0, z],
            [outer_radius * np.cos(theta0), outer_radius * np.sin(theta0), z],
            [outer_radius * np.cos(theta1), outer_radius * np.sin(theta1), z],
        ]
    else:
        vertices = [
            [inner_radius * np.cos(theta0), inner_radius * np.sin(theta0), z],
            [outer_radius * np.cos(theta0), outer_radius * np.sin(theta0), z],
            [outer_radius * np.cos(theta1), outer_radius * np.sin(theta1), z],
            [inner_radius * np.cos(theta1), inner_radius * np.sin(theta1), z],
        ]
    return np.asarray(vertices, dtype=np.float64)


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
