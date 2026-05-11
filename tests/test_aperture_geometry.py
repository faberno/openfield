import math

import numpy as np
import pytest

from openfield.apertures import (
    ConcavePiston,
    LineBoundedAperture,
    LinearArray,
    Piston,
    RectangleAperture,
    TriangleAperture,
)
from openfield.focusing import Apodization


def test_piston_tessellation_stays_inside_radius():
    aperture = Piston(radius=1.0, element_size=0.1)

    radii = np.linalg.norm(aperture.centers[:, :2], axis=1)

    assert aperture.physical_element_count == 1
    assert len(aperture.elements) > 0
    assert np.all(radii <= 1.0)
    assert np.allclose(aperture.normals, [0.0, 0.0, 1.0])
    assert np.all(aperture.areas > 0.0)


def test_piston_polar_tessellation_fits_circular_boundary():
    aperture = Piston.adaptive(radius=1.0, element_size=0.1)
    vertices = np.concatenate([element.vertices for element in aperture.elements])
    vertex_radii = np.linalg.norm(vertices[:, :2], axis=1)

    assert aperture.metadata["tessellation"] == "polar"
    assert np.max(vertex_radii) <= 1.0 + 1e-12
    assert np.allclose(aperture.normals, [0.0, 0.0, 1.0])
    assert np.sum(aperture.areas) == pytest.approx(math.pi, rel=0.01)


def test_concave_piston_uses_fieldii_slope_normals():
    focal_radius = 2.0
    aperture = ConcavePiston(radius=0.8, focal_radius=focal_radius, element_size=0.1)

    first = aperture.elements[0]
    edge_x = first.vertices[1] - first.vertices[0]
    edge_y = first.vertices[3] - first.vertices[0]
    slope_xz = edge_x[2] / edge_x[0]
    slope_yz = edge_y[2] / edge_y[1]
    expected_first_normal = np.array([-slope_xz, slope_yz, 1.0])
    expected_first_normal /= np.linalg.norm(expected_first_normal)

    assert aperture.physical_element_count == 1
    assert np.all(aperture.centers[:, 2] >= 0.0)
    assert np.allclose(first.normal, expected_first_normal)
    assert np.all(aperture.areas > 0.0)


def test_concave_piston_polar_tessellation_fits_circular_boundary():
    radius = 0.8
    focal_radius = 2.0
    aperture = ConcavePiston(
        radius=radius,
        focal_radius=focal_radius,
        element_size=0.1,
        tessellation="polar",
    )

    vertices = np.concatenate([element.vertices for element in aperture.elements])
    vertex_radii = np.linalg.norm(vertices[:, :2], axis=1)
    cap_height = focal_radius - math.sqrt(focal_radius * focal_radius - radius * radius)
    cap_area = 2.0 * math.pi * focal_radius * cap_height

    assert aperture.metadata["tessellation"] == "polar"
    assert np.max(vertex_radii) <= radius + 1e-12
    assert np.array_equal(aperture.subelement_indices, np.arange(len(aperture.elements)))
    assert all(element.vertices.shape in {(3, 3), (4, 3)} for element in aperture.elements)
    assert np.all(aperture.normals[:, 2] > 0.0)
    assert np.sum(aperture.areas) == pytest.approx(cap_area, rel=0.01)


def test_concave_piston_adaptive_uses_polar_tessellation():
    aperture = ConcavePiston.adaptive(radius=0.8, focal_radius=2.0, element_size=0.1)

    assert aperture.metadata["tessellation"] == "polar"
    assert len(aperture.elements) == len(ConcavePiston(0.8, 2.0, 0.1, tessellation="polar").elements)


def test_linear_array_tessellation_and_indexing():
    aperture = LinearArray(
        elements=4,
        width=0.3,
        height=1.0,
        kerf=0.1,
        subdivisions=(2, 3),
    )

    assert aperture.physical_element_count == 4
    assert len(aperture.elements) == 4 * 2 * 3
    assert np.array_equal(np.unique(aperture.physical_indices), np.arange(4))
    assert np.array_equal(aperture.subelement_indices, np.arange(len(aperture.elements)))
    assert np.allclose(aperture.physical_centers[:, 1:], 0.0)
    assert np.allclose(aperture.physical_centers[:, 0], [-0.6, -0.2, 0.2, 0.6])
    assert all(element.vertices.shape == (4, 3) for element in aperture.elements)


def test_rectangle_aperture_fieldii_roundtrip():
    rect = np.array(
        [
            [
                1,
                -0.5,
                -0.25,
                0.0,
                0.5,
                -0.25,
                0.0,
                0.5,
                0.25,
                0.0,
                -0.5,
                0.25,
                0.0,
                0.75,
                1.0,
                0.5,
                0.0,
                0.0,
                0.0,
            ]
        ],
        dtype=np.float64,
    )

    aperture = RectangleAperture.from_fieldii_rectangles(rect).with_apodization(Apodization([0.75]))
    exported = aperture.to_fieldii_rectangles()

    assert aperture.physical_indices.tolist() == [0]
    assert np.allclose(exported, rect)


def test_rectangle_aperture_uses_fieldii_supplied_centers_and_area():
    rect = np.array(
        [
            [
                1,
                -0.5,
                -0.25,
                0.0,
                0.5,
                -0.25,
                0.1,
                0.5,
                0.25,
                0.1,
                -0.5,
                0.25,
                0.0,
                1.0,
                1.0,
                0.6,
                0.1,
                0.2,
                0.3,
            ]
        ],
        dtype=np.float64,
    )

    aperture = RectangleAperture.from_fieldii_rectangles(rect)

    assert np.allclose(aperture.centers, [[0.1, 0.2, 0.3]])
    assert np.allclose(aperture.areas, [0.6])


def test_triangle_aperture_fieldii_roundtrip():
    data = np.array(
        [[1, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5]],
        dtype=np.float64,
    )

    aperture = TriangleAperture.from_fieldii_triangles(data).with_apodization(Apodization([0.5]))
    exported = aperture.to_fieldii_triangles()

    assert aperture.physical_indices.tolist() == [0]
    assert np.allclose(exported, data)


def test_aperture_to_triangles_preserves_area_and_physical_indices():
    rectangle = RectangleAperture(
        [[[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]]],
        physical_indices=[2],
    )

    triangles = rectangle.to_triangles()

    assert isinstance(triangles, TriangleAperture)
    assert len(triangles.elements) == 2
    assert np.all(triangles.physical_indices == 2)
    assert np.sum(triangles.areas) == pytest.approx(np.sum(rectangle.areas))


def test_line_bounded_aperture_exports_fieldii_one_based_indices():
    lines = np.array(
        [
            [1, 1, 0, 1, -0.5, 0],
            [1, 1, 0, 1, 0.5, 1],
            [1, 1, 0, 0, -0.5, 1],
            [1, 1, 0, 0, 0.5, 0],
        ],
        dtype=np.float64,
    )

    aperture = LineBoundedAperture.from_fieldii_lines(lines, bounding_extent=1.0)

    assert np.allclose(aperture.to_fieldii_lines(), lines)
