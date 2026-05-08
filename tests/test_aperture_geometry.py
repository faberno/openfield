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


def test_concave_piston_normals_point_to_focus():
    focal_radius = 2.0
    aperture = ConcavePiston(radius=0.8, focal_radius=focal_radius, element_size=0.1)

    focus = np.array([0.0, 0.0, focal_radius])
    expected_normals = focus - aperture.centers
    expected_normals = expected_normals / np.linalg.norm(expected_normals, axis=1)[:, None]

    assert aperture.physical_element_count == 1
    assert np.all(aperture.centers[:, 2] >= 0.0)
    assert np.allclose(aperture.normals, expected_normals)
    assert np.all(aperture.areas > 0.0)


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
