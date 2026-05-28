import numpy as np

from openfield.apertures import (
    Array2D,
    ConvexArray,
    ConvexFocusedArray,
    ConvexFocusedMultirowArray,
    FocusedLinearArray,
    FocusedMultirowArray,
    LineBoundedAperture,
    LinearMultirowArray,
    RectangleAperture,
    TriangleAperture,
    TwoDimensionalArray,
)


def test_focused_linear_array_curves_in_elevation():
    aperture = FocusedLinearArray(
        elements=3,
        width=0.2,
        height=0.8,
        kerf=0.05,
        elevation_focus=2.0,
        subdivisions=(1, 4),
        tessellation="fieldii",
    )

    assert aperture.physical_element_count == 3
    assert len(aperture.elements) == 3 * 4
    assert np.max(aperture.centers[:, 2]) > np.min(aperture.centers[:, 2])
    assert np.all(aperture.normals[:, 2] > 0.0)


def test_focused_linear_array_adaptive_tessellation_refines_elevation_curvature():
    fieldii = FocusedLinearArray(
        elements=3,
        width=0.2,
        height=0.8,
        kerf=0.05,
        elevation_focus=2.0,
        tessellation="fieldii",
    )
    adaptive = FocusedLinearArray(
        elements=3,
        width=0.2,
        height=0.8,
        kerf=0.05,
        elevation_focus=2.0,
        tessellation="adaptive",
        max_sagitta=1e-3,
    )

    assert adaptive.metadata["tessellation"] == "adaptive"
    assert "fieldii_focus_centers" not in adaptive.metadata
    assert adaptive.physical_element_count == fieldii.physical_element_count
    assert len(adaptive.elements) > len(fieldii.elements)
    assert np.all(adaptive.normals[:, 2] > 0.0)


def test_linear_multirow_array_indexes_all_rows():
    aperture = LinearMultirowArray(
        elements_x=2,
        width=0.2,
        elements_y=3,
        heights=[0.2, 0.3, 0.4],
        kerf_x=0.05,
        kerf_y=0.02,
        subdivisions=(2, 1),
    )

    assert aperture.physical_element_count == 6
    assert len(aperture.elements) == 2 * 3 * 2
    assert np.array_equal(np.unique(aperture.physical_indices), np.arange(6))
    assert len(np.unique(aperture.physical_centers[:, 1])) == 3


def test_focused_multirow_array_has_curved_rows():
    aperture = FocusedMultirowArray(
        elements_x=2,
        width=0.2,
        elements_y=3,
        heights=[0.2, 0.3, 0.4],
        kerf_x=0.05,
        kerf_y=0.02,
        elevation_focus=2.0,
        subdivisions=(1, 2),
        tessellation="fieldii",
    )

    assert aperture.physical_element_count == 6
    assert np.min(aperture.centers[:, 2]) < 0.0
    assert np.max(aperture.centers[:, 2]) <= 0.0
    assert np.all(aperture.normals[:, 2] > 0.0)


def test_focused_multirow_array_adaptive_tessellation_keeps_physical_indices():
    aperture = FocusedMultirowArray(
        elements_x=2,
        width=0.2,
        elements_y=3,
        heights=[0.2, 0.3, 0.4],
        kerf_x=0.05,
        kerf_y=0.02,
        elevation_focus=2.0,
        tessellation="adaptive",
        max_sagitta=1e-3,
    )

    assert aperture.metadata["tessellation"] == "adaptive"
    assert np.array_equal(np.unique(aperture.physical_indices), np.arange(6))
    assert np.all(aperture.areas > 0.0)


def test_convex_array_fans_normals_in_azimuth():
    aperture = ConvexArray(
        elements=5,
        width=0.2,
        height=0.5,
        kerf=0.02,
        convex_radius=5.0,
        subdivisions=(2, 1),
        tessellation="fieldii",
    )

    assert aperture.physical_element_count == 5
    assert np.min(aperture.centers[:, 2]) < 0.0
    assert aperture.normals[0, 0] < 0.0
    assert aperture.normals[-1, 0] > 0.0
    assert np.all(aperture.normals[:, 2] > 0.0)


def test_convex_array_adaptive_tessellation_refines_azimuth_curvature():
    fieldii = ConvexArray(
        elements=5,
        width=0.2,
        height=0.5,
        kerf=0.02,
        convex_radius=5.0,
        tessellation="fieldii",
    )
    adaptive = ConvexArray(
        elements=5,
        width=0.2,
        height=0.5,
        kerf=0.02,
        convex_radius=5.0,
        tessellation="adaptive",
        max_sagitta=1e-4,
    )

    assert adaptive.metadata["tessellation"] == "adaptive"
    assert adaptive.physical_element_count == fieldii.physical_element_count
    assert len(adaptive.elements) > len(fieldii.elements)
    assert adaptive.normals[0, 0] < 0.0
    assert adaptive.normals[-1, 0] > 0.0


def test_convex_array_uses_fieldii_curved_rectangle_convention():
    aperture = ConvexArray(
        elements=6,
        width=0.3e-3,
        height=5e-3,
        kerf=0.03e-3,
        convex_radius=25e-3,
        subdivisions=(1, 3),
        tessellation="fieldii",
    )

    assert np.allclose(
        aperture.centers[0],
        [-0.00082486828000874696, -0.0016666666666666668, -1.361185924153227e-05],
        rtol=1e-9,
        atol=1e-12,
    )
    assert np.allclose(aperture.areas[0], 5.00009000567048e-07, rtol=1e-9, atol=1e-12)


def test_convex_focused_array_combines_azimuth_and_elevation_curvature():
    aperture = ConvexFocusedArray(
        elements=3,
        width=0.2,
        height=0.8,
        kerf=0.02,
        convex_radius=5.0,
        elevation_focus=2.0,
        subdivisions=(2, 3),
        tessellation="fieldii",
    )

    assert aperture.physical_element_count == 3
    assert len(aperture.elements) == 3 * 2 * 3
    assert np.ptp(aperture.centers[:, 0]) > 0.0
    assert np.ptp(aperture.centers[:, 2]) > 0.0


def test_convex_focused_array_adaptive_tessellation_combines_curvatures():
    aperture = ConvexFocusedArray(
        elements=3,
        width=0.2,
        height=0.8,
        kerf=0.02,
        convex_radius=5.0,
        elevation_focus=2.0,
        tessellation="adaptive",
        max_sagitta=1e-3,
    )

    assert aperture.metadata["tessellation"] == "adaptive"
    assert "fieldii_focus_centers" not in aperture.metadata
    assert np.ptp(aperture.centers[:, 0]) > 0.0
    assert np.ptp(aperture.centers[:, 2]) > 0.0
    assert np.all(aperture.normals[:, 2] > 0.0)


def test_convex_focused_multirow_array_indexes_all_rows():
    aperture = ConvexFocusedMultirowArray(
        elements_x=2,
        width=0.2,
        elements_y=2,
        heights=[0.3, 0.3],
        kerf_x=0.02,
        kerf_y=0.02,
        convex_radius=5.0,
        elevation_focus=2.0,
        subdivisions=(1, 2),
        tessellation="fieldii",
    )

    assert aperture.physical_element_count == 4
    assert len(aperture.elements) == 2 * 2 * 2
    assert np.array_equal(np.unique(aperture.physical_indices), np.arange(4))


def test_convex_focused_multirow_array_adaptive_tessellation_indexes_all_rows():
    aperture = ConvexFocusedMultirowArray(
        elements_x=2,
        width=0.2,
        elements_y=2,
        heights=[0.3, 0.3],
        kerf_x=0.02,
        kerf_y=0.02,
        convex_radius=5.0,
        elevation_focus=2.0,
        tessellation="adaptive",
        max_sagitta=1e-3,
    )

    assert aperture.metadata["tessellation"] == "adaptive"
    assert np.array_equal(np.unique(aperture.physical_indices), np.arange(4))
    assert np.all(aperture.areas > 0.0)


def test_two_dimensional_array_supports_sparse_enabled_mask():
    enabled = np.array([[True, False, True], [False, True, False]])

    aperture = TwoDimensionalArray(
        elements_x=2,
        elements_y=3,
        width=0.2,
        height=0.3,
        kerf_x=0.02,
        kerf_y=0.03,
        enabled=enabled,
        subdivisions=(2, 2),
    )

    assert aperture.physical_element_count == 3
    assert len(aperture.elements) == 3 * 2 * 2
    assert Array2D is TwoDimensionalArray


def test_rectangle_and_triangle_apertures_use_supplied_vertices():
    rectangles = np.array(
        [
            [
                [-0.5, -0.5, 0.0],
                [0.5, -0.5, 0.0],
                [0.5, 0.5, 0.0],
                [-0.5, 0.5, 0.0],
            ]
        ]
    )
    triangles = np.array(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ]
    )

    rectangle_aperture = RectangleAperture(rectangles)
    triangle_aperture = TriangleAperture(triangles)

    assert np.allclose(rectangle_aperture.areas, [1.0])
    assert np.allclose(triangle_aperture.areas, [0.5])
    assert np.allclose(rectangle_aperture.normals, [[0.0, 0.0, 1.0]])
    assert np.allclose(triangle_aperture.normals, [[0.0, 0.0, 1.0]])


def test_line_bounded_aperture_clips_half_planes():
    lines = np.array(
        [
            [1, 1, 0.0, 1, -0.5, 0],  # x >= -0.5
            [1, 1, 0.0, 1, 0.5, 1],   # x <= 0.5
            [1, 1, 0.0, 0, -0.5, 1],  # y >= -0.5
            [1, 1, 0.0, 0, 0.5, 0],   # y <= 0.5
        ],
        dtype=float,
    )

    aperture = LineBoundedAperture(lines, bounding_extent=2.0)

    assert aperture.physical_element_count == 1
    assert len(aperture.elements) == 1
    assert np.allclose(aperture.areas, [1.0])
    assert np.allclose(aperture.centers, [[0.0, 0.0, 0.0]])
