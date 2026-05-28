import numpy as np
import pytest

from openfield import Medium, Simulation
from openfield.apertures import (
    Aperture,
    ConcavePiston,
    ConvexArray,
    ConvexFocusedArray,
    ConvexFocusedMultirowArray,
    FocusedLinearArray,
    FocusedMultirowArray,
    LineBoundedAperture,
    Piston,
    RectangleAperture,
    SubElement,
    TriangleAperture,
)


def test_simulation_spatial_impulse_response_shape_and_time_axis():
    sim = Simulation(
        sampling_frequency=100.0,
        medium=Medium(sound_speed=10.0),
    )
    aperture = Piston(radius=0.2, element_size=0.1)

    response = sim.spatial_impulse_response(
        aperture=aperture,
        points=[[0.0, 0.0, 1.0], [0.0, 0.0, 1.5]],
    )

    assert response.samples.ndim == 2
    assert response.samples.shape[1] == 2
    assert response.sample_count == response.time.shape[0]
    assert np.any(response.samples > 0.0)


def test_adaptive_curved_apertures_produce_spatial_impulse_responses():
    sim = Simulation(sampling_frequency=50e6, medium=Medium(sound_speed=1540.0))
    apertures = [
        Piston.adaptive(radius=1.0e-3, element_size=0.5e-3),
        ConcavePiston.adaptive(radius=1.0e-3, focal_radius=20e-3, element_size=0.5e-3),
        FocusedLinearArray(
            elements=2,
            width=0.3e-3,
            height=2.0e-3,
            kerf=0.03e-3,
            elevation_focus=20e-3,
            tessellation="adaptive",
        ),
        FocusedMultirowArray(
            elements_x=2,
            width=0.3e-3,
            elements_y=2,
            heights=[1.0e-3, 1.0e-3],
            kerf_x=0.03e-3,
            kerf_y=0.05e-3,
            elevation_focus=20e-3,
            tessellation="adaptive",
        ),
        ConvexArray(
            elements=2,
            width=0.3e-3,
            height=2.0e-3,
            kerf=0.03e-3,
            convex_radius=25e-3,
            tessellation="adaptive",
        ),
        ConvexFocusedArray(
            elements=2,
            width=0.3e-3,
            height=2.0e-3,
            kerf=0.03e-3,
            convex_radius=25e-3,
            elevation_focus=20e-3,
            tessellation="adaptive",
        ),
        ConvexFocusedMultirowArray(
            elements_x=2,
            width=0.3e-3,
            elements_y=2,
            heights=[1.0e-3, 1.0e-3],
            kerf_x=0.03e-3,
            kerf_y=0.05e-3,
            convex_radius=25e-3,
            elevation_focus=20e-3,
            tessellation="adaptive",
        ),
    ]

    for aperture in apertures:
        response = sim.spatial_impulse_response(aperture, [[0.0, 0.0, 30e-3]])

        assert response.samples.shape[1] == 1
        assert np.all(np.isfinite(response.samples))
        assert np.any(response.samples[:, 0] != 0.0)


def test_nonfacet_subelement_rejects_coincident_field_point():
    sim = Simulation(sampling_frequency=100e6, medium=Medium(sound_speed=1540.0))
    aperture = Aperture(
        elements=(
            SubElement(
                center=[0.0, 0.0, 0.0],
                normal=[0.0, 0.0, 1.0],
                area=1.0e-6,
                physical_index=0,
                subelement_index=0,
            ),
        ),
        physical_element_count=1,
    )

    with pytest.raises(ValueError, match="must not coincide"):
        sim.spatial_impulse_response(aperture, [[0.0, 0.0, 0.0]])


def test_triangle_and_line_bounded_flat_polygon_responses_match():
    sim = Simulation(sampling_frequency=100e6, medium=Medium(sound_speed=1540.0))
    triangles = np.array(
        [
            [1, -0.5e-3, -0.25e-3, 0.0, 0.5e-3, -0.25e-3, 0.0, 0.5e-3, 0.25e-3, 0.0, 1.0],
            [1, -0.5e-3, -0.25e-3, 0.0, 0.5e-3, 0.25e-3, 0.0, -0.5e-3, 0.25e-3, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    lines = np.array(
        [
            [1, 1, 0.0, 1, -0.5e-3, 0],
            [1, 1, 0.0, 1, 0.5e-3, 1],
            [1, 1, 0.0, 0, -0.25e-3, 1],
            [1, 1, 0.0, 0, 0.25e-3, 0],
        ],
        dtype=np.float64,
    )
    points = [[0.0, 0.0, 30e-3], [2e-3, 0.0, 40e-3]]

    triangle_response = sim.spatial_impulse_response(
        TriangleAperture.from_fieldii_triangles(triangles),
        points,
    )
    line_response = sim.spatial_impulse_response(
        LineBoundedAperture.from_fieldii_lines(lines, bounding_extent=1e-3),
        points,
    )

    shared_samples = min(triangle_response.sample_count, line_response.sample_count)
    assert np.allclose(triangle_response.start_time, line_response.start_time)
    assert np.allclose(
        triangle_response.samples[:shared_samples],
        line_response.samples[:shared_samples],
    )
    assert np.allclose(triangle_response.samples[shared_samples:], 0.0)
    assert np.allclose(line_response.samples[shared_samples:], 0.0)


def test_flat_rectangle_response_matches_exact_triangle_facets():
    sim = Simulation(sampling_frequency=100e6, medium=Medium(sound_speed=1540.0))
    rectangle_vertices = np.array(
        [[[-0.5e-3, -0.25e-3, 0.0], [0.5e-3, -0.25e-3, 0.0], [0.5e-3, 0.25e-3, 0.0], [-0.5e-3, 0.25e-3, 0.0]]],
        dtype=np.float64,
    )
    triangle_vertices = np.array(
        [
            [rectangle_vertices[0, 0], rectangle_vertices[0, 1], rectangle_vertices[0, 2]],
            [rectangle_vertices[0, 0], rectangle_vertices[0, 2], rectangle_vertices[0, 3]],
        ],
        dtype=np.float64,
    )
    points = [[0.0, 0.0, 30e-3], [2e-3, 0.0, 40e-3]]

    rectangle = sim.spatial_impulse_response(RectangleAperture(rectangle_vertices), points)
    triangles = sim.spatial_impulse_response(TriangleAperture(triangle_vertices, physical_indices=[0, 0]), points)

    shared_samples = min(rectangle.sample_count, triangles.sample_count)
    assert np.allclose(rectangle.start_time, triangles.start_time)
    assert np.allclose(rectangle.samples[:shared_samples], triangles.samples[:shared_samples])
    assert np.allclose(rectangle.samples[shared_samples:], 0.0)
    assert np.allclose(triangles.samples[shared_samples:], 0.0)


def test_nonplanar_rectangle_response_is_triangulated():
    sim = Simulation(sampling_frequency=100e6, medium=Medium(sound_speed=1540.0))
    rectangle_vertices = np.array(
        [[[-0.5e-3, -0.25e-3, 0.0], [0.5e-3, -0.25e-3, 0.1e-3], [0.5e-3, 0.25e-3, 0.0], [-0.5e-3, 0.25e-3, 0.05e-3]]],
        dtype=np.float64,
    )
    triangle_vertices = np.array(
        [
            [rectangle_vertices[0, 0], rectangle_vertices[0, 1], rectangle_vertices[0, 2]],
            [rectangle_vertices[0, 0], rectangle_vertices[0, 2], rectangle_vertices[0, 3]],
        ],
        dtype=np.float64,
    )
    points = [[0.0, 0.0, 30e-3], [2e-3, 0.0, 40e-3]]

    rectangle = sim.spatial_impulse_response(RectangleAperture(rectangle_vertices), points)
    triangles = sim.spatial_impulse_response(TriangleAperture(triangle_vertices, physical_indices=[0, 0]), points)

    shared_samples = min(rectangle.sample_count, triangles.sample_count)
    assert np.allclose(rectangle.start_time, triangles.start_time)
    assert np.allclose(rectangle.samples[:shared_samples], triangles.samples[:shared_samples])
    assert np.allclose(rectangle.samples[shared_samples:], 0.0)
    assert np.allclose(triangles.samples[shared_samples:], 0.0)


def test_flat_polygon_time_axis_includes_projected_interior_support():
    triangles = np.array(
        [
            [1, -0.5e-3, -0.25e-3, 0.0, 0.5e-3, -0.25e-3, 0.0, 0.5e-3, 0.25e-3, 0.0, 1.0],
            [1, -0.5e-3, -0.25e-3, 0.0, 0.5e-3, 0.25e-3, 0.0, -0.5e-3, 0.25e-3, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    aperture = TriangleAperture.from_fieldii_triangles(triangles)
    points = [[0.0, 0.0, 30e-3]]

    low_fs = Simulation(sampling_frequency=100e6, medium=Medium(sound_speed=1540.0))
    high_fs = Simulation(sampling_frequency=1e9, medium=Medium(sound_speed=1540.0))
    low_response = low_fs.spatial_impulse_response(aperture, points)
    high_response = high_fs.spatial_impulse_response(aperture, points)

    low_area = np.sum(low_response.samples[:, 0]) / low_fs.sampling_frequency
    high_area = np.sum(high_response.samples[:, 0]) / high_fs.sampling_frequency
    assert np.isclose(high_area, low_area, rtol=5e-3)
