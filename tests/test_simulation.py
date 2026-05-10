import numpy as np

from openfield import Medium, Simulation
from openfield.apertures import LineBoundedAperture, Piston, TriangleAperture


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
