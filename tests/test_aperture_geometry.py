import numpy as np

from openfield.apertures import ConcavePiston, LinearArray, Piston


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
