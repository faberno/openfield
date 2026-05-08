import sys

import numpy as np
import pytest

from openfield.apertures import LinearArray
from openfield.visualize.apertures import aperture_mesh_data


def test_aperture_mesh_data_has_one_face_per_subelement():
    aperture = LinearArray(
        elements=2,
        width=0.3,
        height=0.6,
        kerf=0.1,
        subdivisions=(2, 2),
    )

    vertices, faces = aperture_mesh_data(aperture)

    assert vertices.shape == (len(aperture.elements) * 4, 3)
    assert len(faces) == len(aperture.elements)
    assert all(len(face) == 4 for face in faces)


def test_aperture_show_reports_missing_vedo(monkeypatch):
    monkeypatch.setitem(sys.modules, "vedo", None)
    aperture = LinearArray(elements=1, width=0.3, height=0.6, kerf=0.0)

    with pytest.raises(ImportError, match="requires vedo"):
        aperture.show_3d(interactive=False)


def test_aperture_mesh_data_can_shrink_elements_around_centers():
    aperture = LinearArray(elements=1, width=1.0, height=1.0, kerf=0.0)

    full_vertices, _ = aperture_mesh_data(aperture, element_scale=1.0)
    shrunk_vertices, _ = aperture_mesh_data(aperture, element_scale=0.5)

    center = aperture.elements[0].center
    assert np.allclose(shrunk_vertices - center, (full_vertices - center) * 0.5)
