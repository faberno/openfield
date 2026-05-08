import numpy as np

from openfield import Medium, Simulation
from openfield.apertures import Piston


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
