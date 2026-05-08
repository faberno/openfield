import numpy as np
import pytest

from openfield import Medium, Simulation
from openfield.apertures import LinearArray, Piston
from openfield.waveforms import Waveform


def test_emitted_pressure_convolves_spatial_response_with_waveforms():
    sim = Simulation(sampling_frequency=100.0, medium=Medium(sound_speed=10.0))
    aperture = (
        Piston(radius=0.2, element_size=0.1)
        .with_excitation(Waveform([1.0, 2.0], sampling_frequency=100.0, start_time=0.01))
        .with_impulse_response(Waveform([1.0, -1.0, 0.5], sampling_frequency=100.0, start_time=0.02))
    )

    spatial = sim.spatial_impulse_response(aperture, [[0.0, 0.0, 1.0]])
    pressure = sim.emitted_pressure(aperture, [[0.0, 0.0, 1.0]])

    expected = np.convolve(spatial.samples[:, 0], [1.0, 2.0], mode="full") / sim.sampling_frequency
    expected = np.convolve(expected, [1.0, -1.0, 0.5], mode="full") / sim.sampling_frequency

    assert pressure.samples.shape == (spatial.sample_count + 2 + 3 - 2, 1)
    assert np.allclose(pressure.samples[:, 0], expected)
    assert pressure.start_time == pytest.approx(spatial.start_time + 0.01 + 0.02)


def test_emitted_pressure_requires_excitation_and_impulse_response():
    sim = Simulation(sampling_frequency=100.0, medium=Medium(sound_speed=10.0))
    aperture = Piston(radius=0.2, element_size=0.1)

    with pytest.raises(ValueError, match="excitation"):
        sim.emitted_pressure(aperture, [[0.0, 0.0, 1.0]])

    aperture = aperture.with_excitation(Waveform([1.0], sampling_frequency=100.0))
    with pytest.raises(ValueError, match="impulse response"):
        sim.emitted_pressure(aperture, [[0.0, 0.0, 1.0]])


def test_emitted_pressure_samples_waveform_objects_at_simulation_frequency():
    class UnitPulse:
        def sample(self, sampling_frequency):
            return Waveform([1.0], sampling_frequency=sampling_frequency)

    sim = Simulation(sampling_frequency=100.0, medium=Medium(sound_speed=10.0))
    aperture = (
        Piston(radius=0.2, element_size=0.1)
        .with_excitation(UnitPulse())
        .with_impulse_response(UnitPulse())
    )

    pressure = sim.emitted_pressure(aperture, [[0.0, 0.0, 1.0]])

    assert pressure.samples.shape[1] == 1
    assert np.any(pressure.samples > 0.0)


def test_emitted_pressure_supports_per_element_waveforms():
    sim = Simulation(sampling_frequency=100.0, medium=Medium(sound_speed=10.0))
    aperture = (
        LinearArray(elements=2, width=0.1, height=0.1, kerf=0.0, subdivisions=(1, 1))
        .with_element_waveforms(
            [
                Waveform([1.0], sampling_frequency=100.0),
                Waveform([0.0], sampling_frequency=100.0),
            ]
        )
        .with_impulse_response(Waveform([1.0], sampling_frequency=100.0))
    )

    response = sim.emitted_pressure(aperture, [[0.0, 0.0, 1.0]])
    first_only = sim.emitted_pressure(
        aperture.select_physical_elements([0]).with_element_waveforms([Waveform([1.0], 100.0)]),
        [[0.0, 0.0, 1.0]],
    )

    assert np.allclose(response.samples, first_only.samples)
