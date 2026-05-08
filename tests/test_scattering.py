import numpy as np
import pytest

from openfield import Medium, Simulation
from openfield.apertures import LinearArray, Piston
from openfield.waveforms import Waveform


def _configured_piston():
    return (
        Piston(radius=0.2, element_size=0.2)
        .with_excitation(Waveform([1.0, 0.5], sampling_frequency=100.0, start_time=0.01))
        .with_impulse_response(Waveform([1.0, -0.25], sampling_frequency=100.0, start_time=0.02))
    )


def test_pulse_echo_response_convolves_transmit_receive_paths_and_waveforms():
    sim = Simulation(sampling_frequency=100.0, medium=Medium(sound_speed=10.0))
    transmit = _configured_piston()
    receive = _configured_piston().with_impulse_response(
        Waveform([0.5, 1.0], sampling_frequency=100.0, start_time=0.03)
    )
    point = [[0.0, 0.0, 1.0]]

    h_tx = sim.spatial_impulse_response(transmit, point)
    h_rx = sim.spatial_impulse_response(receive, point)
    response = sim.pulse_echo_response(transmit, receive, point)

    expected = np.convolve(h_tx.samples[:, 0], h_rx.samples[:, 0], mode="full") / 100.0
    expected = np.convolve(expected, [1.0, 0.5], mode="full") / 100.0
    expected = np.convolve(expected, [1.0, -0.25], mode="full") / 100.0
    expected = np.convolve(expected, [0.5, 1.0], mode="full") / 100.0
    expected = expected[:-1]

    assert response.samples.shape == (expected.size, 1)
    assert np.allclose(response.samples[:, 0], expected)
    assert response.start_time == pytest.approx(
        h_tx.start_time + h_rx.start_time + 0.01 + 0.02 + 0.03
    )


def test_scatterer_response_weights_and_sums_points():
    sim = Simulation(sampling_frequency=100.0, medium=Medium(sound_speed=10.0))
    aperture = _configured_piston()
    points = [[0.0, 0.0, 1.0], [0.0, 0.0, 1.2]]
    amplitudes = [2.0, -0.5]

    per_point = sim.pulse_echo_response(aperture, aperture, points)
    response = sim.scatterer_response(aperture, aperture, points, amplitudes)

    expected = per_point.samples @ np.asarray(amplitudes)
    assert response.samples.shape == (per_point.sample_count - 1, 1)
    assert np.allclose(response.samples[:, 0], expected[:-1])
    assert response.start_time == per_point.start_time


def test_channelized_responses_have_fieldii_column_order():
    sim = Simulation(sampling_frequency=100.0, medium=Medium(sound_speed=10.0))
    aperture = (
        LinearArray(elements=2, width=0.1, height=0.1, kerf=0.01, subdivisions=(1, 1))
        .with_excitation(Waveform([1.0], sampling_frequency=100.0))
        .with_impulse_response(Waveform([1.0], sampling_frequency=100.0))
    )
    points = [[0.0, 0.0, 1.0]]

    receive_channels = sim.receive_channel_responses(aperture, aperture, points, [1.0])
    full_matrix = sim.full_matrix_capture(aperture, aperture, points, [1.0])

    assert receive_channels.samples.shape[1] == 2
    assert full_matrix.samples.shape[1] == 4
    assert np.any(receive_channels.samples)
    assert np.any(full_matrix.samples)


def test_full_matrix_capture_decimates_output_sampling_frequency():
    sim = Simulation(sampling_frequency=100.0, medium=Medium(sound_speed=10.0))
    aperture = (
        LinearArray(elements=2, width=0.1, height=0.1, kerf=0.01, subdivisions=(1, 1))
        .with_excitation(Waveform([1.0], sampling_frequency=100.0))
        .with_impulse_response(Waveform([1.0], sampling_frequency=100.0))
    )

    full_rate = sim.full_matrix_capture(aperture, aperture, [[0.0, 0.0, 1.0]], [1.0])
    decimated = sim.full_matrix_capture(
        aperture,
        aperture,
        [[0.0, 0.0, 1.0]],
        [1.0],
        decimation_factor=2,
    )

    assert decimated.sampling_frequency == 50.0
    assert np.allclose(decimated.samples, full_rate.samples[::2])


def test_pulse_echo_response_requires_required_waveforms():
    sim = Simulation(sampling_frequency=100.0, medium=Medium(sound_speed=10.0))
    aperture = Piston(radius=0.2, element_size=0.2)

    with pytest.raises(ValueError, match="excitation"):
        sim.pulse_echo_response(aperture, aperture, [[0.0, 0.0, 1.0]])

    transmit = aperture.with_excitation(Waveform([1.0], sampling_frequency=100.0))
    with pytest.raises(ValueError, match="impulse response"):
        sim.pulse_echo_response(transmit, aperture, [[0.0, 0.0, 1.0]])
