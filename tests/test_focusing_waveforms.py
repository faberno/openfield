import numpy as np

from openfield.apertures import LinearArray, Piston
from openfield.focusing import Apodization, FixedFocus
from openfield.waveforms import ToneBurst, Waveform, sampled_waveform


def test_fixed_focus_returns_one_delay_per_physical_element():
    aperture = LinearArray(elements=3, width=0.2, height=1.0, kerf=0.0)
    focus = FixedFocus(point=[0.0, 0.0, 10.0])

    delays = focus.delays(aperture, sound_speed=2.0)

    assert delays.shape == (3,)
    assert delays[1] == np.max(delays)
    assert np.allclose(delays[0], delays[2])


def test_apodization_validates_and_indexes_time_rows():
    apodization = Apodization(
        values=[[1.0, 0.5, 1.0], [0.0, 1.0, 0.0]],
        times=[0.0, 1.0],
    )

    assert np.allclose(apodization.at_time(0.5), [1.0, 0.5, 1.0])
    assert np.allclose(apodization.at_time(1.0), [0.0, 1.0, 0.0])


def test_tone_burst_sampling():
    pulse = ToneBurst(center_frequency=2.0, cycles=2, window="hann").sample(20.0)

    assert isinstance(pulse, Waveform)
    assert pulse.samples.ndim == 1
    assert pulse.sampling_frequency == 20.0
    assert np.isclose(pulse.samples[0], 0.0)
    assert sampled_waveform(pulse, 20.0) is pulse


def test_aperture_with_methods_preserve_concrete_type():
    pulse = ToneBurst(center_frequency=2.0, cycles=2)
    aperture = Piston(radius=1.0, element_size=0.25)

    configured = aperture.focused_at([0.0, 0.0, 10.0]).with_excitation(pulse)

    assert isinstance(configured, Piston)
    assert configured.focus is not None
    assert configured.excitation is pulse
    assert configured.metadata == aperture.metadata
