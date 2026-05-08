import numpy as np

from openfield import Medium, Simulation
from openfield.apertures import LinearArray, Piston
from openfield.focusing import Apodization, DelayTimeline, DynamicFocus, FixedFocus, FocusTimeline
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


def test_focus_timeline_uses_active_focus_point():
    aperture = LinearArray(elements=3, width=0.2, height=1.0, kerf=0.0)
    timeline = FocusTimeline(
        times=[0.0, 1.0],
        points=[[0.0, 0.0, 10.0], [1.0, 0.0, 10.0]],
    )

    early = timeline.delays(aperture, sound_speed=2.0, time=0.5)
    late = timeline.delays(aperture, sound_speed=2.0, time=1.0)

    assert np.allclose(early, FixedFocus([0.0, 0.0, 10.0]).delays(aperture, 2.0))
    assert not np.allclose(early, late)


def test_delay_timeline_validates_and_selects_elements():
    timeline = DelayTimeline(times=[0.0, 1.0], values=[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    aperture = LinearArray(elements=3, width=0.2, height=1.0, kerf=0.0)

    assert np.allclose(timeline.delays(aperture, sound_speed=2.0, time=0.5), [0.0, 1.0, 2.0])
    assert np.allclose(timeline.delays(aperture, sound_speed=2.0, time=1.0), [3.0, 4.0, 5.0])
    assert np.allclose(timeline.select_physical_elements([2, 0]).values, [[2.0, 0.0], [5.0, 3.0]])


def test_dynamic_focus_returns_steering_delays_after_start_time():
    aperture = LinearArray(elements=3, width=0.2, height=1.0, kerf=0.0)
    focus = DynamicFocus(start_time=1.0, direction_zx=0.1, direction_zy=0.0)

    before = focus.delays(aperture, sound_speed=2.0, time=0.5)
    after = focus.delays(aperture, sound_speed=2.0, time=1.0)

    assert np.allclose(before, 0.0)
    assert after.shape == (3,)
    assert after[0] < after[-1]


def test_spatial_impulse_response_applies_static_apodization():
    sim = Simulation(sampling_frequency=100.0, medium=Medium(sound_speed=10.0))
    aperture = LinearArray(elements=2, width=0.1, height=0.1, kerf=0.0, subdivisions=(1, 1))
    apodized = aperture.with_apodization(Apodization([1.0, 0.0]))

    full = sim.spatial_impulse_response(aperture, [[0.0, 0.0, 1.0]])
    muted = sim.spatial_impulse_response(apodized, [[0.0, 0.0, 1.0]])

    assert np.sum(muted.samples) < np.sum(full.samples)


def test_spatial_impulse_response_applies_subelement_controls_and_baffle():
    sim = Simulation(sampling_frequency=100.0, medium=Medium(sound_speed=10.0))
    aperture = LinearArray(elements=1, width=0.2, height=0.2, kerf=0.0, subdivisions=(2, 1))

    reference = sim.spatial_impulse_response(aperture, [[0.0, 0.0, 1.0]])
    muted = sim.spatial_impulse_response(
        aperture.with_subelement_apodization([1.0, 0.0]),
        [[0.0, 0.0, 1.0]],
    )
    delayed = sim.spatial_impulse_response(
        aperture.with_subelement_delays([0.01, 0.01]),
        [[0.0, 0.0, 1.0]],
    )
    soft = sim.spatial_impulse_response(
        aperture.with_baffle("soft"),
        [[0.5, 0.0, 1.0]],
    )
    rigid = sim.spatial_impulse_response(
        aperture.with_baffle("rigid"),
        [[0.5, 0.0, 1.0]],
    )

    assert np.sum(muted.samples) < np.sum(reference.samples)
    assert delayed.start_time > reference.start_time
    assert np.sum(soft.samples) < np.sum(rigid.samples)


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
