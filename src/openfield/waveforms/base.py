from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Waveform:
    samples: np.ndarray
    sampling_frequency: float
    start_time: float = 0.0

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples, dtype=np.float64)
        if samples.ndim != 1:
            raise ValueError("samples must be one-dimensional")
        if samples.size == 0:
            raise ValueError("samples must not be empty")
        if self.sampling_frequency <= 0:
            raise ValueError("sampling_frequency must be positive")
        object.__setattr__(self, "samples", samples)

    @property
    def duration(self) -> float:
        return self.samples.size / self.sampling_frequency

    @property
    def time(self) -> np.ndarray:
        return self.start_time + np.arange(self.samples.size) / self.sampling_frequency


def sampled_waveform(value, sampling_frequency: float) -> Waveform:
    if isinstance(value, Waveform):
        if value.sampling_frequency != sampling_frequency:
            raise ValueError("sampled Waveform uses a different sampling_frequency")
        return value
    if hasattr(value, "sample"):
        return value.sample(sampling_frequency)
    return Waveform(samples=value, sampling_frequency=sampling_frequency)
