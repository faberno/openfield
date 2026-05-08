from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimeResponse:
    """Sampled time-domain response with an absolute start time."""

    samples: np.ndarray
    sampling_frequency: float
    start_time: float

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples, dtype=np.float64)
        if samples.ndim < 1:
            raise ValueError("samples must have at least one dimension")
        if self.sampling_frequency <= 0:
            raise ValueError("sampling_frequency must be positive")
        object.__setattr__(self, "samples", samples)

    @property
    def sample_count(self) -> int:
        return int(self.samples.shape[0])

    @property
    def time(self) -> np.ndarray:
        return self.start_time + np.arange(self.sample_count) / self.sampling_frequency
