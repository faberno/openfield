from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import Waveform


@dataclass(frozen=True)
class ToneBurst:
    center_frequency: float
    cycles: float
    amplitude: float = 1.0
    phase: float = 0.0
    window: str | None = None

    def __post_init__(self) -> None:
        if self.center_frequency <= 0:
            raise ValueError("center_frequency must be positive")
        if self.cycles <= 0:
            raise ValueError("cycles must be positive")

    def sample(self, sampling_frequency: float) -> Waveform:
        if sampling_frequency <= 0:
            raise ValueError("sampling_frequency must be positive")
        duration = self.cycles / self.center_frequency
        sample_count = max(1, int(np.floor(duration * sampling_frequency)) + 1)
        time = np.arange(sample_count, dtype=np.float64) / sampling_frequency
        samples = self.amplitude * np.sin(2 * np.pi * self.center_frequency * time + self.phase)

        if self.window is not None:
            window = self.window.lower()
            if window in {"hann", "hanning"}:
                samples = samples * np.hanning(sample_count)
            elif window == "hamming":
                samples = samples * np.hamming(sample_count)
            else:
                raise ValueError(f"unsupported window: {self.window}")

        return Waveform(samples=samples, sampling_frequency=sampling_frequency)
