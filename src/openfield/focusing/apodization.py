from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Apodization:
    """Time-indexed physical-element apodization values."""

    values: np.ndarray
    times: np.ndarray | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2:
            raise ValueError("values must be one- or two-dimensional")
        if values.shape[1] < 1:
            raise ValueError("values must contain at least one physical element")

        if self.times is None:
            times = np.zeros(values.shape[0], dtype=np.float64)
        else:
            times = np.asarray(self.times, dtype=np.float64)
        if times.shape != (values.shape[0],):
            raise ValueError("times must match the number of apodization rows")
        if np.any(np.diff(times) < 0):
            raise ValueError("times must be sorted")
        if np.any(np.all(values == 0.0, axis=1)):
            raise ValueError("each apodization row must contain a non-zero value")

        object.__setattr__(self, "values", values)
        object.__setattr__(self, "times", times)

    @classmethod
    def uniform(cls, elements: int, value: float = 1.0) -> "Apodization":
        return cls(np.full(elements, value, dtype=np.float64))

    @classmethod
    def hamming(cls, elements: int) -> "Apodization":
        return cls(np.hamming(elements))

    def at_time(self, time: float) -> np.ndarray:
        index = int(np.searchsorted(self.times, time, side="right") - 1)
        index = max(index, 0)
        return self.values[index]

    def select_physical_elements(self, indices) -> "Apodization":
        """Return an apodization timeline restricted to selected elements."""

        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("indices must be one-dimensional")
        if np.any(indices < 0) or np.any(indices >= self.values.shape[1]):
            raise ValueError("indices are outside the apodization element range")
        return Apodization(values=self.values[:, indices], times=self.times)
