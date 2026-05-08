from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class Baffle:
    """Aperture baffle condition."""

    kind: Literal["rigid", "soft"] = "rigid"

    def __post_init__(self) -> None:
        if self.kind not in {"rigid", "soft"}:
            raise ValueError("kind must be 'rigid' or 'soft'")


@dataclass(frozen=True)
class SubElementApodization:
    """One apodization weight per subelement in aperture element order."""

    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("values must be one-dimensional")
        if values.size == 0:
            raise ValueError("values must not be empty")
        object.__setattr__(self, "values", values)

    def values_for(self, aperture) -> np.ndarray:
        if self.values.shape != (len(aperture.elements),):
            raise ValueError("subelement apodization length must match subelement count")
        return self.values

    def select_subelements(self, indices) -> "SubElementApodization":
        return SubElementApodization(self.values[np.asarray(indices, dtype=np.int64)])


@dataclass(frozen=True)
class SubElementDelays:
    """One extra delay per subelement in aperture element order."""

    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("values must be one-dimensional")
        if values.size == 0:
            raise ValueError("values must not be empty")
        object.__setattr__(self, "values", values)

    def values_for(self, aperture) -> np.ndarray:
        if self.values.shape != (len(aperture.elements),):
            raise ValueError("subelement delay length must match subelement count")
        return self.values

    def select_subelements(self, indices) -> "SubElementDelays":
        return SubElementDelays(self.values[np.asarray(indices, dtype=np.int64)])
