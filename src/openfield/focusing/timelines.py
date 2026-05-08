from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fixed import FixedFocus, _point3


def _times_array(times) -> np.ndarray:
    times = np.asarray(times, dtype=np.float64)
    if times.ndim == 0:
        times = times[None]
    if times.ndim != 1:
        raise ValueError("times must be one-dimensional")
    if times.size == 0:
        raise ValueError("times must not be empty")
    if np.any(np.diff(times) < 0):
        raise ValueError("times must be sorted")
    return times


@dataclass(frozen=True)
class FocusTimeline:
    """Time-indexed focal points."""

    times: np.ndarray
    points: np.ndarray
    origin: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float64))
    quantization: float | None = None

    def __post_init__(self) -> None:
        times = _times_array(self.times)
        points = np.asarray(self.points, dtype=np.float64)
        if points.ndim == 1:
            points = points[None, :]
        if points.shape != (times.size, 3):
            raise ValueError("points must have shape (len(times), 3)")
        origin = _point3(self.origin, "origin")
        if self.quantization is not None and self.quantization < 0:
            raise ValueError("quantization must be non-negative")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "origin", origin)

    def point_at_time(self, time: float) -> np.ndarray:
        index = int(np.searchsorted(self.times, time, side="right") - 1)
        index = max(index, 0)
        return self.points[index]

    def delays(self, aperture, sound_speed: float, time: float = 0.0) -> np.ndarray:
        return FixedFocus(
            point=self.point_at_time(time),
            origin=self.origin,
            quantization=self.quantization,
        ).delays(aperture, sound_speed)


@dataclass(frozen=True)
class DelayTimeline:
    """Time-indexed explicit physical-element delays."""

    times: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        times = _times_array(self.times)
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2:
            raise ValueError("values must be one- or two-dimensional")
        if values.shape[0] != times.size:
            raise ValueError("values must have one row per time")
        if values.shape[1] < 1:
            raise ValueError("values must contain at least one physical element")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "values", values)

    def at_time(self, time: float) -> np.ndarray:
        index = int(np.searchsorted(self.times, time, side="right") - 1)
        index = max(index, 0)
        return self.values[index]

    def delays(self, aperture, sound_speed: float, time: float = 0.0) -> np.ndarray:
        del sound_speed
        delays = self.at_time(time)
        if delays.shape != (aperture.physical_element_count,):
            raise ValueError("delay count must match physical element count")
        return delays

    def select_physical_elements(self, indices) -> "DelayTimeline":
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("indices must be one-dimensional")
        if np.any(indices < 0) or np.any(indices >= self.values.shape[1]):
            raise ValueError("indices are outside the delay element range")
        return DelayTimeline(times=self.times, values=self.values[:, indices])


@dataclass(frozen=True)
class DynamicFocus:
    """Far-field dynamic focus direction active from a given time."""

    start_time: float
    direction_zx: float
    direction_zy: float
    origin: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float64))
    quantization: float | None = None

    def __post_init__(self) -> None:
        origin = _point3(self.origin, "origin")
        if self.quantization is not None and self.quantization < 0:
            raise ValueError("quantization must be non-negative")
        object.__setattr__(self, "start_time", float(self.start_time))
        object.__setattr__(self, "direction_zx", float(self.direction_zx))
        object.__setattr__(self, "direction_zy", float(self.direction_zy))
        object.__setattr__(self, "origin", origin)

    @property
    def direction(self) -> np.ndarray:
        x = np.sin(self.direction_zx)
        y = np.sin(self.direction_zy)
        z_squared = max(0.0, 1.0 - x * x - y * y)
        direction = np.array([x, y, np.sqrt(z_squared)], dtype=np.float64)
        return direction / np.linalg.norm(direction)

    def delays(self, aperture, sound_speed: float, time: float = 0.0) -> np.ndarray:
        if sound_speed <= 0:
            raise ValueError("sound_speed must be positive")
        if time < self.start_time:
            return np.zeros(aperture.physical_element_count, dtype=np.float64)
        relative_centers = aperture.physical_centers - self.origin
        delays = relative_centers @ self.direction / sound_speed
        if self.quantization:
            delays = np.round(delays / self.quantization) * self.quantization
        return delays
