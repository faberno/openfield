from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _point3(value, name: str) -> np.ndarray:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector")
    point = point.copy()
    point.setflags(write=False)
    return point


@dataclass(frozen=True)
class FixedFocus:
    """Fixed focal point relative to a reference origin on the aperture."""

    point: np.ndarray
    origin: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float64))
    quantization: float | None = None

    def __post_init__(self) -> None:
        point = _point3(self.point, "point")
        origin = _point3(self.origin, "origin")
        if self.quantization is not None and self.quantization < 0:
            raise ValueError("quantization must be non-negative")
        object.__setattr__(self, "point", point)
        object.__setattr__(self, "origin", origin)

    def delays(self, aperture, sound_speed: float, time: float = 0.0) -> np.ndarray:
        del time
        if sound_speed <= 0:
            raise ValueError("sound_speed must be positive")
        reference_distance = float(np.linalg.norm(self.origin - self.point))
        focus_centers = aperture.metadata.get("fieldii_focus_centers", aperture.physical_centers)
        focus_centers = np.asarray(focus_centers, dtype=np.float64)
        if focus_centers.shape != (aperture.physical_element_count, 3):
            raise ValueError("focus center count must match physical element count")
        distances = np.linalg.norm(focus_centers - self.point, axis=1)
        delays = (reference_distance - distances) / sound_speed
        if self.quantization:
            delays = np.round(delays / self.quantization) * self.quantization
        return delays
