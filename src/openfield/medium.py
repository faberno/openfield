from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Medium:
    """Acoustic medium parameters.

    Units are SI: meters, seconds, kilograms, hertz.
    """

    sound_speed: float = 1540.0
    density: float | None = None
    attenuation_db_per_m: float = 0.0
    frequency_attenuation_db_per_m_hz: float = 0.0
    attenuation_center_frequency: float = 0.0

    def __post_init__(self) -> None:
        if self.sound_speed <= 0:
            raise ValueError("sound_speed must be positive")
        if self.density is not None and self.density <= 0:
            raise ValueError("density must be positive when provided")
        if self.attenuation_db_per_m < 0:
            raise ValueError("attenuation_db_per_m must be non-negative")
        if self.frequency_attenuation_db_per_m_hz < 0:
            raise ValueError("frequency_attenuation_db_per_m_hz must be non-negative")
