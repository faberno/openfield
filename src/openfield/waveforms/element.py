from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ElementWaveforms:
    """Per-physical-element transmit waveforms."""

    waveforms: tuple[object | None, ...]

    def __init__(self, waveforms):
        waveforms = tuple(waveforms)
        if not waveforms:
            raise ValueError("waveforms must not be empty")
        object.__setattr__(self, "waveforms", waveforms)

    def waveform_for(self, physical_index: int, default=None):
        waveform = self.waveforms[physical_index]
        return default if waveform is None else waveform

    def select_physical_elements(self, indices) -> "ElementWaveforms":
        return ElementWaveforms([self.waveforms[int(index)] for index in indices])
