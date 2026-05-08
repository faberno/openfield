from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .elements import SubElement


@dataclass(frozen=True)
class Aperture:
    """A tessellated acoustic aperture.

    The package uses zero-based physical element indices internally.
    """

    elements: tuple[SubElement, ...]
    physical_element_count: int
    name: str = "aperture"
    focus: Any | None = None
    apodization: Any | None = None
    impulse_response: Any | None = None
    excitation: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.physical_element_count < 1:
            raise ValueError("physical_element_count must be at least one")
        if not self.elements:
            raise ValueError("aperture must contain at least one subelement")
        elements = tuple(self.elements)
        max_index = max(element.physical_index for element in elements)
        if max_index >= self.physical_element_count:
            raise ValueError("element physical_index exceeds physical_element_count")
        object.__setattr__(self, "elements", elements)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def focused_at(self, point, origin=(0.0, 0.0, 0.0)) -> "Aperture":
        from openfield.focusing import FixedFocus

        return self.with_focus(FixedFocus(point=point, origin=origin))

    def with_focus(self, focus) -> "Aperture":
        return self._replace(focus=focus)

    def with_apodization(self, apodization) -> "Aperture":
        return self._replace(apodization=apodization)

    def with_impulse_response(self, impulse_response) -> "Aperture":
        return self._replace(impulse_response=impulse_response)

    def with_excitation(self, excitation) -> "Aperture":
        return self._replace(excitation=excitation)

    def show(self, **kwargs):
        """Display the aperture in 3D using the optional vedo backend."""

        from openfield.visualize.apertures import show_aperture

        return show_aperture(self, **kwargs)

    def show_3d(self, **kwargs):
        """Display the aperture in 3D using the optional vedo backend."""

        return self.show(**kwargs)

    def _replace(self, **changes) -> "Aperture":
        values = {
            "elements": self.elements,
            "physical_element_count": self.physical_element_count,
            "name": self.name,
            "focus": self.focus,
            "apodization": self.apodization,
            "impulse_response": self.impulse_response,
            "excitation": self.excitation,
            "metadata": self.metadata,
        }
        values.update(changes)

        copied = object.__new__(self.__class__)
        for key, value in values.items():
            object.__setattr__(copied, key, value)
        Aperture.__post_init__(copied)
        return copied

    @property
    def centers(self) -> np.ndarray:
        return np.stack([element.center for element in self.elements], axis=0)

    @property
    def normals(self) -> np.ndarray:
        return np.stack([element.normal for element in self.elements], axis=0)

    @property
    def areas(self) -> np.ndarray:
        return np.asarray([element.area for element in self.elements], dtype=np.float64)

    @property
    def physical_indices(self) -> np.ndarray:
        return np.asarray([element.physical_index for element in self.elements], dtype=np.int64)

    @property
    def subelement_indices(self) -> np.ndarray:
        return np.asarray([element.subelement_index for element in self.elements], dtype=np.int64)

    @property
    def physical_centers(self) -> np.ndarray:
        centers = np.zeros((self.physical_element_count, 3), dtype=np.float64)
        weights = np.zeros(self.physical_element_count, dtype=np.float64)
        for element in self.elements:
            centers[element.physical_index] += element.center * element.area
            weights[element.physical_index] += element.area
        return centers / weights[:, None]
