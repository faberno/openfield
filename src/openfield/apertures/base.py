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
    baffle: Any | None = None
    subelement_apodization: Any | None = None
    subelement_delays: Any | None = None
    element_waveforms: Any | None = None
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

    def with_focus_timeline(
        self,
        times,
        points,
        *,
        origin=(0.0, 0.0, 0.0),
        quantization: float | None = None,
    ) -> "Aperture":
        from openfield.focusing import FocusTimeline

        return self.with_focus(
            FocusTimeline(times=times, points=points, origin=origin, quantization=quantization)
        )

    def with_delay_timeline(self, times, delays) -> "Aperture":
        from openfield.focusing import DelayTimeline

        return self.with_focus(DelayTimeline(times=times, values=delays))

    def with_dynamic_focus(
        self,
        *,
        start_time: float,
        direction_zx: float,
        direction_zy: float,
        origin=(0.0, 0.0, 0.0),
        quantization: float | None = None,
    ) -> "Aperture":
        from openfield.focusing import DynamicFocus

        return self.with_focus(
            DynamicFocus(
                start_time=start_time,
                direction_zx=direction_zx,
                direction_zy=direction_zy,
                origin=origin,
                quantization=quantization,
            )
        )

    def with_focus(self, focus) -> "Aperture":
        return self._replace(focus=focus)

    def with_apodization(self, apodization) -> "Aperture":
        return self._replace(apodization=apodization)

    def with_baffle(self, kind: str) -> "Aperture":
        from openfield.apertures import Baffle

        return self._replace(baffle=Baffle(kind=kind))

    def with_subelement_apodization(self, values) -> "Aperture":
        from openfield.apertures import SubElementApodization

        return self._replace(subelement_apodization=SubElementApodization(values))

    def with_subelement_delays(self, values) -> "Aperture":
        from openfield.apertures import SubElementDelays

        return self._replace(subelement_delays=SubElementDelays(values))

    def with_impulse_response(self, impulse_response) -> "Aperture":
        return self._replace(impulse_response=impulse_response)

    def with_excitation(self, excitation) -> "Aperture":
        return self._replace(excitation=excitation)

    def with_element_waveforms(self, waveforms) -> "Aperture":
        from openfield.waveforms import ElementWaveforms

        return self._replace(element_waveforms=ElementWaveforms(waveforms))

    def select_physical_elements(self, indices) -> "Aperture":
        """Return a new aperture containing a subset of physical elements.

        The returned aperture uses contiguous zero-based physical indices while
        storing the original indices in metadata.
        """

        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim == 0:
            indices = indices[None]
        if indices.ndim != 1:
            raise ValueError("indices must be one-dimensional")
        if indices.size == 0:
            raise ValueError("at least one physical element must be selected")
        if len(set(indices.tolist())) != indices.size:
            raise ValueError("indices must not contain duplicates")
        if np.any(indices < 0) or np.any(indices >= self.physical_element_count):
            raise ValueError("indices are outside the aperture element range")

        remap = {int(old): new for new, old in enumerate(indices.tolist())}
        selected_positions = [
            position
            for position, element in enumerate(self.elements)
            if element.physical_index in remap
        ]
        elements = tuple(
            SubElement(
                center=self.elements[position].center,
                normal=self.elements[position].normal,
                area=self.elements[position].area,
                physical_index=remap[self.elements[position].physical_index],
                subelement_index=self.elements[position].subelement_index,
                vertices=self.elements[position].vertices,
            )
            for position in selected_positions
        )
        if not elements:
            raise ValueError("selected physical elements do not contain any subelements")

        apodization = self.apodization
        if apodization is not None and hasattr(apodization, "select_physical_elements"):
            apodization = apodization.select_physical_elements(indices)
        focus = self.focus
        if focus is not None and hasattr(focus, "select_physical_elements"):
            focus = focus.select_physical_elements(indices)
        subelement_apodization = self.subelement_apodization
        if subelement_apodization is not None and hasattr(subelement_apodization, "select_subelements"):
            subelement_apodization = subelement_apodization.select_subelements(selected_positions)
        subelement_delays = self.subelement_delays
        if subelement_delays is not None and hasattr(subelement_delays, "select_subelements"):
            subelement_delays = subelement_delays.select_subelements(selected_positions)
        element_waveforms = self.element_waveforms
        if element_waveforms is not None and hasattr(element_waveforms, "select_physical_elements"):
            element_waveforms = element_waveforms.select_physical_elements(indices)

        metadata = dict(self.metadata)
        metadata["selected_physical_indices"] = indices.copy()
        if "fieldii_focus_centers" in metadata:
            metadata["fieldii_focus_centers"] = np.asarray(
                metadata["fieldii_focus_centers"],
                dtype=np.float64,
            )[indices].copy()
        return self._replace(
            elements=elements,
            physical_element_count=int(indices.size),
            focus=focus,
            apodization=apodization,
            subelement_apodization=subelement_apodization,
            subelement_delays=subelement_delays,
            element_waveforms=element_waveforms,
            metadata=metadata,
        )

    def show(self, **kwargs):
        """Display the aperture in 3D using the optional vedo backend."""

        from openfield.visualize.apertures import show_aperture

        return show_aperture(self, **kwargs)

    def show_3d(self, **kwargs):
        """Display the aperture in 3D using the optional vedo backend."""

        return self.show(**kwargs)

    def to_triangles(self) -> "Aperture":
        """Convert polygonal subelements to a triangular aperture."""

        from openfield.apertures.primitives import TriangleAperture

        triangles = []
        physical_indices = []
        for element in self.elements:
            if element.vertices is None:
                continue
            vertices = np.asarray(element.vertices, dtype=np.float64)
            for vertex_index in range(1, len(vertices) - 1):
                triangles.append([vertices[0], vertices[vertex_index], vertices[vertex_index + 1]])
                physical_indices.append(element.physical_index)
        if not triangles:
            raise ValueError("aperture does not contain polygon vertices to triangulate")
        return TriangleAperture(
            np.asarray(triangles, dtype=np.float64),
            physical_indices=np.asarray(physical_indices, dtype=np.int64),
        )

    def _replace(self, **changes) -> "Aperture":
        values = {
            "elements": self.elements,
            "physical_element_count": self.physical_element_count,
            "name": self.name,
            "focus": self.focus,
            "apodization": self.apodization,
            "impulse_response": self.impulse_response,
            "excitation": self.excitation,
            "baffle": self.baffle,
            "subelement_apodization": self.subelement_apodization,
            "subelement_delays": self.subelement_delays,
            "element_waveforms": self.element_waveforms,
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
