from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .base import Aperture
from .tessellation import mapped_rectangular_grid, rectangular_grid


class LinearArray(Aperture):
    """Flat linear array with rectangular physical elements."""

    def __init__(
        self,
        elements: int,
        width: float,
        height: float,
        kerf: float,
        subdivisions: tuple[int, int] = (1, 1),
    ):
        subelements = _linear_multirow_subelements(
            elements_x=elements,
            width=width,
            elements_y=1,
            heights=[height],
            kerf_x=kerf,
            kerf_y=0.0,
            subdivisions=subdivisions,
            surface_kind="flat",
        )
        super().__init__(
            elements=subelements,
            physical_element_count=elements,
            name="linear_array",
            metadata={
                "elements": elements,
                "width": width,
                "height": height,
                "kerf": kerf,
                "subdivisions": subdivisions,
            },
        )


class FocusedLinearArray(Aperture):
    """Linear array with mechanical elevation focus."""

    def __init__(
        self,
        elements: int,
        width: float,
        height: float,
        kerf: float,
        elevation_focus: float,
        subdivisions: tuple[int, int] = (1, 1),
    ):
        subelements = _linear_multirow_subelements(
            elements_x=elements,
            width=width,
            elements_y=1,
            heights=[height],
            kerf_x=kerf,
            kerf_y=0.0,
            subdivisions=subdivisions,
            surface_kind="elevation_focused",
            elevation_focus=elevation_focus,
        )
        super().__init__(
            elements=subelements,
            physical_element_count=elements,
            name="focused_linear_array",
            metadata={
                "elements": elements,
                "width": width,
                "height": height,
                "kerf": kerf,
                "elevation_focus": elevation_focus,
                "subdivisions": subdivisions,
            },
        )


class LinearMultirowArray(Aperture):
    """Flat 1.5D linear array with multiple rows in elevation."""

    def __init__(
        self,
        elements_x: int,
        width: float,
        elements_y: int,
        heights: Sequence[float],
        kerf_x: float,
        kerf_y: float,
        subdivisions: tuple[int, int] = (1, 1),
    ):
        subelements = _linear_multirow_subelements(
            elements_x=elements_x,
            width=width,
            elements_y=elements_y,
            heights=heights,
            kerf_x=kerf_x,
            kerf_y=kerf_y,
            subdivisions=subdivisions,
            surface_kind="flat",
        )
        super().__init__(
            elements=subelements,
            physical_element_count=elements_x * elements_y,
            name="linear_multirow_array",
            metadata={
                "elements_x": elements_x,
                "width": width,
                "elements_y": elements_y,
                "heights": list(heights),
                "kerf_x": kerf_x,
                "kerf_y": kerf_y,
                "subdivisions": subdivisions,
            },
        )


class FocusedMultirowArray(Aperture):
    """1.5D linear array with mechanical elevation focus."""

    def __init__(
        self,
        elements_x: int,
        width: float,
        elements_y: int,
        heights: Sequence[float],
        kerf_x: float,
        kerf_y: float,
        elevation_focus: float,
        subdivisions: tuple[int, int] = (1, 1),
    ):
        subelements = _linear_multirow_subelements(
            elements_x=elements_x,
            width=width,
            elements_y=elements_y,
            heights=heights,
            kerf_x=kerf_x,
            kerf_y=kerf_y,
            subdivisions=subdivisions,
            surface_kind="elevation_focused",
            elevation_focus=elevation_focus,
        )
        super().__init__(
            elements=subelements,
            physical_element_count=elements_x * elements_y,
            name="focused_multirow_array",
            metadata={
                "elements_x": elements_x,
                "width": width,
                "elements_y": elements_y,
                "heights": list(heights),
                "kerf_x": kerf_x,
                "kerf_y": kerf_y,
                "elevation_focus": elevation_focus,
                "subdivisions": subdivisions,
            },
        )


class ConvexArray(Aperture):
    """Convex linear array curved in azimuth."""

    def __init__(
        self,
        elements: int,
        width: float,
        height: float,
        kerf: float,
        convex_radius: float,
        subdivisions: tuple[int, int] = (1, 1),
    ):
        subelements = _convex_multirow_subelements(
            elements_x=elements,
            width=width,
            elements_y=1,
            heights=[height],
            kerf_x=kerf,
            kerf_y=0.0,
            convex_radius=convex_radius,
            subdivisions=subdivisions,
        )
        super().__init__(
            elements=subelements,
            physical_element_count=elements,
            name="convex_array",
            metadata={
                "elements": elements,
                "width": width,
                "height": height,
                "kerf": kerf,
                "convex_radius": convex_radius,
                "subdivisions": subdivisions,
            },
        )


class ConvexFocusedArray(Aperture):
    """Convex array with mechanical elevation focus."""

    def __init__(
        self,
        elements: int,
        width: float,
        height: float,
        kerf: float,
        convex_radius: float,
        elevation_focus: float,
        subdivisions: tuple[int, int] = (1, 1),
    ):
        subelements = _convex_multirow_subelements(
            elements_x=elements,
            width=width,
            elements_y=1,
            heights=[height],
            kerf_x=kerf,
            kerf_y=0.0,
            convex_radius=convex_radius,
            elevation_focus=elevation_focus,
            subdivisions=subdivisions,
        )
        super().__init__(
            elements=subelements,
            physical_element_count=elements,
            name="convex_focused_array",
            metadata={
                "elements": elements,
                "width": width,
                "height": height,
                "kerf": kerf,
                "convex_radius": convex_radius,
                "elevation_focus": elevation_focus,
                "subdivisions": subdivisions,
            },
        )


class ConvexFocusedMultirowArray(Aperture):
    """Convex multirow array with mechanical elevation focus."""

    def __init__(
        self,
        elements_x: int,
        width: float,
        elements_y: int,
        heights: Sequence[float],
        kerf_x: float,
        kerf_y: float,
        convex_radius: float,
        elevation_focus: float,
        subdivisions: tuple[int, int] = (1, 1),
    ):
        subelements = _convex_multirow_subelements(
            elements_x=elements_x,
            width=width,
            elements_y=elements_y,
            heights=heights,
            kerf_x=kerf_x,
            kerf_y=kerf_y,
            convex_radius=convex_radius,
            elevation_focus=elevation_focus,
            subdivisions=subdivisions,
        )
        super().__init__(
            elements=subelements,
            physical_element_count=elements_x * elements_y,
            name="convex_focused_multirow_array",
            metadata={
                "elements_x": elements_x,
                "width": width,
                "elements_y": elements_y,
                "heights": list(heights),
                "kerf_x": kerf_x,
                "kerf_y": kerf_y,
                "convex_radius": convex_radius,
                "elevation_focus": elevation_focus,
                "subdivisions": subdivisions,
            },
        )


class TwoDimensionalArray(Aperture):
    """Flat sparse 2D matrix array."""

    def __init__(
        self,
        elements_x: int,
        elements_y: int,
        width: float,
        height: float,
        kerf_x: float,
        kerf_y: float,
        enabled=None,
        subdivisions: tuple[int, int] = (1, 1),
    ):
        _validate_positive_count(elements_x, "elements_x")
        _validate_positive_count(elements_y, "elements_y")
        _validate_positive(width, "width")
        _validate_positive(height, "height")
        _validate_non_negative(kerf_x, "kerf_x")
        _validate_non_negative(kerf_y, "kerf_y")
        _validate_subdivisions(subdivisions)

        if enabled is None:
            enabled = np.ones((elements_x, elements_y), dtype=bool)
        else:
            enabled = np.asarray(enabled, dtype=bool)
        if enabled.shape != (elements_x, elements_y):
            raise ValueError("enabled must have shape (elements_x, elements_y)")
        if not np.any(enabled):
            raise ValueError("at least one 2D array element must be enabled")

        pitch_x = width + kerf_x
        pitch_y = height + kerf_y
        x_positions = _centered_positions(elements_x, pitch_x)
        y_positions = _centered_positions(elements_y, pitch_y)
        subelements = []
        subelement_index = 0
        physical_index = 0

        for ix, x in enumerate(x_positions):
            for iy, y in enumerate(y_positions):
                if not enabled[ix, iy]:
                    continue
                tessellated = rectangular_grid(
                    center=(x, y, 0.0),
                    width=width,
                    height=height,
                    subdivisions=subdivisions,
                    physical_index=physical_index,
                    start_subelement_index=subelement_index,
                )
                subelements.extend(tessellated)
                subelement_index += len(tessellated)
                physical_index += 1

        super().__init__(
            elements=tuple(subelements),
            physical_element_count=physical_index,
            name="two_dimensional_array",
            metadata={
                "elements_x": elements_x,
                "elements_y": elements_y,
                "width": width,
                "height": height,
                "kerf_x": kerf_x,
                "kerf_y": kerf_y,
                "enabled": enabled.copy(),
                "subdivisions": subdivisions,
            },
        )


Array2D = TwoDimensionalArray


def _linear_multirow_subelements(
    *,
    elements_x: int,
    width: float,
    elements_y: int,
    heights: Sequence[float],
    kerf_x: float,
    kerf_y: float,
    subdivisions: tuple[int, int],
    surface_kind: str,
    elevation_focus: float | None = None,
):
    _validate_array_parameters(elements_x, width, elements_y, heights, kerf_x, kerf_y, subdivisions)
    heights = np.asarray(heights, dtype=np.float64)
    x_positions = _centered_positions(elements_x, width + kerf_x)
    y_positions = _row_centers(heights, kerf_y)

    if surface_kind == "elevation_focused":
        _validate_positive(elevation_focus, "elevation_focus")
        _validate_elevation_extent(heights, kerf_y, elevation_focus)
    elif surface_kind != "flat":
        raise ValueError(f"unsupported surface_kind: {surface_kind}")

    subelements = []
    subelement_index = 0
    for ix, x in enumerate(x_positions):
        for iy, y in enumerate(y_positions):
            physical_index = ix * elements_y + iy
            if surface_kind == "flat":
                tessellated = rectangular_grid(
                    center=(x, y, 0.0),
                    width=width,
                    height=float(heights[iy]),
                    subdivisions=subdivisions,
                    physical_index=physical_index,
                    start_subelement_index=subelement_index,
                )
            else:
                tessellated = mapped_rectangular_grid(
                    surface=lambda local_x, local_y, x=x, y=y: _elevation_focused_surface(
                        x + local_x,
                        y + local_y,
                        elevation_focus,
                    ),
                    width=width,
                    height=float(heights[iy]),
                    subdivisions=subdivisions,
                    physical_index=physical_index,
                    start_subelement_index=subelement_index,
                )
            subelements.extend(tessellated)
            subelement_index += len(tessellated)
    return tuple(subelements)


def _convex_multirow_subelements(
    *,
    elements_x: int,
    width: float,
    elements_y: int,
    heights: Sequence[float],
    kerf_x: float,
    kerf_y: float,
    convex_radius: float,
    subdivisions: tuple[int, int],
    elevation_focus: float | None = None,
):
    _validate_array_parameters(elements_x, width, elements_y, heights, kerf_x, kerf_y, subdivisions)
    _validate_positive(convex_radius, "convex_radius")
    _validate_convex_extent(elements_x, width, kerf_x, convex_radius)
    heights = np.asarray(heights, dtype=np.float64)
    y_positions = _row_centers(heights, kerf_y)
    arc_positions = _centered_positions(elements_x, width + kerf_x)

    if elevation_focus is not None:
        _validate_positive(elevation_focus, "elevation_focus")
        _validate_elevation_extent(heights, kerf_y, elevation_focus)

    subelements = []
    subelement_index = 0
    for ix, arc in enumerate(arc_positions):
        for iy, y in enumerate(y_positions):
            physical_index = ix * elements_y + iy
            expected_normal = _convex_expected_normal(arc, convex_radius)
            tessellated = mapped_rectangular_grid(
                surface=lambda local_x, local_y, arc=arc, y=y: _convex_surface(
                    arc + local_x,
                    y + local_y,
                    convex_radius,
                    elevation_focus=elevation_focus,
                ),
                width=width,
                height=float(heights[iy]),
                subdivisions=subdivisions,
                physical_index=physical_index,
                start_subelement_index=subelement_index,
                expected_normal=expected_normal,
            )
            subelements.extend(tessellated)
            subelement_index += len(tessellated)
    return tuple(subelements)


def _elevation_focused_surface(x: float, y: float, elevation_focus: float) -> np.ndarray:
    z = elevation_focus - np.sqrt(elevation_focus * elevation_focus - y * y)
    return np.array([x, y, z], dtype=np.float64)


def _convex_surface(
    arc_position: float,
    y: float,
    convex_radius: float,
    *,
    elevation_focus: float | None,
) -> np.ndarray:
    theta = arc_position / convex_radius
    x = convex_radius * np.sin(theta)
    z = convex_radius * (np.cos(theta) - 1.0)
    if elevation_focus is not None:
        z += elevation_focus - np.sqrt(elevation_focus * elevation_focus - y * y)
    return np.array([x, y, z], dtype=np.float64)


def _convex_expected_normal(arc_position: float, convex_radius: float) -> np.ndarray:
    theta = arc_position / convex_radius
    return np.array([np.sin(theta), 0.0, np.cos(theta)], dtype=np.float64)


def _validate_array_parameters(
    elements_x: int,
    width: float,
    elements_y: int,
    heights: Sequence[float],
    kerf_x: float,
    kerf_y: float,
    subdivisions: tuple[int, int],
) -> None:
    _validate_positive_count(elements_x, "elements_x")
    _validate_positive_count(elements_y, "elements_y")
    _validate_positive(width, "width")
    _validate_non_negative(kerf_x, "kerf_x")
    _validate_non_negative(kerf_y, "kerf_y")
    _validate_subdivisions(subdivisions)
    heights = np.asarray(heights, dtype=np.float64)
    if heights.shape != (elements_y,):
        raise ValueError("heights must contain one value per row")
    if np.any(heights <= 0):
        raise ValueError("all heights must be positive")


def _validate_positive_count(value: int, name: str) -> None:
    if value < 1:
        raise ValueError(f"{name} must be at least one")


def _validate_positive(value: float | None, name: str) -> None:
    if value is None or value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_non_negative(value: float, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _validate_subdivisions(subdivisions: tuple[int, int]) -> None:
    if len(subdivisions) != 2 or subdivisions[0] < 1 or subdivisions[1] < 1:
        raise ValueError("subdivisions must contain two positive integers")


def _validate_elevation_extent(heights: np.ndarray, kerf_y: float, elevation_focus: float) -> None:
    total_height = float(np.sum(heights) + kerf_y * (len(heights) - 1))
    if total_height / 2 >= elevation_focus:
        raise ValueError("elevation_focus must be larger than half the elevation aperture")


def _validate_convex_extent(elements_x: int, width: float, kerf_x: float, convex_radius: float) -> None:
    total_arc = elements_x * width + (elements_x - 1) * kerf_x
    if total_arc / 2 >= np.pi * convex_radius / 2:
        raise ValueError("convex array angular aperture must be smaller than 180 degrees")


def _centered_positions(count: int, pitch: float) -> np.ndarray:
    return (np.arange(count, dtype=np.float64) - (count - 1) / 2) * pitch


def _row_centers(heights: np.ndarray, kerf_y: float) -> np.ndarray:
    total_height = float(np.sum(heights) + kerf_y * (len(heights) - 1))
    current = -total_height / 2
    centers = []
    for height in heights:
        centers.append(current + height / 2)
        current += height + kerf_y
    return np.asarray(centers, dtype=np.float64)
