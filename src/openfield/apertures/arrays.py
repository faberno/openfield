from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .base import Aperture
from .elements import SubElement
from .tessellation import quadrilateral_area_normal, rectangular_grid


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
                "fieldii_leading_pad_samples": -1,
                "fieldii_focus_centers": _focused_linear_focus_centers(
                    elements=elements,
                    pitch=width + kerf,
                    elevation_focus=elevation_focus,
                    aperture_half_height=height / 2.0,
                ),
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
                "fieldii_leading_pad_samples": 0,
                "fieldii_trailing_pad_samples": 4,
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
                "fieldii_leading_pad_samples": 0,
                "fieldii_trailing_pad_samples": 4,
                "fieldii_focus_centers": _focused_multirow_focus_centers(
                    elements_x=elements_x,
                    width=width,
                    kerf_x=kerf_x,
                    y_positions=_row_centers(np.asarray(heights, dtype=np.float64), kerf_y),
                    elevation_focus=elevation_focus,
                    aperture_half_height=(
                        float(np.sum(np.asarray(heights, dtype=np.float64)))
                        + kerf_y * (elements_y - 1)
                    )
                    / 2.0,
                ),
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
                "fieldii_trailing_pad_samples": 3,
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
                "fieldii_leading_pad_samples": 0,
                "fieldii_trailing_pad_samples": 4,
                "fieldii_focus_centers": _convex_focus_centers(
                    elements_x=elements,
                    width=width,
                    kerf_x=kerf,
                    y_positions=np.array([0.0], dtype=np.float64),
                    convex_radius=convex_radius,
                    elevation_focus=elevation_focus,
                    aperture_half_height=height / 2.0,
                ),
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
                "fieldii_leading_pad_samples": 0,
                "fieldii_focus_centers": _convex_focus_centers(
                    elements_x=elements_x,
                    width=width,
                    kerf_x=kerf_x,
                    y_positions=_row_centers(np.asarray(heights, dtype=np.float64), kerf_y),
                    convex_radius=convex_radius,
                    elevation_focus=elevation_focus,
                    aperture_half_height=(
                        float(np.sum(np.asarray(heights, dtype=np.float64)))
                        + kerf_y * (elements_y - 1)
                    )
                    / 2.0,
                ),
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

        for iy, y in enumerate(y_positions):
            for ix, x in enumerate(x_positions):
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
                "fieldii_leading_pad_samples": 0,
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
    aperture_half_height = float(np.sum(heights) + kerf_y * (len(heights) - 1)) / 2.0

    if surface_kind == "elevation_focused":
        _validate_positive(elevation_focus, "elevation_focus")
        _validate_elevation_extent(heights, kerf_y, elevation_focus)
    elif surface_kind != "flat":
        raise ValueError(f"unsupported surface_kind: {surface_kind}")

    subelements = []
    subelement_index = 0
    for iy, y in enumerate(y_positions):
        for ix, x in enumerate(x_positions):
            physical_index = iy * elements_x + ix
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
                tessellated = _elevation_focused_grid(
                    x_center=x,
                    y_center=y,
                    width=width,
                    height=float(heights[iy]),
                    subdivisions=subdivisions,
                    physical_index=physical_index,
                    start_subelement_index=subelement_index,
                    elevation_focus=elevation_focus,
                    aperture_half_height=aperture_half_height,
                    angular_subdivision=elements_y == 1,
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
    width_angle = _convex_fieldii_segment_angle(width, convex_radius)
    kerf_angle = _convex_fieldii_segment_angle(kerf_x, convex_radius) if kerf_x else 0.0
    angle_positions = _centered_positions(elements_x, width_angle + kerf_angle)
    aperture_half_height = float(np.sum(heights) + kerf_y * (len(heights) - 1)) / 2.0

    if elevation_focus is not None:
        _validate_positive(elevation_focus, "elevation_focus")
        _validate_elevation_extent(heights, kerf_y, elevation_focus)

    subelements = []
    subelement_index = 0
    for iy, y in enumerate(y_positions):
        for ix, angle in enumerate(angle_positions):
            physical_index = iy * elements_x + ix
            tessellated = _convex_fieldii_grid(
                angle_center=float(angle),
                y_center=float(y),
                width_angle=width_angle,
                height=float(heights[iy]),
                convex_radius=convex_radius,
                subdivisions=subdivisions,
                physical_index=physical_index,
                start_subelement_index=subelement_index,
                elevation_focus=elevation_focus,
                aperture_half_height=aperture_half_height,
                angular_elevation_subdivision=elements_y == 1,
            )
            subelements.extend(tessellated)
            subelement_index += len(tessellated)
    return tuple(subelements)


def _convex_fieldii_grid(
    *,
    angle_center: float,
    y_center: float,
    width_angle: float,
    height: float,
    convex_radius: float,
    subdivisions: tuple[int, int],
    physical_index: int,
    start_subelement_index: int,
    elevation_focus: float | None,
    aperture_half_height: float,
    angular_elevation_subdivision: bool,
) -> tuple[SubElement, ...]:
    sub_x, sub_y = subdivisions
    angle_edges = (
        angle_center
        - width_angle / 2.0
        + np.arange(sub_x + 1, dtype=np.float64) * (width_angle / sub_x)
    )

    y_lower = y_center - height / 2.0
    y_upper = y_center + height / 2.0
    if elevation_focus is not None and angular_elevation_subdivision:
        theta_edges = np.linspace(
            np.arcsin(y_lower / elevation_focus),
            np.arcsin(y_upper / elevation_focus),
            sub_y + 1,
        )
        y_edges = elevation_focus * np.sin(theta_edges)
    else:
        y_edges = np.linspace(y_lower, y_upper, sub_y + 1)

    elements: list[SubElement] = []
    subelement_index = start_subelement_index
    for iy in range(sub_y):
        y0 = float(y_edges[iy])
        y1 = float(y_edges[iy + 1])
        y_mid = 0.5 * (y0 + y1)
        slope_yz = 0.0

        for ix in range(sub_x):
            angle0 = float(angle_edges[ix])
            angle1 = float(angle_edges[ix + 1])
            angle_mid = 0.5 * (angle0 + angle1)
            vertices = np.array(
                [
                    _convex_surface_from_angle(
                        angle0,
                        y0,
                        convex_radius,
                        elevation_focus=elevation_focus,
                        aperture_half_height=aperture_half_height,
                    ),
                    _convex_surface_from_angle(
                        angle1,
                        y0,
                        convex_radius,
                        elevation_focus=elevation_focus,
                        aperture_half_height=aperture_half_height,
                    ),
                    _convex_surface_from_angle(
                        angle1,
                        y1,
                        convex_radius,
                        elevation_focus=elevation_focus,
                        aperture_half_height=aperture_half_height,
                    ),
                    _convex_surface_from_angle(
                        angle0,
                        y1,
                        convex_radius,
                        elevation_focus=elevation_focus,
                        aperture_half_height=aperture_half_height,
                    ),
                ],
                dtype=np.float64,
            )
            if elevation_focus is not None:
                slope_yz = (vertices[3, 2] - vertices[0, 2]) / (vertices[3, 1] - vertices[0, 1])
            width = float(np.linalg.norm(vertices[1] - vertices[0]))
            height = float(np.linalg.norm((vertices[3] - vertices[0])[[1, 2]]))
            area = width * height
            normal = _convex_fieldii_normal(angle_mid, slope_yz)
            elements.append(
                SubElement(
                    center=_convex_surface_from_angle(
                        angle_mid,
                        y_mid,
                        convex_radius,
                        elevation_focus=elevation_focus,
                        aperture_half_height=aperture_half_height,
                    ),
                    normal=normal,
                    area=area,
                    physical_index=physical_index,
                    subelement_index=subelement_index,
                    vertices=vertices,
                )
            )
            subelement_index += 1
    return tuple(elements)


def _elevation_focused_grid(
    *,
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    subdivisions: tuple[int, int],
    physical_index: int,
    start_subelement_index: int,
    elevation_focus: float,
    aperture_half_height: float,
    angular_subdivision: bool,
) -> tuple[SubElement, ...]:
    sub_x, sub_y = subdivisions
    dx = width / sub_x
    x_edges = x_center - width / 2.0 + np.arange(sub_x + 1, dtype=np.float64) * dx

    y_lower = y_center - height / 2.0
    y_upper = y_center + height / 2.0
    if angular_subdivision:
        theta_edges = np.linspace(
            np.arcsin(y_lower / elevation_focus),
            np.arcsin(y_upper / elevation_focus),
            sub_y + 1,
        )
        y_edges = elevation_focus * np.sin(theta_edges)
    else:
        y_edges = np.linspace(y_lower, y_upper, sub_y + 1)

    elements: list[SubElement] = []
    subelement_index = start_subelement_index
    for iy in range(sub_y):
        y0 = float(y_edges[iy])
        y1 = float(y_edges[iy + 1])
        y_mid = 0.5 * (y0 + y1)
        z0 = _elevation_focused_z_fieldii(y0, elevation_focus, aperture_half_height)
        z1 = _elevation_focused_z_fieldii(y1, elevation_focus, aperture_half_height)
        normal = _elevation_focused_normal_fieldii((z1 - z0) / (y1 - y0))
        for ix in range(sub_x):
            x0 = float(x_edges[ix])
            x1 = float(x_edges[ix + 1])
            x_mid = 0.5 * (x0 + x1)
            vertices = np.array(
                [
                    _elevation_focused_surface_fieldii(x0, y0, elevation_focus, aperture_half_height),
                    _elevation_focused_surface_fieldii(x1, y0, elevation_focus, aperture_half_height),
                    _elevation_focused_surface_fieldii(x1, y1, elevation_focus, aperture_half_height),
                    _elevation_focused_surface_fieldii(x0, y1, elevation_focus, aperture_half_height),
                ],
                dtype=np.float64,
            )
            area, _ = quadrilateral_area_normal(vertices)
            elements.append(
                SubElement(
                    center=_elevation_focused_surface_fieldii(
                        x_mid,
                        y_mid,
                        elevation_focus,
                        aperture_half_height,
                    ),
                    normal=normal,
                    area=area,
                    physical_index=physical_index,
                    subelement_index=subelement_index,
                    vertices=vertices,
                )
            )
            subelement_index += 1
    return tuple(elements)


def _elevation_focused_surface_fieldii(
    x: float,
    y: float,
    elevation_focus: float,
    aperture_half_height: float,
) -> np.ndarray:
    z = _elevation_focused_z_fieldii(y, elevation_focus, aperture_half_height)
    return np.array([x, y, z], dtype=np.float64)


def _elevation_focused_z_fieldii(
    y: float,
    elevation_focus: float,
    aperture_half_height: float,
) -> float:
    reference = np.sqrt(elevation_focus * elevation_focus - aperture_half_height * aperture_half_height)
    return float(reference - np.sqrt(elevation_focus * elevation_focus - y * y))


def _elevation_focused_normal_fieldii(slope_yz: float) -> np.ndarray:
    raw = np.array(
        [
            0.0,
            slope_yz,
            1.0,
        ],
        dtype=np.float64,
    )
    return raw / np.linalg.norm(raw)


def _focused_linear_focus_centers(
    *,
    elements: int,
    pitch: float,
    elevation_focus: float,
    aperture_half_height: float,
) -> np.ndarray:
    x_positions = _centered_positions(elements, pitch)
    z = -_elevation_focused_z_fieldii(0.0, elevation_focus, aperture_half_height)
    return np.column_stack(
        [
            x_positions,
            np.zeros(elements, dtype=np.float64),
            np.full(elements, z, dtype=np.float64),
        ]
    )


def _focused_multirow_focus_centers(
    *,
    elements_x: int,
    width: float,
    kerf_x: float,
    y_positions: np.ndarray,
    elevation_focus: float,
    aperture_half_height: float,
) -> np.ndarray:
    x_positions = _centered_positions(elements_x, width + kerf_x)
    centers = []
    for y in y_positions:
        z = _elevation_focused_z_fieldii(float(y), elevation_focus, aperture_half_height)
        for x in x_positions:
            centers.append([float(x), float(y), z])
    return np.asarray(centers, dtype=np.float64)


def _convex_focus_centers(
    *,
    elements_x: int,
    width: float,
    kerf_x: float,
    y_positions: np.ndarray,
    convex_radius: float,
    elevation_focus: float,
    aperture_half_height: float,
) -> np.ndarray:
    width_angle = _convex_fieldii_segment_angle(width, convex_radius)
    kerf_angle = _convex_fieldii_segment_angle(kerf_x, convex_radius) if kerf_x else 0.0
    angle_positions = _centered_positions(elements_x, width_angle + kerf_angle)
    centers = []
    for y in y_positions:
        for angle in angle_positions:
            centers.append(
                _convex_surface_from_angle(
                    float(angle),
                    float(y),
                    convex_radius,
                    elevation_focus=elevation_focus,
                    aperture_half_height=aperture_half_height,
                )
            )
    return np.asarray(centers, dtype=np.float64)


def _convex_fieldii_segment_angle(length: float, convex_radius: float) -> float:
    if length == 0.0:
        return 0.0
    chord = length / np.cos(length / (2.0 * convex_radius))
    ratio = chord / (2.0 * convex_radius)
    if ratio >= 1.0:
        raise ValueError("convex segment length is too large for the chosen radius")
    return float(2.0 * np.arcsin(ratio))


def _convex_surface_from_angle(
    angle: float,
    y: float,
    convex_radius: float,
    *,
    elevation_focus: float | None,
    aperture_half_height: float,
) -> np.ndarray:
    x = convex_radius * np.sin(angle)
    z = convex_radius * (np.cos(angle) - 1.0)
    if elevation_focus is not None:
        elevation_z = _elevation_focused_z_fieldii(y, elevation_focus, aperture_half_height)
        radius = convex_radius + elevation_z
        x = radius * np.sin(angle)
        z = radius * np.cos(angle) - convex_radius
    return np.array([x, y, z], dtype=np.float64)


def _convex_fieldii_normal(angle: float, slope_yz: float) -> np.ndarray:
    raw = np.array(
        [
            np.tan(angle),
            slope_yz,
            1.0,
        ],
        dtype=np.float64,
    )
    return raw / np.linalg.norm(raw)


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
