from __future__ import annotations

import numpy as np

from openfield import Medium, Simulation
from openfield.apertures import (
    ConcavePiston,
    ConvexArray,
    ConvexFocusedArray,
    ConvexFocusedMultirowArray,
    FocusedLinearArray,
    FocusedMultirowArray,
    LineBoundedAperture,
    LinearMultirowArray,
    RectangleAperture,
    TriangleAperture,
    TwoDimensionalArray,
)
from python_helpers.io import write_time_response


POINTS = [
    [0.0, 0.0, 30e-3],
    [0.0, 0.0, 40e-3],
    [2e-3, 0.0, 40e-3],
]
FOCUS = [0.0, 0.0, 40e-3]


def run_case(case_name: str) -> None:
    simulation = Simulation(sampling_frequency=100e6, medium=Medium(sound_speed=1540.0))
    aperture = _aperture(case_name)
    response = simulation.spatial_impulse_response(aperture, POINTS)
    write_time_response(case_name, "openfield", response)


def _aperture(case_name: str):
    if case_name == "focused_linear_array_spatial_impulse":
        return FocusedLinearArray(
            elements=6,
            width=0.3e-3,
            height=5e-3,
            kerf=0.03e-3,
            elevation_focus=20e-3,
            subdivisions=(1, 4),
            tessellation="fieldii",
        ).focused_at(FOCUS)

    if case_name == "linear_multirow_array_spatial_impulse":
        return LinearMultirowArray(
            elements_x=4,
            width=0.3e-3,
            elements_y=3,
            heights=[1.0e-3, 1.2e-3, 1.0e-3],
            kerf_x=0.03e-3,
            kerf_y=0.05e-3,
            subdivisions=(1, 2),
        ).focused_at(FOCUS)

    if case_name == "focused_multirow_array_spatial_impulse":
        return FocusedMultirowArray(
            elements_x=4,
            width=0.3e-3,
            elements_y=3,
            heights=[1.0e-3, 1.2e-3, 1.0e-3],
            kerf_x=0.03e-3,
            kerf_y=0.05e-3,
            elevation_focus=20e-3,
            subdivisions=(1, 2),
            tessellation="fieldii",
        ).focused_at(FOCUS)

    if case_name == "convex_array_spatial_impulse":
        return ConvexArray(
            elements=6,
            width=0.3e-3,
            height=5e-3,
            kerf=0.03e-3,
            convex_radius=25e-3,
            subdivisions=(1, 3),
            tessellation="fieldii",
        ).focused_at(FOCUS)

    if case_name == "convex_focused_array_spatial_impulse":
        return ConvexFocusedArray(
            elements=6,
            width=0.3e-3,
            height=5e-3,
            kerf=0.03e-3,
            convex_radius=25e-3,
            elevation_focus=20e-3,
            subdivisions=(1, 4),
            tessellation="fieldii",
        ).focused_at(FOCUS)

    if case_name == "convex_focused_multirow_array_spatial_impulse":
        return ConvexFocusedMultirowArray(
            elements_x=4,
            width=0.3e-3,
            elements_y=3,
            heights=[1.0e-3, 1.2e-3, 1.0e-3],
            kerf_x=0.03e-3,
            kerf_y=0.05e-3,
            convex_radius=25e-3,
            elevation_focus=20e-3,
            subdivisions=(1, 2),
            tessellation="fieldii",
        ).focused_at(FOCUS)

    if case_name == "two_dimensional_array_spatial_impulse":
        enabled = np.array(
            [
                [1, 0, 1],
                [0, 1, 0],
                [1, 1, 0],
                [0, 1, 1],
            ],
            dtype=bool,
        )
        return TwoDimensionalArray(
            elements_x=4,
            elements_y=3,
            width=0.3e-3,
            height=0.4e-3,
            kerf_x=0.03e-3,
            kerf_y=0.04e-3,
            enabled=enabled,
            subdivisions=(2, 2),
        ).focused_at(FOCUS)

    if case_name == "concave_piston_spatial_impulse":
        return ConcavePiston(
            radius=5e-3,
            focal_radius=30e-3,
            element_size=0.5e-3,
            tessellation="cartesian",
        )

    if case_name == "rectangle_aperture_spatial_impulse":
        rect = np.array(
            [
                [
                    1,
                    -0.5e-3,
                    -0.25e-3,
                    0.0,
                    0.0,
                    -0.25e-3,
                    0.0,
                    0.0,
                    0.25e-3,
                    0.0,
                    -0.5e-3,
                    0.25e-3,
                    0.0,
                    1.0,
                    0.5e-3,
                    0.5e-3,
                    -0.25e-3,
                    0.0,
                    0.0,
                ],
                [
                    2,
                    0.1e-3,
                    -0.25e-3,
                    0.0,
                    0.6e-3,
                    -0.25e-3,
                    0.0,
                    0.6e-3,
                    0.25e-3,
                    0.0,
                    0.1e-3,
                    0.25e-3,
                    0.0,
                    1.0,
                    0.5e-3,
                    0.5e-3,
                    0.35e-3,
                    0.0,
                    0.0,
                ],
            ],
            dtype=np.float64,
        )
        return RectangleAperture.from_fieldii_rectangles(rect, focus=FOCUS)

    if case_name == "triangle_aperture_spatial_impulse":
        triangles = np.array(
            [
                [
                    1,
                    -0.5e-3,
                    -0.25e-3,
                    0.0,
                    0.5e-3,
                    -0.25e-3,
                    0.0,
                    0.5e-3,
                    0.25e-3,
                    0.0,
                    1.0,
                ],
                [
                    1,
                    -0.5e-3,
                    -0.25e-3,
                    0.0,
                    0.5e-3,
                    0.25e-3,
                    0.0,
                    -0.5e-3,
                    0.25e-3,
                    0.0,
                    1.0,
                ],
            ],
            dtype=np.float64,
        )
        return TriangleAperture.from_fieldii_triangles(triangles, focus=FOCUS)

    if case_name == "line_bounded_aperture_spatial_impulse":
        lines = np.array(
            [
                [1, 1, 0.0, 1, -0.5e-3, 0],
                [1, 1, 0.0, 1, 0.5e-3, 1],
                [1, 1, 0.0, 0, -0.25e-3, 1],
                [1, 1, 0.0, 0, 0.25e-3, 0],
            ],
            dtype=np.float64,
        )
        return LineBoundedAperture.from_fieldii_lines(
            lines,
            centers=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
            focus=FOCUS,
            bounding_extent=1e-3,
        )

    raise ValueError(f"unknown aperture calc_h case: {case_name}")
