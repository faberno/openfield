from __future__ import annotations

import numpy as np

from openfield.apertures import (
    ConcavePiston,
    ConvexArray,
    ConvexFocusedArray,
    ConvexFocusedMultirowArray,
    FocusedLinearArray,
    FocusedMultirowArray,
)
from python_helpers.io import write_geometry


FOCUS = [0.0, 0.0, 40e-3]


def run_case(case_name: str) -> None:
    aperture = _aperture(case_name)
    write_geometry(case_name, "openfield", aperture)


def _aperture(case_name: str):
    if case_name == "focused_linear_array_geometry":
        return FocusedLinearArray(
            elements=6,
            width=0.3e-3,
            height=5e-3,
            kerf=0.03e-3,
            elevation_focus=20e-3,
            subdivisions=(1, 4),
            tessellation="fieldii",
        ).focused_at(FOCUS)

    if case_name == "focused_multirow_array_geometry":
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

    if case_name == "convex_array_geometry":
        return ConvexArray(
            elements=6,
            width=0.3e-3,
            height=5e-3,
            kerf=0.03e-3,
            convex_radius=25e-3,
            subdivisions=(1, 3),
            tessellation="fieldii",
        ).focused_at(FOCUS)

    if case_name == "convex_focused_array_geometry":
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

    if case_name == "convex_focused_multirow_array_geometry":
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

    if case_name == "concave_piston_geometry":
        return ConcavePiston(
            radius=5e-3,
            focal_radius=30e-3,
            element_size=0.5e-3,
            tessellation="cartesian",
        )

    raise ValueError(f"unknown aperture geometry case: {case_name}")
