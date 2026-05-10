from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "validation"))

from openfield import Medium, Simulation
from openfield.physics.spatial_impulse import (
    _fieldii_rectangle_projection_half_widths,
    _projected_rectangle_fraction,
    _sample_weight,
)
from python_helpers.aperture_calc_h_cases import POINTS, _aperture


KERNEL_RESULTS = ROOT / "validation" / "results" / "kernel_modes"
CASE_MAP = {
    "convex_focused_array": "convex_focused_array_spatial_impulse",
    "convex_focused_multirow_array": "convex_focused_multirow_array_spatial_impulse",
    "concave_piston": "concave_piston_spatial_impulse",
}


def main() -> None:
    paths = sorted(KERNEL_RESULTS.glob("*_isolated_rect*"))
    if not paths:
        print(
            "No isolated curved-rectangle diagnostics found. Run "
            "run_kernel_mode_diagnostics(repo_root, true) from MATLAB first.",
        )
        return

    simulation = Simulation(sampling_frequency=100e6, medium=Medium(sound_speed=1540.0))
    by_case: dict[str, list[dict[str, object]]] = {}
    for path in paths:
        case_key, rect_index = _parse_isolate_name(path.name)
        validation_case = CASE_MAP[case_key]
        aperture = _aperture(validation_case)
        weights = np.zeros(len(aperture.elements), dtype=np.float64)
        weights[rect_index] = 1.0
        aperture = aperture.with_subelement_apodization(weights)
        response = simulation.spatial_impulse_response(aperture, POINTS)
        openfield = response.samples
        fieldii = _read_samples(path / "fieldii" / "samples.csv")
        count = min(len(openfield), len(fieldii))
        difference = openfield[:count] - fieldii[:count]
        max_index = np.unravel_index(np.argmax(np.abs(difference)), difference.shape)
        element = aperture.elements[rect_index]
        point = np.asarray(POINTS[max_index[1]], dtype=np.float64)
        direction = point - element.center
        direction /= np.linalg.norm(direction)
        half_x, half_y = _fieldii_rectangle_projection_half_widths(
            element,
            direction=direction,
            sound_speed=simulation.medium.sound_speed,
        )
        by_case.setdefault(case_key, []).append(
            {
                "max_abs": float(np.max(np.abs(difference))),
                "rect_index": rect_index,
                "point_index": int(max_index[1]),
                "sample_index": int(max_index[0]),
                "sample_time": float(response.start_time + max_index[0] / response.sampling_frequency),
                "sum_diff": np.sum(difference, axis=0),
                "best_shift": _best_integer_shift(openfield[:count, max_index[1]], fieldii[:count, max_index[1]]),
                "center_radius": float(np.linalg.norm(element.center[:2])),
                "normal": element.normal.copy(),
                "half_widths": np.array([half_x, half_y], dtype=np.float64),
                "edge_z": np.array(
                    [
                        element.vertices[1, 2] - element.vertices[0, 2],
                        element.vertices[3, 2] - element.vertices[0, 2],
                    ],
                    dtype=np.float64,
                ),
            },
        )

    for case_key, rows in by_case.items():
        rows.sort(key=lambda row: float(row["max_abs"]), reverse=True)
        max_values = [float(row["max_abs"]) for row in rows]
        print(f"{case_key}: {len(rows)} isolated rectangles")
        print(f"  worst max_abs={float(rows[0]['max_abs']):.6g} at rect {int(rows[0]['rect_index']) + 1}")
        print(f"  median max_abs={np.median(max_values):.6g}")
        for row in rows[:5]:
            fit = _fit_half_width_ratios(simulation, case_key, int(row["rect_index"]), int(row["point_index"]))
            print(
                f"  rect {int(row['rect_index']) + 1:03d}: "
                f"max_abs={float(row['max_abs']):.6g}, "
                f"point={int(row['point_index'])}, sample={int(row['sample_index'])}, "
                f"best_shift={row['best_shift']}, "
                f"fit={fit}, "
                f"center_radius={float(row['center_radius']):.6g}, "
                f"half_widths={row['half_widths']}, "
                f"edge_z={row['edge_z']}, "
                f"normal={row['normal']}, "
                f"sum_diff={row['sum_diff']}",
            )


def _parse_isolate_name(name: str) -> tuple[str, int]:
    case_key, rect = name.split("_isolated_rect")
    return case_key, int(rect) - 1


def _read_samples(path: Path) -> np.ndarray:
    samples = np.loadtxt(path, delimiter=",")
    if samples.ndim == 1:
        samples = samples[:, None]
    return samples


def _fit_half_width_ratios(
    simulation: Simulation,
    case_key: str,
    rect_index: int,
    point_index: int,
) -> tuple[float, float, float]:
    validation_case = CASE_MAP[case_key]
    aperture = _aperture(validation_case)
    weights = np.zeros(len(aperture.elements), dtype=np.float64)
    weights[rect_index] = 1.0
    aperture = aperture.with_subelement_apodization(weights)
    response = simulation.spatial_impulse_response(aperture, POINTS)
    fieldii = _read_samples(
        KERNEL_RESULTS / f"{case_key}_isolated_rect{rect_index + 1:03d}" / "fieldii" / "samples.csv",
    )
    target = fieldii[: response.sample_count, point_index]

    point = np.asarray(POINTS[point_index], dtype=np.float64)
    element = aperture.elements[rect_index]
    distance_vector = point - element.center
    distance = float(np.linalg.norm(distance_vector))
    direction = distance_vector / distance
    base_half_x, base_half_y = _fieldii_rectangle_projection_half_widths(
        element,
        direction=direction,
        sound_speed=simulation.medium.sound_speed,
    )
    physical_delays = np.zeros(aperture.physical_element_count, dtype=np.float64)
    if aperture.focus is not None:
        physical_delays = aperture.focus.delays(aperture, simulation.medium.sound_speed, time=0.0)
    propagation_delay = float(physical_delays[element.physical_index])
    arrival_time = distance / simulation.medium.sound_speed + propagation_delay

    if case_key == "concave_piston":
        ratio_values = np.linspace(0.97, 1.06, 46)
    else:
        ratio_values = np.linspace(0.995, 1.005, 41)

    best = (np.inf, 1.0, 1.0)
    for ratio_x in ratio_values:
        half_x = base_half_x * float(ratio_x)
        for ratio_y in ratio_values:
            half_y = base_half_y * float(ratio_y)
            predicted = _rectangle_samples_for_half_widths(
                response=response,
                element=element,
                arrival_time=arrival_time,
                propagation_delay=propagation_delay,
                half_x=half_x,
                half_y=half_y,
                sound_speed=simulation.medium.sound_speed,
                sampling_frequency=simulation.sampling_frequency,
            )
            error = float(np.max(np.abs(predicted - target)))
            if error < best[0]:
                best = (error, float(ratio_x), float(ratio_y))
    return best[1], best[2], best[0]


def _rectangle_samples_for_half_widths(
    *,
    response,
    element,
    arrival_time: float,
    propagation_delay: float,
    half_x: float,
    half_y: float,
    sound_speed: float,
    sampling_frequency: float,
) -> np.ndarray:
    output = np.zeros(response.sample_count, dtype=np.float64)
    dt = 1.0 / sampling_frequency
    support_start = arrival_time - half_x - half_y
    support_end = arrival_time + half_x + half_y
    first = int(np.floor((support_start - response.start_time) * sampling_frequency - 0.5)) - 1
    last = int(np.ceil((support_end - response.start_time) * sampling_frequency + 0.5)) + 1
    for index in range(first, last + 1):
        if not 0 <= index < output.shape[0]:
            continue
        sample_time = response.start_time + index * dt
        fraction = _projected_rectangle_fraction(
            sample_time - 0.5 * dt - arrival_time,
            sample_time + 0.5 * dt - arrival_time,
            half_x,
            half_y,
        )
        if fraction <= 0.0:
            continue
        output[index] = _sample_weight(
            element=element,
            sample_time=sample_time,
            propagation_delay=propagation_delay,
            weight_scale=1.0,
            baffle_distance=None,
            sound_speed=sound_speed,
            sampling_frequency=sampling_frequency,
        ) * fraction
    return output


def _best_integer_shift(openfield: np.ndarray, fieldii: np.ndarray) -> tuple[int, float]:
    best_shift = 0
    best_error = np.inf
    for shift in range(-3, 4):
        shifted = np.zeros_like(openfield)
        if shift < 0:
            shifted[:shift] = openfield[-shift:]
        elif shift > 0:
            shifted[shift:] = openfield[:-shift]
        else:
            shifted = openfield
        error = float(np.max(np.abs(shifted - fieldii)))
        if error < best_error:
            best_shift = shift
            best_error = error
    return best_shift, best_error


if __name__ == "__main__":
    main()
