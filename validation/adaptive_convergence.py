from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openfield import Medium, Simulation  # noqa: E402
from openfield.apertures import (  # noqa: E402
    ConvexArray,
    ConvexFocusedArray,
    FocusedLinearArray,
    Piston,
)
from openfield.physics.spatial_impulse import _pack_aperture  # noqa: E402


POINTS = np.asarray(
    [
        [0.0, 0.0, 30e-3],
        [0.0, 0.0, 40e-3],
        [2e-3, 0.0, 40e-3],
    ],
    dtype=np.float64,
)
FOCUS = [0.0, 0.0, 40e-3]
RESULTS = ROOT / "validation" / "results" / "convergence" / "adaptive_curved"
EPS = 1e-30


@dataclass(frozen=True)
class CaseSpec:
    name: str
    label: str
    builder: Callable[[str, float], object]
    parameter_name: str
    candidate_values: tuple[float, ...]
    reference_value: float
    compatibility_value: float


@dataclass(frozen=True)
class RunResult:
    case: str
    label: str
    mode: str
    parameter_name: str
    parameter_value: float
    subelements: int
    facets: int
    runtime: float
    time: np.ndarray
    samples: np.ndarray


@dataclass(frozen=True)
class MetricRow:
    case: str
    label: str
    mode: str
    parameter_name: str
    parameter_value: float
    subelements: int
    facets: int
    runtime: float
    relative_l2: float
    relative_max_abs: float
    max_abs: float


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare adaptive curved-aperture tessellations against fine adaptive OpenField references.",
    )
    parser.add_argument(
        "cases",
        nargs="*",
        help="Cases to run. Defaults to all cases.",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Timing repeats per run.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS, help="Directory for CSV and SVG output.")
    args = parser.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")

    known = {case.name for case in cases()}
    unknown = sorted(set(args.cases) - known)
    if unknown:
        raise SystemExit(f"Unknown case(s): {', '.join(unknown)}")
    selected = set(args.cases) if args.cases else known
    specs = [case for case in cases() if case.name in selected]
    simulation = Simulation(sampling_frequency=100e6, medium=Medium(sound_speed=1540.0))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _warm_numba(simulation)

    rows: list[MetricRow] = []
    for spec in specs:
        reference = run_case(
            simulation,
            spec,
            mode="adaptive_reference",
            parameter_value=spec.reference_value,
            repeats=args.repeats,
        )
        case_runs = [
            run_case(
                simulation,
                spec,
                mode="compatibility",
                parameter_value=spec.compatibility_value,
                repeats=args.repeats,
            )
        ]
        case_runs.extend(
            run_case(
                simulation,
                spec,
                mode="adaptive",
                parameter_value=value,
                repeats=args.repeats,
            )
            for value in spec.candidate_values
        )
        case_runs.append(reference)
        for run in case_runs:
            rows.append(compare_to_reference(run, reference))

    write_summary(args.output_dir / "summary.csv", rows)
    write_case_plots(args.output_dir, rows)
    write_metadata(args.output_dir / "metadata.json", specs, rows)
    print(f"Wrote adaptive convergence study to {args.output_dir}")


def cases() -> tuple[CaseSpec, ...]:
    return (
        CaseSpec(
            name="piston",
            label="Piston",
            builder=build_piston,
            parameter_name="element_size_m",
            candidate_values=(1.0e-3, 0.75e-3, 0.5e-3, 0.35e-3, 0.25e-3),
            reference_value=0.125e-3,
            compatibility_value=0.5e-3,
        ),
        CaseSpec(
            name="focused_linear_array",
            label="Focused Linear",
            builder=build_focused_linear,
            parameter_name="max_sagitta_m",
            candidate_values=(20e-6, 10e-6, 5e-6, 2.5e-6, 1.0e-6),
            reference_value=0.5e-6,
            compatibility_value=0.0,
        ),
        CaseSpec(
            name="convex_array",
            label="Convex",
            builder=build_convex,
            parameter_name="max_sagitta_m",
            candidate_values=(20e-6, 10e-6, 5e-6, 2.5e-6, 1.0e-6),
            reference_value=0.5e-6,
            compatibility_value=0.0,
        ),
        CaseSpec(
            name="convex_focused_array",
            label="Convex Focused",
            builder=build_convex_focused,
            parameter_name="max_sagitta_m",
            candidate_values=(20e-6, 10e-6, 5e-6, 2.5e-6, 1.0e-6),
            reference_value=0.5e-6,
            compatibility_value=0.0,
        ),
    )


def build_piston(mode: str, parameter_value: float):
    if mode == "compatibility":
        return Piston(radius=5e-3, element_size=parameter_value, tessellation="cartesian")
    return Piston(radius=5e-3, element_size=parameter_value, tessellation="polar")


def build_focused_linear(mode: str, parameter_value: float):
    if mode == "compatibility":
        return FocusedLinearArray(
            elements=6,
            width=0.3e-3,
            height=5e-3,
            kerf=0.03e-3,
            elevation_focus=20e-3,
            subdivisions=(1, 4),
            tessellation="fieldii",
        ).focused_at(FOCUS)
    return FocusedLinearArray(
        elements=6,
        width=0.3e-3,
        height=5e-3,
        kerf=0.03e-3,
        elevation_focus=20e-3,
        tessellation="adaptive",
        max_sagitta=parameter_value,
    ).focused_at(FOCUS)


def build_convex(mode: str, parameter_value: float):
    if mode == "compatibility":
        return ConvexArray(
            elements=6,
            width=0.3e-3,
            height=5e-3,
            kerf=0.03e-3,
            convex_radius=25e-3,
            subdivisions=(1, 3),
            tessellation="fieldii",
        ).focused_at(FOCUS)
    return ConvexArray(
        elements=6,
        width=0.3e-3,
        height=5e-3,
        kerf=0.03e-3,
        convex_radius=25e-3,
        tessellation="adaptive",
        max_sagitta=parameter_value,
    ).focused_at(FOCUS)


def build_convex_focused(mode: str, parameter_value: float):
    if mode == "compatibility":
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
    return ConvexFocusedArray(
        elements=6,
        width=0.3e-3,
        height=5e-3,
        kerf=0.03e-3,
        convex_radius=25e-3,
        elevation_focus=20e-3,
        tessellation="adaptive",
        max_sagitta=parameter_value,
    ).focused_at(FOCUS)


def _warm_numba(simulation: Simulation) -> None:
    simulation.spatial_impulse_response(Piston(radius=1e-3, element_size=1e-3), POINTS[:1])


def run_case(
    simulation: Simulation,
    spec: CaseSpec,
    *,
    mode: str,
    parameter_value: float,
    repeats: int,
) -> RunResult:
    aperture = spec.builder(mode, parameter_value)
    packed = _pack_aperture(aperture)
    timings = []
    response = None
    for _ in range(repeats):
        start = time.perf_counter()
        response = simulation.spatial_impulse_response(aperture, POINTS)
        timings.append(time.perf_counter() - start)
    if response is None:
        raise RuntimeError("response was not calculated")
    return RunResult(
        case=spec.name,
        label=spec.label,
        mode=mode,
        parameter_name=spec.parameter_name,
        parameter_value=parameter_value,
        subelements=len(aperture.elements),
        facets=int(packed.facet_vertices.shape[0]),
        runtime=float(statistics.median(timings)),
        time=np.asarray(response.time, dtype=np.float64),
        samples=ensure_2d(np.asarray(response.samples, dtype=np.float64)),
    )


def compare_to_reference(run: RunResult, reference: RunResult) -> MetricRow:
    aligned = np.zeros_like(reference.samples)
    channels = min(run.samples.shape[1], reference.samples.shape[1])
    for channel in range(channels):
        aligned[:, channel] = np.interp(
            reference.time,
            run.time,
            run.samples[:, channel],
            left=0.0,
            right=0.0,
        )
    diff = aligned - reference.samples
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    reference_max = max(float(np.max(np.abs(reference.samples))) if reference.samples.size else 0.0, EPS)
    reference_norm = max(float(np.linalg.norm(reference.samples.ravel())), EPS)
    return MetricRow(
        case=run.case,
        label=run.label,
        mode=run.mode,
        parameter_name=run.parameter_name,
        parameter_value=run.parameter_value,
        subelements=run.subelements,
        facets=run.facets,
        runtime=run.runtime,
        relative_l2=float(np.linalg.norm(diff.ravel()) / reference_norm),
        relative_max_abs=max_abs / reference_max,
        max_abs=max_abs,
    )


def write_summary(path: Path, rows: list[MetricRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case",
                "label",
                "mode",
                "parameter_name",
                "parameter_value",
                "subelements",
                "facets",
                "median_seconds",
                "relative_l2",
                "relative_max_abs",
                "max_abs",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.case,
                    row.label,
                    row.mode,
                    row.parameter_name,
                    row.parameter_value,
                    row.subelements,
                    row.facets,
                    row.runtime,
                    row.relative_l2,
                    row.relative_max_abs,
                    row.max_abs,
                ]
            )


def write_case_plots(output_dir: Path, rows: list[MetricRow]) -> None:
    for case_name in sorted({row.case for row in rows}):
        case_rows = [row for row in rows if row.case == case_name]
        write_runtime_error_svg(output_dir / f"{case_name}_runtime_vs_error.svg", case_rows)


def write_runtime_error_svg(path: Path, rows: list[MetricRow]) -> None:
    plot_rows = [row for row in rows if row.runtime > 0]
    if not plot_rows:
        path.write_text(empty_svg("No runtime data"), encoding="utf-8")
        return
    min_error = positive_floor([row.relative_l2 for row in plot_rows])
    x_values = [max(row.relative_l2, min_error) for row in plot_rows]
    y_values = [row.runtime for row in plot_rows]
    x_min, x_max = log_bounds(x_values)
    y_min, y_max = log_bounds(y_values)

    width = 820
    height = 500
    left = 86
    top = 58
    plot_width = 560
    plot_height = 340
    right = left + plot_width
    bottom = top + plot_height

    title = plot_rows[0].label
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="24" y="34" font-family="Arial, Helvetica, sans-serif" font-size="21" font-weight="700">{title}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#fff" stroke="#d8d8d8"/>',
    ]
    for power in range(x_min, x_max + 1):
        x = map_log(10.0**power, (x_min, x_max), left, right)
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" stroke="#ececec"/>')
        parts.append(text(x, bottom + 22, f"1e{power}", anchor="middle", size=10))
    for power in range(y_min, y_max + 1):
        y = map_log(10.0**power, (y_min, y_max), bottom, top)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#ececec"/>')
        parts.append(text(left - 10, y + 4, f"1e{power}", anchor="end", size=10))
    parts.append(text((left + right) / 2, bottom + 48, "relative L2 error", anchor="middle"))
    parts.append(
        f'<text x="24" y="{(top + bottom) / 2:.2f}" font-family="Arial, Helvetica, sans-serif" font-size="12" '
        f'text-anchor="middle" fill="#222" transform="rotate(-90 24 {(top + bottom) / 2:.2f})">median runtime [s]</text>'
    )

    for row in plot_rows:
        color = "#2A9D8F" if row.mode.startswith("adaptive") else "#6667AB"
        shape = "square" if row.mode == "compatibility" else "circle"
        draw_marker(
            parts,
            map_log(max(row.relative_l2, min_error), (x_min, x_max), left, right),
            map_log(row.runtime, (y_min, y_max), bottom, top),
            color,
            shape,
        )
    parts.append(legend(675, 82, "#2A9D8F", "adaptive"))
    parts.append(legend(675, 106, "#6667AB", "compatibility", shape="square"))
    parts.append("</svg>\n")
    path.write_text("".join(parts), encoding="utf-8")


def write_metadata(path: Path, specs: list[CaseSpec], rows: list[MetricRow]) -> None:
    payload = {
        "case": "adaptive_curved_convergence",
        "points": POINTS.tolist(),
        "specs": [
            {
                "name": spec.name,
                "label": spec.label,
                "parameter_name": spec.parameter_name,
                "candidate_values": list(spec.candidate_values),
                "reference_value": spec.reference_value,
                "compatibility_value": spec.compatibility_value,
            }
            for spec in specs
        ],
        "rows": [row.__dict__ for row in rows],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ensure_2d(values: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        return values[:, None]
    return values


def positive_floor(values: list[float]) -> float:
    positive = [float(value) for value in values if np.isfinite(value) and value > 0.0]
    if not positive:
        return 1e-16
    return max(min(positive) / 10.0, 1e-16)


def log_bounds(values: list[float]) -> tuple[int, int]:
    finite = [max(float(value), EPS) for value in values if np.isfinite(value) and value > 0.0]
    if not finite:
        return -12, 0
    lower = int(np.floor(np.log10(min(finite))))
    upper = int(np.ceil(np.log10(max(finite))))
    if lower == upper:
        upper += 1
    return lower, upper


def map_log(value: float, bounds: tuple[int, int], out_min: float, out_max: float) -> float:
    lower, upper = bounds
    return out_min + (np.log10(max(value, EPS)) - lower) / (upper - lower) * (out_max - out_min)


def draw_marker(parts: list[str], x: float, y: float, color: str, shape: str) -> None:
    if shape == "square":
        parts.append(f'<rect x="{x - 4:.2f}" y="{y - 4:.2f}" width="8" height="8" fill="{color}"/>')
    else:
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.3" fill="{color}"/>')


def legend(x: float, y: float, color: str, label: str, *, shape: str = "circle") -> str:
    if shape == "square":
        mark = f'<rect x="{x:.2f}" y="{y - 8:.2f}" width="10" height="10" fill="{color}"/>'
    else:
        mark = f'<circle cx="{x + 5:.2f}" cy="{y - 3:.2f}" r="5" fill="{color}"/>'
    return mark + text(x + 26, y, label)


def text(x: float, y: float, content: str, *, anchor: str = "start", size: int = 12) -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" text-anchor="{anchor}" fill="#222">{content}</text>'
    )


def empty_svg(message: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="180">'
        '<rect width="100%" height="100%" fill="#fff"/>'
        f'<text x="24" y="44" font-family="Arial, Helvetica, sans-serif" font-size="18">{message}</text>'
        "</svg>\n"
    )


if __name__ == "__main__":
    main()
