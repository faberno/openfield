from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openfield import Medium, Simulation  # noqa: E402
from openfield.apertures import ConcavePiston  # noqa: E402
from openfield.physics.spatial_impulse import _pack_aperture  # noqa: E402


RESULTS = ROOT / "validation" / "results"
DEFAULT_OUTPUT_DIR = RESULTS / "convergence" / "concave_piston"
FIELDII_CASE_DIR = RESULTS / "concave_piston_spatial_impulse" / "fieldii"
POINTS = np.asarray(
    [
        [0.0, 0.0, 30e-3],
        [0.0, 0.0, 40e-3],
        [2e-3, 0.0, 40e-3],
    ],
    dtype=np.float64,
)
SVG_FONT = "Arial, Helvetica, sans-serif"
EPS = 1e-30


@dataclass(frozen=True)
class ResponseRun:
    label: str
    source: str
    tessellation: str | None
    element_size: float | None
    subelements: int | None
    facets: int | None
    runtime: float | None
    time: np.ndarray
    samples: np.ndarray


@dataclass(frozen=True)
class Metrics:
    label: str
    source: str
    tessellation: str | None
    element_size: float | None
    subelements: int | None
    facets: int | None
    runtime: float | None
    relative_l2: float
    relative_max_abs: float
    max_abs: float
    relative_area_l2: float


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an OpenField convergence study for the concave piston spatial impulse response.",
    )
    parser.add_argument(
        "--element-sizes",
        type=float,
        nargs="+",
        default=[1.0e-3, 0.75e-3, 0.5e-3, 0.35e-3, 0.25e-3],
        help="OpenField element sizes to compare, in meters.",
    )
    parser.add_argument(
        "--reference-element-size",
        type=float,
        default=0.125e-3,
        help="Fine OpenField element size used as the convergence reference, in meters.",
    )
    parser.add_argument(
        "--tessellations",
        nargs="+",
        choices=["cartesian", "polar", "adaptive"],
        default=["cartesian", "polar"],
        help="OpenField concave piston tessellations to compare.",
    )
    parser.add_argument(
        "--reference-tessellation",
        choices=["cartesian", "polar", "adaptive"],
        default="polar",
        help="OpenField tessellation used for the fine convergence reference.",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Timing repeats per OpenField element size.")
    parser.add_argument("--radius", type=float, default=5e-3, help="Concave piston radius, in meters.")
    parser.add_argument("--focal-radius", type=float, default=30e-3, help="Concave piston focal radius, in meters.")
    parser.add_argument("--sampling-frequency", type=float, default=100e6, help="Sampling frequency in Hz.")
    parser.add_argument("--sound-speed", type=float, default=1540.0, help="Sound speed in m/s.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for CSV and SVG output.")
    parser.add_argument("--skip-fieldii", action="store_true", help="Do not include saved Field II validation output.")
    args = parser.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    if args.reference_element_size <= 0 or any(size <= 0 for size in args.element_sizes):
        raise SystemExit("element sizes must be positive")
    tessellations = unique_tessellations(args.tessellations)
    reference_tessellation = normalize_tessellation(args.reference_tessellation)

    simulation = Simulation(
        sampling_frequency=args.sampling_frequency,
        medium=Medium(sound_speed=args.sound_speed),
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    _warm_numba(simulation, args.radius, args.focal_radius)

    reference = run_openfield(
        simulation,
        radius=args.radius,
        focal_radius=args.focal_radius,
        element_size=args.reference_element_size,
        tessellation=reference_tessellation,
        repeats=args.repeats,
        label=f"OpenField {reference_tessellation} reference {format_mm(args.reference_element_size)}",
    )

    runs: list[ResponseRun] = []
    sizes = sorted(set(float(size) for size in args.element_sizes), reverse=True)
    reference_included = False
    for tessellation in tessellations:
        for element_size in sizes:
            if tessellation == reference_tessellation and element_size == args.reference_element_size:
                runs.append(reference)
                reference_included = True
                continue
            runs.append(
                run_openfield(
                    simulation,
                    radius=args.radius,
                    focal_radius=args.focal_radius,
                    element_size=element_size,
                    tessellation=tessellation,
                    repeats=args.repeats,
                    label=f"OpenField {tessellation} {format_mm(element_size)}",
                )
            )
    if not reference_included:
        runs.append(reference)

    if not args.skip_fieldii:
        fieldii = load_fieldii_run()
        if fieldii is not None:
            runs.append(fieldii)

    metrics = [compare_to_reference(run, reference) for run in runs]
    write_summary_csv(output_dir / "summary.csv", metrics)
    write_error_vs_element_size(output_dir / "error_vs_element_size.svg", metrics, args.reference_element_size)
    write_runtime_vs_error(output_dir / "runtime_vs_error.svg", metrics)
    write_waveform_overlay(output_dir / "waveform_overlay.svg", runs, reference)
    write_metadata(
        output_dir / "metadata.json",
        args=args,
        reference=reference,
        metrics=metrics,
    )

    print(f"Wrote concave piston convergence study to {output_dir}")


def _warm_numba(simulation: Simulation, radius: float, focal_radius: float) -> None:
    aperture = ConcavePiston(radius=radius, focal_radius=focal_radius, element_size=1.0e-3)
    simulation.spatial_impulse_response(aperture, POINTS[:1])


def run_openfield(
    simulation: Simulation,
    *,
    radius: float,
    focal_radius: float,
    element_size: float,
    tessellation: str,
    repeats: int,
    label: str,
) -> ResponseRun:
    timings = []
    response = None
    aperture = ConcavePiston(
        radius=radius,
        focal_radius=focal_radius,
        element_size=element_size,
        tessellation=tessellation,
    )
    packed = _pack_aperture(aperture)
    for _ in range(repeats):
        start = time.perf_counter()
        response = simulation.spatial_impulse_response(aperture, POINTS)
        timings.append(time.perf_counter() - start)
    if response is None:
        raise RuntimeError("response was not calculated")
    return ResponseRun(
        label=label,
        source="openfield",
        tessellation=tessellation,
        element_size=element_size,
        subelements=len(aperture.elements),
        facets=int(packed.facet_vertices.shape[0]),
        runtime=float(statistics.median(timings)),
        time=np.asarray(response.time, dtype=np.float64),
        samples=ensure_2d(np.asarray(response.samples, dtype=np.float64)),
    )


def load_fieldii_run() -> ResponseRun | None:
    if not FIELDII_CASE_DIR.exists():
        return None
    samples_path = FIELDII_CASE_DIR / "samples.csv"
    time_path = FIELDII_CASE_DIR / "time.csv"
    if not samples_path.exists() or not time_path.exists():
        return None
    metadata = load_json(FIELDII_CASE_DIR / "metadata.json") or {}
    runtime = load_runtime(FIELDII_CASE_DIR / "runtime.json")
    return ResponseRun(
        label="Field II xdc_concave 0.500 mm",
        source="fieldii",
        tessellation=None,
        element_size=0.5e-3,
        subelements=metadata.get("subelement_count"),
        facets=metadata.get("subelement_count"),
        runtime=runtime,
        time=np.asarray(load_csv(time_path), dtype=np.float64).ravel(),
        samples=ensure_2d(load_csv(samples_path)),
    )


def compare_to_reference(run: ResponseRun, reference: ResponseRun) -> Metrics:
    aligned = np.empty_like(reference.samples)
    channels = min(run.samples.shape[1], reference.samples.shape[1])
    aligned[:, :] = 0.0
    for channel in range(channels):
        aligned[:, channel] = np.interp(
            reference.time,
            run.time,
            run.samples[:, channel],
            left=0.0,
            right=0.0,
        )

    reference_samples = reference.samples
    diff = aligned - reference_samples
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    relative_max_abs = max_abs / max(float(np.max(np.abs(reference_samples))), EPS)
    relative_l2 = float(np.linalg.norm(diff.ravel()) / max(np.linalg.norm(reference_samples.ravel()), EPS))

    reference_area = np.trapezoid(reference_samples, reference.time, axis=0)
    run_area = np.trapezoid(aligned, reference.time, axis=0)
    relative_area_l2 = float(np.linalg.norm(run_area - reference_area) / max(np.linalg.norm(reference_area), EPS))

    return Metrics(
        label=run.label,
        source=run.source,
        tessellation=run.tessellation,
        element_size=run.element_size,
        subelements=run.subelements,
        facets=run.facets,
        runtime=run.runtime,
        relative_l2=relative_l2,
        relative_max_abs=relative_max_abs,
        max_abs=max_abs,
        relative_area_l2=relative_area_l2,
    )


def write_summary_csv(path: Path, metrics: list[Metrics]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "label",
                "source",
                "tessellation",
                "element_size_m",
                "subelements",
                "facets",
                "median_seconds",
                "relative_l2",
                "relative_max_abs",
                "max_abs",
                "relative_area_l2",
            ]
        )
        for item in metrics:
            writer.writerow(
                [
                    item.label,
                    item.source,
                    "" if item.tessellation is None else item.tessellation,
                    "" if item.element_size is None else item.element_size,
                    "" if item.subelements is None else item.subelements,
                    "" if item.facets is None else item.facets,
                    "" if item.runtime is None else item.runtime,
                    item.relative_l2,
                    item.relative_max_abs,
                    item.max_abs,
                    item.relative_area_l2,
                ]
            )


def write_error_vs_element_size(path: Path, metrics: list[Metrics], reference_element_size: float) -> None:
    openfield = [item for item in metrics if item.source == "openfield" and item.element_size is not None]
    fieldii = [item for item in metrics if item.source == "fieldii" and item.element_size is not None]
    if not openfield:
        write_empty_svg(path, "No OpenField convergence data", "Run this script with at least one element size.")
        return
    plot_floor = positive_floor([item.relative_l2 for item in openfield + fieldii])
    x_values = [item.element_size * 1e3 for item in openfield + fieldii if item.element_size is not None]
    y_values = [max(item.relative_l2, plot_floor) for item in openfield + fieldii]
    x_range = log_range(x_values)
    y_range = log_range(y_values)

    width = 880
    height = 560
    plot = PlotBox(left=92, top=78, width=680, height=390)
    parts = svg_header(width, height)
    parts.append(text(28, 34, "Concave Piston Convergence", size=22, weight="700"))
    parts.append(text(28, 58, "Error relative to a fine OpenField reference; lower is better.", size=12, fill="#555"))
    draw_log_axes(parts, plot, x_range, y_range, "element size [mm]", "relative L2 error")
    legend_y = 95
    for tessellation in sorted_openfield_tessellations(openfield):
        color = tessellation_color(tessellation)
        series = [item for item in openfield if item.tessellation == tessellation]
        draw_polyline_for_metrics(parts, plot, series, x_range, y_range, color, floor=plot_floor)
        for item in series:
            draw_marker(parts, plot, item.element_size * 1e3, max(item.relative_l2, plot_floor), x_range, y_range, color)
        parts.append(legend_item(610, legend_y, color, f"OpenField {tessellation}"))
        legend_y += 23
    for item in fieldii:
        draw_marker(parts, plot, item.element_size * 1e3, max(item.relative_l2, plot_floor), x_range, y_range, "#6667AB", shape="square")
    parts.append(legend_item(610, legend_y, "#6667AB", "Field II", shape="square"))
    legend_y += 28
    parts.append(text(610, legend_y, f"reference: {format_mm(reference_element_size)}", size=11, fill="#555"))
    parts.append("</svg>\n")
    path.write_text("".join(parts), encoding="utf-8")


def write_runtime_vs_error(path: Path, metrics: list[Metrics]) -> None:
    plot_metrics = [
        item
        for item in metrics
        if item.runtime is not None and math.isfinite(item.runtime) and item.runtime > 0
    ]
    if not plot_metrics:
        write_empty_svg(path, "No runtime data", "Runtime data is collected for OpenField runs and loaded from Field II benchmark output when available.")
        return
    plot_floor = positive_floor([item.relative_l2 for item in plot_metrics])
    x_values = [max(item.relative_l2, plot_floor) for item in plot_metrics]
    y_values = [item.runtime for item in plot_metrics if item.runtime is not None]
    x_range = log_range(x_values)
    y_range = log_range(y_values)

    width = 880
    height = 560
    plot = PlotBox(left=92, top=78, width=680, height=390)
    parts = svg_header(width, height)
    parts.append(text(28, 34, "Runtime vs Error", size=22, weight="700"))
    parts.append(text(28, 58, "Tradeoff against the fine OpenField reference.", size=12, fill="#555"))
    draw_log_axes(parts, plot, x_range, y_range, "relative L2 error", "median runtime [s]")
    for item in plot_metrics:
        color = "#6667AB" if item.source == "fieldii" else tessellation_color(item.tessellation)
        shape = "square" if item.source == "fieldii" else "circle"
        draw_marker(parts, plot, max(item.relative_l2, plot_floor), item.runtime or EPS, x_range, y_range, color, shape=shape)
    legend_y = 95
    for tessellation in sorted_openfield_tessellations(plot_metrics):
        parts.append(legend_item(610, legend_y, tessellation_color(tessellation), f"OpenField {tessellation}"))
        legend_y += 23
    parts.append(legend_item(610, legend_y, "#6667AB", "Field II", shape="square"))
    parts.append("</svg>\n")
    path.write_text("".join(parts), encoding="utf-8")


def write_waveform_overlay(path: Path, runs: list[ResponseRun], reference: ResponseRun) -> None:
    selected = [reference]
    for label in (
        "OpenField polar 0.500 mm",
        "OpenField cartesian 0.500 mm",
        "Field II xdc_concave 0.500 mm",
    ):
        match = next((run for run in runs if run.label == label), None)
        if match is not None and match not in selected:
            selected.append(match)

    width = 1000
    height = 520
    plot = PlotBox(left=78, top=76, width=840, height=330)
    channel = 0
    time_min = min(float(run.time[0]) for run in selected) * 1e6
    time_max = max(float(run.time[-1]) for run in selected) * 1e6
    amplitude = max(float(np.max(np.abs(run.samples[:, channel]))) for run in selected)
    y_limit = max(amplitude, EPS) * 1.05

    parts = svg_header(width, height)
    parts.append(text(28, 34, "Concave Piston Waveform Overlay", size=22, weight="700"))
    parts.append(text(28, 58, "First validation point/channel; reference is a fine OpenField solve.", size=12, fill="#555"))
    draw_linear_axes(parts, plot, (time_min, time_max), (-y_limit, y_limit), "time [us]", "SIR amplitude")
    colors = ["#111111", "#D95D39", "#E0A458", "#6667AB"]
    for run, color in zip(selected, colors, strict=False):
        x = map_linear(run.time * 1e6, (time_min, time_max), plot.left, plot.right)
        y = map_linear(run.samples[:, channel], (-y_limit, y_limit), plot.bottom, plot.top)
        parts.append(polyline(x, y, color))
    for index, (run, color) in enumerate(zip(selected, colors, strict=False)):
        parts.append(legend_item(650, 92 + 23 * index, color, run.label, shape="line"))
    parts.append("</svg>\n")
    path.write_text("".join(parts), encoding="utf-8")


def write_metadata(path: Path, *, args: argparse.Namespace, reference: ResponseRun, metrics: list[Metrics]) -> None:
    payload = {
        "case": "concave_piston_convergence",
        "points": POINTS.tolist(),
        "sampling_frequency": float(args.sampling_frequency),
        "sound_speed": float(args.sound_speed),
        "radius": float(args.radius),
        "focal_radius": float(args.focal_radius),
        "tessellations": unique_tessellations(args.tessellations),
        "reference_tessellation": normalize_tessellation(args.reference_tessellation),
        "reference_label": reference.label,
        "reference_element_size": float(args.reference_element_size),
        "rows": [
            {
                "label": item.label,
                "source": item.source,
                "tessellation": item.tessellation,
                "element_size": item.element_size,
                "subelements": item.subelements,
                "facets": item.facets,
                "runtime": item.runtime,
                "relative_l2": item.relative_l2,
                "relative_max_abs": item.relative_max_abs,
                "relative_area_l2": item.relative_area_l2,
            }
            for item in metrics
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class PlotBox:
    left: float
    top: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def bottom(self) -> float:
        return self.top + self.height


def draw_log_axes(parts: list[str], plot: PlotBox, x_range: tuple[int, int], y_range: tuple[int, int], x_label: str, y_label: str) -> None:
    parts.append(f'<rect x="{plot.left}" y="{plot.top}" width="{plot.width}" height="{plot.height}" fill="#fff" stroke="#d8d8d8"/>')
    for power in range(x_range[0], x_range[1] + 1):
        x = map_log(10.0**power, x_range, plot.left, plot.right)
        parts.append(f'<line x1="{x:.2f}" y1="{plot.top}" x2="{x:.2f}" y2="{plot.bottom}" stroke="#ececec"/>')
        parts.append(text(x, plot.bottom + 22, f"1e{power}", size=10, anchor="middle", fill="#555"))
    for power in range(y_range[0], y_range[1] + 1):
        y = map_log(10.0**power, y_range, plot.bottom, plot.top)
        parts.append(f'<line x1="{plot.left}" y1="{y:.2f}" x2="{plot.right}" y2="{y:.2f}" stroke="#ececec"/>')
        parts.append(text(plot.left - 10, y + 4, f"1e{power}", size=10, anchor="end", fill="#555"))
    parts.append(text((plot.left + plot.right) / 2, plot.bottom + 48, x_label, size=12, anchor="middle"))
    parts.append(
        f'<text x="24" y="{(plot.top + plot.bottom) / 2:.2f}" font-family="{SVG_FONT}" font-size="12" '
        'text-anchor="middle" fill="#222" transform="rotate(-90 24 '
        f'{(plot.top + plot.bottom) / 2:.2f})">{html.escape(y_label)}</text>'
    )


def draw_linear_axes(parts: list[str], plot: PlotBox, x_range: tuple[float, float], y_range: tuple[float, float], x_label: str, y_label: str) -> None:
    parts.append(f'<rect x="{plot.left}" y="{plot.top}" width="{plot.width}" height="{plot.height}" fill="#fff" stroke="#d8d8d8"/>')
    for fraction in np.linspace(0.0, 1.0, 6):
        x_value = x_range[0] + fraction * (x_range[1] - x_range[0])
        x = plot.left + fraction * plot.width
        parts.append(f'<line x1="{x:.2f}" y1="{plot.top}" x2="{x:.2f}" y2="{plot.bottom}" stroke="#ececec"/>')
        parts.append(text(x, plot.bottom + 22, f"{x_value:.2f}", size=10, anchor="middle", fill="#555"))
    for fraction in np.linspace(0.0, 1.0, 5):
        y_value = y_range[0] + fraction * (y_range[1] - y_range[0])
        y = plot.bottom - fraction * plot.height
        parts.append(f'<line x1="{plot.left}" y1="{y:.2f}" x2="{plot.right}" y2="{y:.2f}" stroke="#ececec"/>')
        parts.append(text(plot.left - 10, y + 4, f"{y_value:.2g}", size=10, anchor="end", fill="#555"))
    parts.append(text((plot.left + plot.right) / 2, plot.bottom + 48, x_label, size=12, anchor="middle"))
    parts.append(
        f'<text x="24" y="{(plot.top + plot.bottom) / 2:.2f}" font-family="{SVG_FONT}" font-size="12" '
        'text-anchor="middle" fill="#222" transform="rotate(-90 24 '
        f'{(plot.top + plot.bottom) / 2:.2f})">{html.escape(y_label)}</text>'
    )


def draw_polyline_for_metrics(
    parts: list[str],
    plot: PlotBox,
    metrics: list[Metrics],
    x_range: tuple[int, int],
    y_range: tuple[int, int],
    color: str,
    *,
    floor: float,
) -> None:
    ordered = sorted(metrics, key=lambda item: item.element_size or 0.0)
    x = np.asarray([map_log((item.element_size or EPS) * 1e3, x_range, plot.left, plot.right) for item in ordered])
    y = np.asarray([map_log(max(item.relative_l2, floor), y_range, plot.bottom, plot.top) for item in ordered])
    parts.append(polyline(x, y, color, width=1.5))


def draw_marker(
    parts: list[str],
    plot: PlotBox,
    x_value: float,
    y_value: float,
    x_range: tuple[int, int],
    y_range: tuple[int, int],
    color: str,
    *,
    shape: str = "circle",
) -> None:
    x = map_log(x_value, x_range, plot.left, plot.right)
    y = map_log(y_value, y_range, plot.bottom, plot.top)
    if shape == "square":
        parts.append(f'<rect x="{x - 4:.2f}" y="{y - 4:.2f}" width="8" height="8" fill="{color}"/>')
    else:
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.3" fill="{color}"/>')


def legend_item(x: float, y: float, color: str, label: str, *, shape: str = "circle") -> str:
    if shape == "square":
        mark = f'<rect x="{x:.2f}" y="{y - 8:.2f}" width="10" height="10" fill="{color}"/>'
    elif shape == "line":
        mark = f'<path d="M {x:.2f} {y - 4:.2f} h 22" stroke="{color}" stroke-width="2"/>'
    else:
        mark = f'<circle cx="{x + 5:.2f}" cy="{y - 3:.2f}" r="5" fill="{color}"/>'
    return mark + text(x + 30, y, label, size=12)


def log_range(values: list[float]) -> tuple[int, int]:
    finite = [max(float(value), EPS) for value in values if math.isfinite(float(value)) and value > 0]
    if not finite:
        return -12, 0
    lower = math.floor(math.log10(min(finite)))
    upper = math.ceil(math.log10(max(finite)))
    if lower == upper:
        upper += 1
    return lower, upper


def positive_floor(values: list[float]) -> float:
    positive = [float(value) for value in values if math.isfinite(float(value)) and float(value) > 0.0]
    if not positive:
        return 1e-16
    return max(min(positive) / 10.0, 1e-16)


def map_log(value: float, log_range_values: tuple[int, int], out_min: float, out_max: float) -> float:
    lower, upper = log_range_values
    return out_min + (math.log10(max(value, EPS)) - lower) / (upper - lower) * (out_max - out_min)


def map_linear(values: np.ndarray, input_range: tuple[float, float], out_min: float, out_max: float) -> np.ndarray:
    lower, upper = input_range
    if upper == lower:
        return np.full_like(values, (out_min + out_max) / 2, dtype=np.float64)
    return out_min + (values - lower) / (upper - lower) * (out_max - out_min)


def ensure_2d(data: np.ndarray) -> np.ndarray:
    array = np.asarray(data, dtype=np.float64)
    if array.ndim == 1:
        return array[:, None]
    return array


def normalize_tessellation(value: str) -> str:
    if value == "adaptive":
        return "polar"
    return value


def unique_tessellations(values: list[str]) -> list[str]:
    unique = []
    for value in values:
        tessellation = normalize_tessellation(value)
        if tessellation not in unique:
            unique.append(tessellation)
    return unique


def sorted_openfield_tessellations(metrics: list[Metrics]) -> list[str]:
    present = {item.tessellation for item in metrics if item.source == "openfield" and item.tessellation is not None}
    preferred = [value for value in ("cartesian", "polar") if value in present]
    return preferred + sorted(str(value) for value in present - set(preferred))


def tessellation_color(tessellation: str | None) -> str:
    return {
        "cartesian": "#D95D39",
        "polar": "#2A9D8F",
    }.get(tessellation or "", "#D95D39")


def load_csv(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",")
    return np.asarray(data, dtype=np.float64)


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_runtime(path: Path) -> float | None:
    payload = load_json(path)
    if not payload:
        return None
    value = payload.get("median_seconds", payload.get("min_seconds"))
    return None if value is None else float(value)


def format_mm(value: float) -> str:
    return f"{value * 1e3:.3f} mm"


def write_empty_svg(path: Path, title: str, subtitle: str) -> None:
    parts = svg_header(820, 220)
    parts.append(text(24, 44, title, size=22, weight="700"))
    parts.append(text(24, 72, subtitle, size=13, fill="#555"))
    parts.append("</svg>\n")
    path.write_text("".join(parts), encoding="utf-8")


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
    ]


def text(
    x: float,
    y: float,
    content: str,
    *,
    size: int = 12,
    fill: str = "#222",
    weight: str = "400",
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="{SVG_FONT}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{html.escape(content)}</text>'
    )


def polyline(x: np.ndarray, y: np.ndarray, color: str, *, width: float = 2.0) -> str:
    points = " ".join(f"{float(px):.2f},{float(py):.2f}" for px, py in zip(x, y, strict=True))
    return f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{width}"/>'


if __name__ == "__main__":
    main()
