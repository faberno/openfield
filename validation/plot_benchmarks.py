from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation" / "results"
PLOTS = RESULTS / "plots"

SVG_FONT = "Arial, Helvetica, sans-serif"
EPS = 1e-30


def load_cases() -> list[dict]:
    return json.loads((ROOT / "validation" / "cases.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create benchmark plots from validation outputs.")
    parser.add_argument("cases", nargs="*", help="Case names to include. Defaults to every case with results.")
    parser.add_argument("--all", action="store_true", help="Include every case from cases.json.")
    parser.add_argument("--output-dir", type=Path, default=PLOTS, help="Directory for SVG plots.")
    parser.add_argument("--overlay-cases", nargs="*", default=None, help="Time-response cases for waveform overlays.")
    args = parser.parse_args()

    cases = load_cases()
    by_name = {case["name"]: case for case in cases}
    if args.all:
        selected_names = list(by_name)
    elif args.cases:
        selected_names = args.cases
    else:
        selected_names = [case["name"] for case in cases if (RESULTS / case["name"]).exists()]

    unknown = sorted(set(selected_names) - set(by_name))
    if unknown:
        raise SystemExit(f"Unknown validation case(s): {', '.join(unknown)}")

    rows = []
    for name in selected_names:
        metrics = compare_case(name, by_name[name])
        if metrics is not None:
            rows.append(metrics)

    if not rows:
        raise SystemExit("No comparable validation results found.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    write_summary_csv(output_dir / "benchmark_summary.csv", rows)
    write_accuracy_plot(output_dir / "accuracy_relative_l2.svg", rows)
    write_runtime_plot(output_dir / "runtime_seconds.svg", rows)
    write_speedup_plot(output_dir / "runtime_speedup.svg", rows)

    overlay_cases = args.overlay_cases
    if overlay_cases is None:
        overlay_cases = [
            "single_rectangle_spatial_impulse",
            "linear_array_spatial_impulse",
            "triangle_aperture_spatial_impulse",
            "linear_array_pulse_echo",
        ]
    overlay_rows = []
    for name in overlay_cases:
        if name in by_name and by_name[name].get("kind") == "time_response":
            overlay_rows.append(name)
    if overlay_rows:
        write_waveform_overlay(output_dir / "waveform_overlays.svg", overlay_rows)

    print(f"Wrote benchmark summary and plots to {output_dir}")


def compare_case(case_name: str, case: dict) -> dict | None:
    fieldii_dir = RESULTS / case_name / "fieldii"
    openfield_dir = RESULTS / case_name / "openfield"
    if not fieldii_dir.exists() or not openfield_dir.exists():
        return None

    openfield_runtime = load_runtime(openfield_dir / "runtime.json")
    fieldii_runtime = load_runtime(fieldii_dir / "runtime.json")
    speedup = None
    if openfield_runtime and fieldii_runtime and openfield_runtime > 0:
        speedup = fieldii_runtime / openfield_runtime

    if case.get("kind") == "time_response" and (fieldii_dir / "samples.csv").exists() and (openfield_dir / "samples.csv").exists():
        compared_files, shape_status, relative_l2, relative_max_abs, max_abs = compare_time_response(
            fieldii_dir, openfield_dir
        )
    else:
        compared_files, shape_status, relative_l2, relative_max_abs, max_abs = compare_common_csvs(
            fieldii_dir, openfield_dir
        )

    if not compared_files and openfield_runtime is None and fieldii_runtime is None:
        return None

    return {
        "case": case_name,
        "kind": case.get("kind", ""),
        "fieldii_must_match": bool(case.get("fieldii_must_match", False)),
        "reference_only": case.get("kind") == "time_response" and not bool(case.get("fieldii_must_match", False)),
        "compared_files": ";".join(compared_files),
        "shape_status": shape_status,
        "relative_l2": relative_l2,
        "relative_max_abs": relative_max_abs,
        "max_abs": max_abs,
        "openfield_seconds": openfield_runtime,
        "fieldii_seconds": fieldii_runtime,
        "fieldii_over_openfield": speedup,
    }


def compare_time_response(fieldii_dir: Path, openfield_dir: Path) -> tuple[list[str], str, float, float, float]:
    fieldii_time = load_csv(fieldii_dir / "time.csv").ravel()
    openfield_time = load_csv(openfield_dir / "time.csv").ravel()
    fieldii_samples = ensure_2d(load_csv(fieldii_dir / "samples.csv"))
    openfield_samples = ensure_2d(load_csv(openfield_dir / "samples.csv"))
    shape_status = f"fieldii={fieldii_samples.shape}, openfield={openfield_samples.shape}"

    if fieldii_time.size == 0 or openfield_time.size == 0:
        return [], shape_status, math.nan, math.nan, math.nan
    channels = min(fieldii_samples.shape[1], openfield_samples.shape[1])
    if channels == 0:
        return [], shape_status, math.nan, math.nan, math.nan

    start = max(float(fieldii_time[0]), float(openfield_time[0]))
    stop = min(float(fieldii_time[-1]), float(openfield_time[-1]))
    mask = (fieldii_time >= start) & (fieldii_time <= stop)
    if not np.any(mask):
        return [], f"{shape_status}; no overlapping time support", math.nan, math.nan, math.nan

    reference = fieldii_samples[mask, :channels]
    actual = np.empty_like(reference)
    for channel in range(channels):
        actual[:, channel] = np.interp(fieldii_time[mask], openfield_time, openfield_samples[:, channel])
    relative_l2, relative_max_abs, max_abs = difference_metrics(reference, actual)
    compared = [f"samples.csv interpolated on Field II time grid ({int(np.sum(mask))} samples)"]
    return compared, shape_status, relative_l2, relative_max_abs, max_abs


def compare_common_csvs(fieldii_dir: Path, openfield_dir: Path) -> tuple[list[str], str, float, float, float]:
    fieldii_files = {path.name for path in fieldii_dir.glob("*.csv")}
    openfield_files = {path.name for path in openfield_dir.glob("*.csv")}
    common_files = sorted(fieldii_files & openfield_files)
    if not common_files:
        return [], "no common CSV files", math.nan, math.nan, math.nan

    rel_l2_values = []
    rel_max_values = []
    max_abs_values = []
    shape_status = "ok"
    compared_files = []
    for filename in common_files:
        expected = load_csv(fieldii_dir / filename)
        actual = load_csv(openfield_dir / filename)
        if expected.shape != actual.shape:
            shape_status = f"mismatch:{filename}:{expected.shape}!={actual.shape}"
            continue
        rel_l2, rel_max, max_abs = difference_metrics(expected, actual)
        rel_l2_values.append(rel_l2)
        rel_max_values.append(rel_max)
        max_abs_values.append(max_abs)
        compared_files.append(filename)

    if not compared_files:
        return [], shape_status, math.nan, math.nan, math.nan
    return compared_files, shape_status, max(rel_l2_values), max(rel_max_values), max(max_abs_values)


def difference_metrics(expected: np.ndarray, actual: np.ndarray) -> tuple[float, float, float]:
    diff = actual - expected
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    scale = max(float(np.max(np.abs(expected))) if expected.size else 0.0, EPS)
    rel_max = max_abs / scale
    rel_l2 = float(np.linalg.norm(diff.ravel()) / max(np.linalg.norm(expected.ravel()), EPS))
    return rel_l2, rel_max, max_abs


def ensure_2d(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return data[:, None]
    return data


def load_csv(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",")
    return np.asarray(data, dtype=np.float64)


def load_runtime(path: Path) -> float | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("median_seconds", payload.get("min_seconds"))
    return None if value is None else float(value)


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "case",
        "kind",
        "reference_only",
        "compared_files",
        "shape_status",
        "relative_l2",
        "relative_max_abs",
        "max_abs",
        "fieldii_seconds",
        "openfield_seconds",
        "fieldii_over_openfield",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_accuracy_plot(path: Path, rows: list[dict]) -> None:
    plot_rows = [row for row in rows if math.isfinite(float(row["relative_l2"]))]
    plot_rows = sorted(plot_rows, key=lambda row: row["relative_l2"], reverse=True)
    if not plot_rows:
        write_empty_svg(path, "No accuracy data found", "Generate validation outputs first.")
        return
    write_log_bar_svg(
        path,
        plot_rows,
        value_key="relative_l2",
        title="OpenField Difference vs Field II",
        subtitle="Maximum relative L2 difference over common validation CSVs; Field II time responses are references, not the correctness oracle.",
        x_label="relative L2 difference",
        color_key="kind",
        color_map={"geometry": "#197278", "time_response": "#C44536"},
        floor=1e-16,
    )


def write_runtime_plot(path: Path, rows: list[dict]) -> None:
    plot_rows = [row for row in rows if row.get("openfield_seconds") is not None or row.get("fieldii_seconds") is not None]
    plot_rows.sort(key=lambda row: max(row.get("openfield_seconds") or 0.0, row.get("fieldii_seconds") or 0.0), reverse=True)
    if not plot_rows:
        write_empty_svg(path, "No runtime data found", "Run validation/run_openfield_benchmarks.py and validation/run_fieldii_benchmarks.m first.")
        return
    write_grouped_runtime_svg(path, plot_rows)


def write_speedup_plot(path: Path, rows: list[dict]) -> None:
    plot_rows = [row for row in rows if row.get("fieldii_over_openfield") is not None]
    plot_rows.sort(key=lambda row: row["fieldii_over_openfield"], reverse=True)
    if not plot_rows:
        write_empty_svg(path, "No speedup data found", "Run both benchmark timing scripts first.")
        return
    write_log_bar_svg(
        path,
        plot_rows,
        value_key="fieldii_over_openfield",
        title="Field II Runtime / OpenField Runtime",
        subtitle="End-to-end validation case script timing; values above 1 mean OpenField was faster.",
        x_label="runtime ratio",
        color="#4B6F44",
        reference_x=1.0,
    )


def write_log_bar_svg(
    path: Path,
    rows: list[dict],
    *,
    value_key: str,
    title: str,
    subtitle: str,
    x_label: str,
    color: str | None = None,
    color_key: str | None = None,
    color_map: dict[str, str] | None = None,
    reference_x: float | None = None,
    floor: float = EPS,
) -> None:
    margin_left = 330
    margin_right = 155
    margin_top = 86
    row_height = 25
    width = 1120
    plot_width = width - margin_left - margin_right
    height = margin_top + row_height * len(rows) + 58
    values = [max(float(row[value_key]), floor) for row in rows]
    min_log = math.floor(math.log10(min(values)))
    max_log = math.ceil(math.log10(max(values)))
    if reference_x is not None:
        min_log = min(min_log, math.floor(math.log10(reference_x)))
        max_log = max(max_log, math.ceil(math.log10(reference_x)))
    if min_log == max_log:
        max_log += 1

    def x_for(value: float) -> float:
        log_value = math.log10(max(value, floor))
        return margin_left + (log_value - min_log) / (max_log - min_log) * plot_width

    parts = svg_header(width, height)
    parts.append(text(24, 34, title, size=22, weight="700"))
    parts.append(text(24, 58, subtitle, size=12, fill="#555"))
    parts.append(f'<line x1="{margin_left}" y1="{margin_top - 12}" x2="{margin_left + plot_width}" y2="{margin_top - 12}" stroke="#d8d8d8"/>')
    for power in range(min_log, max_log + 1):
        x = x_for(10.0**power)
        parts.append(f'<line x1="{x:.2f}" y1="{margin_top - 18}" x2="{x:.2f}" y2="{height - 48}" stroke="#ececec"/>')
        parts.append(text(x, height - 28, f"1e{power}", size=11, anchor="middle", fill="#555"))
    if reference_x is not None:
        x = x_for(reference_x)
        parts.append(f'<line x1="{x:.2f}" y1="{margin_top - 20}" x2="{x:.2f}" y2="{height - 48}" stroke="#222" stroke-dasharray="4 3"/>')
        parts.append(text(x + 5, margin_top - 25, "1x", size=11, fill="#222"))

    for index, row in enumerate(rows):
        y = margin_top + index * row_height
        case_label = row["case"].replace("_", " ")
        parts.append(text(margin_left - 12, y + 14, case_label, size=11, anchor="end"))
        value = max(float(row[value_key]), floor)
        x0 = x_for(10.0**min_log)
        x1 = x_for(value)
        fill = color or (color_map or {}).get(row.get(color_key or ""), "#4F6D7A")
        bar_width = max(1.0, x1 - x0)
        parts.append(f'<rect x="{x0:.2f}" y="{y + 2}" width="{bar_width:.2f}" height="16" fill="{fill}"/>')
        parts.append(text(x1 + 6, y + 14, format_value(value), size=11, fill="#333"))
    parts.append(text(margin_left + plot_width / 2, height - 8, x_label, size=12, anchor="middle", fill="#333"))
    parts.append("</svg>\n")
    path.write_text("".join(parts), encoding="utf-8")


def write_grouped_runtime_svg(path: Path, rows: list[dict]) -> None:
    margin_left = 330
    margin_right = 150
    margin_top = 92
    row_height = 31
    width = 1120
    plot_width = width - margin_left - margin_right
    height = margin_top + row_height * len(rows) + 62
    values = []
    for row in rows:
        for key in ("fieldii_seconds", "openfield_seconds"):
            value = row.get(key)
            if value is not None:
                values.append(max(float(value), EPS))
    min_log = math.floor(math.log10(min(values)))
    max_log = math.ceil(math.log10(max(values)))
    if min_log == max_log:
        max_log += 1

    def x_for(value: float) -> float:
        log_value = math.log10(max(value, EPS))
        return margin_left + (log_value - min_log) / (max_log - min_log) * plot_width

    parts = svg_header(width, height)
    parts.append(text(24, 34, "Runtime: Field II vs OpenField", size=22, weight="700"))
    parts.append(text(24, 58, "Median end-to-end validation case runtime, including setup and output writing.", size=12, fill="#555"))
    parts.append(f'<rect x="770" y="23" width="12" height="12" fill="#6667AB"/>{text(788, 34, "Field II", size=12)}')
    parts.append(f'<rect x="846" y="23" width="12" height="12" fill="#D95D39"/>{text(864, 34, "OpenField", size=12)}')
    for power in range(min_log, max_log + 1):
        x = x_for(10.0**power)
        parts.append(f'<line x1="{x:.2f}" y1="{margin_top - 18}" x2="{x:.2f}" y2="{height - 50}" stroke="#ececec"/>')
        parts.append(text(x, height - 30, f"1e{power}", size=11, anchor="middle", fill="#555"))

    for index, row in enumerate(rows):
        y = margin_top + index * row_height
        parts.append(text(margin_left - 12, y + 18, row["case"].replace("_", " "), size=11, anchor="end"))
        for offset, key, fill in ((1, "fieldii_seconds", "#6667AB"), (15, "openfield_seconds", "#D95D39")):
            value = row.get(key)
            if value is None:
                continue
            x0 = x_for(10.0**min_log)
            x1 = x_for(float(value))
            parts.append(f'<rect x="{x0:.2f}" y="{y + offset}" width="{max(1.0, x1 - x0):.2f}" height="12" fill="{fill}"/>')
            parts.append(text(x1 + 6, y + offset + 10, format_seconds(float(value)), size=10, fill="#333"))
    parts.append(text(margin_left + plot_width / 2, height - 9, "seconds, log scale", size=12, anchor="middle", fill="#333"))
    parts.append("</svg>\n")
    path.write_text("".join(parts), encoding="utf-8")


def write_waveform_overlay(path: Path, case_names: list[str]) -> None:
    panels = []
    for name in case_names:
        fieldii_dir = RESULTS / name / "fieldii"
        openfield_dir = RESULTS / name / "openfield"
        if not (fieldii_dir / "samples.csv").exists() or not (openfield_dir / "samples.csv").exists():
            continue
        fieldii_time = load_csv(fieldii_dir / "time.csv").ravel()
        openfield_time = load_csv(openfield_dir / "time.csv").ravel()
        fieldii_samples = first_channel(load_csv(fieldii_dir / "samples.csv"))
        openfield_samples = first_channel(load_csv(openfield_dir / "samples.csv"))
        if fieldii_samples.size == 0 or openfield_samples.size == 0:
            continue
        panels.append((name, fieldii_time, fieldii_samples, openfield_time, openfield_samples))
    if not panels:
        write_empty_svg(path, "No waveform data found", "Generate validation outputs first.")
        return

    width = 1120
    panel_height = 210
    margin_left = 70
    margin_right = 34
    margin_top = 78
    plot_width = width - margin_left - margin_right
    height = margin_top + panel_height * len(panels) + 26
    parts = svg_header(width, height)
    parts.append(text(24, 34, "Waveform Overlays", size=22, weight="700"))
    parts.append(text(24, 58, "First output channel; amplitudes are normalized per case for visual comparison.", size=12, fill="#555"))
    parts.append(f'<path d="M 820 30 h 36" stroke="#6667AB" stroke-width="2"/>{text(864, 34, "Field II", size=12)}')
    parts.append(f'<path d="M 920 30 h 36" stroke="#D95D39" stroke-width="2"/>{text(964, 34, "OpenField", size=12)}')

    for index, (name, ft, fs, ot, os) in enumerate(panels):
        top = margin_top + index * panel_height
        bottom = top + panel_height - 45
        left = margin_left
        right = margin_left + plot_width
        all_time = np.concatenate([ft, ot]) * 1e6
        all_samples = np.concatenate([fs, os])
        t_min = float(np.min(all_time))
        t_max = float(np.max(all_time))
        amp = max(float(np.max(np.abs(all_samples))), EPS)
        y_min = -1.05
        y_max = 1.05

        def x_map(values: np.ndarray) -> np.ndarray:
            if t_max == t_min:
                return np.full_like(values, (left + right) / 2, dtype=np.float64)
            return left + (values * 1e6 - t_min) / (t_max - t_min) * plot_width

        def y_map(values: np.ndarray) -> np.ndarray:
            normalized = np.clip(values / amp, y_min, y_max)
            return bottom - (normalized - y_min) / (y_max - y_min) * (bottom - top)

        parts.append(f'<rect x="{left}" y="{top}" width="{plot_width}" height="{bottom - top}" fill="#fff" stroke="#ddd"/>')
        parts.append(f'<line x1="{left}" y1="{(top + bottom) / 2:.2f}" x2="{right}" y2="{(top + bottom) / 2:.2f}" stroke="#eee"/>')
        parts.append(text(left, top - 9, name.replace("_", " "), size=13, weight="700"))
        parts.append(polyline(x_map(ft), y_map(fs), "#6667AB"))
        parts.append(polyline(x_map(ot), y_map(os), "#D95D39"))
        parts.append(text(left, bottom + 20, f"{t_min:.3f} us", size=10, fill="#555"))
        parts.append(text(right, bottom + 20, f"{t_max:.3f} us", size=10, anchor="end", fill="#555"))
    parts.append("</svg>\n")
    path.write_text("".join(parts), encoding="utf-8")


def first_channel(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return data
    return data[:, 0]


def write_empty_svg(path: Path, title: str, subtitle: str) -> None:
    width = 900
    height = 220
    parts = svg_header(width, height)
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
    escaped = html.escape(content)
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="{SVG_FONT}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{escaped}</text>'
    )


def polyline(x: np.ndarray, y: np.ndarray, color: str) -> str:
    points = " ".join(f"{float(px):.2f},{float(py):.2f}" for px, py in zip(x, y, strict=True))
    return f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>'


def format_value(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e3:
        return f"{value:.2e}"
    return f"{value:.3g}"


def format_seconds(value: float) -> str:
    if value < 1e-3:
        return f"{value * 1e6:.1f} us"
    if value < 1.0:
        return f"{value * 1e3:.1f} ms"
    return f"{value:.2f} s"


if __name__ == "__main__":
    main()
