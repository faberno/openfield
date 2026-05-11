from __future__ import annotations

import argparse
import json
import runpy
import statistics
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = ROOT / "validation" / "cases"
RESULTS = ROOT / "validation" / "results"

DEFAULT_CASES = [
    "single_rectangle_spatial_impulse",
    "linear_array_spatial_impulse",
    "triangle_aperture_spatial_impulse",
    "linear_array_emitted_pressure",
    "linear_array_pulse_echo",
    "linear_array_full_matrix_capture",
    "linear_array_geometry",
    "rectangle_aperture_geometry",
]


def load_cases() -> list[dict]:
    return json.loads((ROOT / "validation" / "cases.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Time openfield validation cases.")
    parser.add_argument("cases", nargs="*", help="Case names to benchmark.")
    parser.add_argument("--all", action="store_true", help="Benchmark every case from cases.json.")
    parser.add_argument("--repeats", type=int, default=1, help="Number of end-to-end repeats per case.")
    parser.add_argument("--list-default", action="store_true", help="List the default benchmark cases and exit.")
    args = parser.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    case_names = [case["name"] for case in load_cases()]
    known = set(case_names)
    if args.list_default:
        for name in DEFAULT_CASES:
            print(name)
        return

    selected = case_names if args.all else (args.cases or DEFAULT_CASES)
    unknown = sorted(set(selected) - known)
    if unknown:
        raise SystemExit(f"Unknown validation case(s): {', '.join(unknown)}")

    for name in selected:
        script = CASE_ROOT / name / "openfield_case.py"
        if not script.exists():
            raise SystemExit(f"Missing openfield validation script: {script}")
        print(f"Benchmarking openfield validation case: {name}")
        elapsed_seconds = []
        for repeat in range(args.repeats):
            start = time.perf_counter()
            runpy.run_path(str(script), run_name="__main__")
            elapsed = time.perf_counter() - start
            elapsed_seconds.append(elapsed)
            print(f"  repeat {repeat + 1}/{args.repeats}: {elapsed:.6g} s")
        write_runtime(name, elapsed_seconds)


def write_runtime(case_name: str, elapsed_seconds: list[float]) -> None:
    output_dir = RESULTS / case_name / "openfield"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "case": case_name,
        "source": "openfield",
        "runtime_scope": "validation case script, including setup and output writing",
        "repeats": len(elapsed_seconds),
        "elapsed_seconds": elapsed_seconds,
        "min_seconds": min(elapsed_seconds),
        "median_seconds": statistics.median(elapsed_seconds),
        "mean_seconds": statistics.fmean(elapsed_seconds),
    }
    (output_dir / "runtime.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
