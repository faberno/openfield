from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "validation" / "results"
KERNEL_RESULTS = RESULTS / "kernel_modes"


def main() -> None:
    if not KERNEL_RESULTS.exists():
        print(
            "No kernel diagnostics found. Run "
            "validation/diagnostics/run_kernel_mode_diagnostics.m from MATLAB first.",
        )
        return

    _summarize_polygon_modes()
    _summarize_accurate_time_modes()
    _summarize_isolated_curved_subelements()


def _summarize_polygon_modes() -> None:
    cases = {
        "triangle": "triangle_aperture_spatial_impulse",
        "line": "line_bounded_aperture_spatial_impulse",
    }
    print("Polygon kernel modes")
    for primitive, validation_case in cases.items():
        openfield_dir = RESULTS / validation_case / "openfield"
        if not openfield_dir.exists():
            print(f"  {primitive}: missing openfield validation output")
            continue
        openfield = _read_response(openfield_dir)
        for fast in (0, 1):
            diagnostic_dir = KERNEL_RESULTS / f"{primitive}_fast{fast}_fs100MHz" / "fieldii"
            if not diagnostic_dir.exists():
                print(f"  {primitive} fast={fast}: missing Field II diagnostic")
                continue
            fieldii = _read_response(diagnostic_dir)
            difference = _aligned_difference(openfield, fieldii)
            print(
                f"  {primitive} fast={fast}: "
                f"max_abs={np.max(np.abs(difference)):.6g}, "
                f"openfield_sum={np.sum(openfield.samples, axis=0)}, "
                f"fieldii_sum={np.sum(fieldii.samples, axis=0)}",
            )


def _summarize_accurate_time_modes() -> None:
    print("Curved rectangle accurate_time_calc modes")
    for case in (
        "convex_focused_array",
        "convex_focused_multirow_array",
        "concave_piston",
    ):
        mode0_dir = KERNEL_RESULTS / f"{case}_accurate0" / "fieldii"
        mode1_dir = KERNEL_RESULTS / f"{case}_accurate1" / "fieldii"
        if not mode0_dir.exists() or not mode1_dir.exists():
            print(f"  {case}: missing one or both accurate_time_calc diagnostics")
            continue
        mode0 = _read_response(mode0_dir)
        mode1 = _read_response(mode1_dir)
        difference = _aligned_difference(mode0, mode1)
        print(
            f"  {case}: "
            f"mode0_vs_mode1_max_abs={np.max(np.abs(difference)):.6g}, "
            f"mode0_sum={np.sum(mode0.samples, axis=0)}, "
            f"mode1_sum={np.sum(mode1.samples, axis=0)}",
        )


def _summarize_isolated_curved_subelements() -> None:
    isolated = sorted(KERNEL_RESULTS.glob("*_isolated_rect*"))
    print("Isolated curved subelements")
    if not isolated:
        print("  missing isolated subelement diagnostics")
        return
    for path in isolated:
        response_dir = path / "fieldii"
        if not response_dir.exists():
            continue
        response = _read_response(response_dir)
        print(
            f"  {path.name}: "
            f"shape={response.samples.shape}, "
            f"start={response.time[0]:.12g}, "
            f"sum={np.sum(response.samples, axis=0)}",
        )


class Response:
    def __init__(self, samples: np.ndarray, time: np.ndarray) -> None:
        self.samples = samples
        self.time = time
        self.sampling_frequency = 1.0 / float(time[1] - time[0]) if len(time) > 1 else 1.0


def _read_response(path: Path) -> Response:
    samples = np.loadtxt(path / "samples.csv", delimiter=",")
    if samples.ndim == 1:
        samples = samples[:, None]
    time = np.loadtxt(path / "time.csv", delimiter=",")
    if time.ndim == 0:
        metadata = _read_metadata(path)
        time = float(metadata["start_time"]) + np.arange(samples.shape[0], dtype=np.float64) / float(
            metadata["sampling_frequency"],
        )
    return Response(samples=samples, time=time)


def _read_metadata(path: Path) -> dict:
    import json

    return json.loads((path / "metadata.json").read_text())


def _aligned_difference(left: Response, right: Response) -> np.ndarray:
    sampling_frequency = left.sampling_frequency
    if not np.isclose(sampling_frequency, right.sampling_frequency):
        raise ValueError("responses must use the same sampling frequency")
    start = max(float(left.time[0]), float(right.time[0]))
    end = min(float(left.time[-1]), float(right.time[-1]))
    left_start = int(round((start - float(left.time[0])) * sampling_frequency))
    right_start = int(round((start - float(right.time[0])) * sampling_frequency))
    count = int(round((end - start) * sampling_frequency)) + 1
    return (
        left.samples[left_start : left_start + count]
        - right.samples[right_start : right_start + count]
    )


if __name__ == "__main__":
    main()
