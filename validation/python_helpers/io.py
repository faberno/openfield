from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def validation_results_dir(case_name: str, source: str) -> Path:
    return repo_root() / "validation" / "results" / case_name / source


def write_time_response(case_name: str, source: str, response, *, metadata: dict | None = None) -> None:
    output_dir = validation_results_dir(case_name, source)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = np.asarray(response.samples, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples[:, None]
    np.savetxt(output_dir / "samples.csv", samples, delimiter=",")
    np.savetxt(output_dir / "time.csv", np.asarray(response.time, dtype=np.float64), delimiter=",")
    payload = {
        "case": case_name,
        "source": source,
        "kind": "time_response",
        "sampling_frequency": float(response.sampling_frequency),
        "start_time": float(response.start_time),
        "samples_shape": list(samples.shape),
    }
    if metadata:
        payload.update(metadata)
    _write_json(output_dir / "metadata.json", payload)


def write_geometry(case_name: str, source: str, aperture, *, metadata: dict | None = None) -> None:
    output_dir = validation_results_dir(case_name, source)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "centers": aperture.physical_centers,
        "subelement_centers": aperture.centers,
        "normals": aperture.normals,
        "areas": aperture.areas[:, None],
        "physical_indices": aperture.physical_indices[:, None],
        "subelement_indices": aperture.subelement_indices[:, None],
    }
    for name, values in arrays.items():
        np.savetxt(output_dir / f"{name}.csv", np.asarray(values), delimiter=",")
    payload = {
        "case": case_name,
        "source": source,
        "kind": "geometry",
        "physical_element_count": int(aperture.physical_element_count),
        "subelement_count": int(len(aperture.elements)),
    }
    if metadata:
        payload.update(metadata)
    _write_json(output_dir / "metadata.json", payload)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
