# Field II Validation

This folder contains paired MATLAB Field II scripts and Python `openfield`
scripts for numerical validation.

The validation flow is intentionally separate from the normal unit tests:

1. Generate Field II reference outputs with MATLAB.
2. Generate matching `openfield` outputs with Python.
3. Compare the two output folders with configurable tolerances.

Generated files are written to `validation/results/`, which is ignored by git.

## Requirements

- MATLAB with Field II on the MATLAB path, or set `FIELDII_PATH`.
- Python with `openfield` importable from this repository.
- NumPy for the Python validation scripts.

The bundled Field II folder is used by default:

```text
supplementary/field2/Field_II_ver_3_30_windows(1)
```

You can override it:

```powershell
$env:FIELDII_PATH = "C:\path\to\field_II"
```

## Run

Generate Field II references from MATLAB:

```powershell
matlab -batch "run('validation/run_fieldii_cases.m')"
```

Generate `openfield` outputs:

```powershell
python validation/run_openfield_cases.py
```

Compare outputs:

```powershell
python validation/compare_outputs.py --all
```

During early development, cases can be marked as expected failures in
`validation/cases.json`. Use strict mode when you want a non-zero exit code for
those differences:

```powershell
python validation/compare_outputs.py --all --strict
```

List cases:

```powershell
python validation/run_openfield_cases.py --list
```

## Current Cases

- `piston_spatial_impulse`: Field II `calc_h` vs `Simulation.spatial_impulse_response`.
- `linear_array_spatial_impulse`: Field II `calc_h` on a focused linear array vs `Simulation.spatial_impulse_response`.
- `linear_array_emitted_pressure`: Field II `calc_hp` vs `Simulation.emitted_pressure`.
- `linear_array_geometry`: Field II `xdc_get(..., 'rect')` derived geometry vs `LinearArray`.
- `two_dimensional_array_geometry`: Field II `xdc_get(..., 'rect')` derived geometry vs `TwoDimensionalArray`.

The physics comparisons are expected to become stricter over time. The current
CPU reference implementation is deliberately simple, so validation may expose
known differences from Field II before the numerical model is complete. Those
known differences are tracked as expected failures in `validation/cases.json`.
