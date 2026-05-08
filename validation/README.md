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
- `single_rectangle_spatial_impulse`: isolated one-rectangle Field II `calc_h` vs `Simulation.spatial_impulse_response`.
- `linear_array_spatial_impulse`: Field II `calc_h` on a focused linear array vs `Simulation.spatial_impulse_response`.
- `linear_array_emitted_pressure`: Field II `calc_hp` vs `Simulation.emitted_pressure`.
- `linear_array_focus_timeline_spatial_impulse`: Field II `xdc_focus` timeline vs `FocusTimeline`.
- `linear_array_delay_timeline_spatial_impulse`: Field II `xdc_focus_times` vs `DelayTimeline`.
- `linear_array_apodization_spatial_impulse`: Field II `xdc_apodization` vs `Apodization`.
- `linear_array_soft_baffle_spatial_impulse`: Field II `xdc_baffle(..., soft=1)` vs `Baffle("soft")`.
- `linear_array_subelement_apodization_spatial_impulse`: Field II `ele_apodization` vs `SubElementApodization`.
- `linear_array_subelement_delay_spatial_impulse`: Field II `ele_delay` vs `SubElementDelays`.
- `linear_array_pulse_echo`: Field II `calc_hhp` vs `Simulation.pulse_echo_response`.
- `linear_array_scatterer_response`: Field II `calc_scat` vs `Simulation.scatterer_response`.
- `linear_array_receive_channels`: Field II `calc_scat_multi` vs `Simulation.receive_channel_responses`.
- `linear_array_full_matrix_capture`: Field II `calc_scat_all` vs `Simulation.full_matrix_capture`.
- `linear_array_geometry`: Field II `xdc_get(..., 'rect')` derived geometry vs `LinearArray`.
- `two_dimensional_array_geometry`: Field II `xdc_get(..., 'rect')` derived geometry vs `TwoDimensionalArray`.
- `rectangle_aperture_geometry`: Field II `xdc_rectangles` geometry vs `RectangleAperture.from_fieldii_rectangles`.

The physics comparisons are expected to become stricter over time. The current
CPU reference implementation is deliberately simple, so validation may expose
known differences from Field II before the numerical model is complete. Those
known differences are tracked as expected failures in `validation/cases.json`.
