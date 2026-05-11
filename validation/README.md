# Field II Validation

This folder contains paired MATLAB Field II scripts and Python `openfield`
scripts for compatibility validation.

Field II remains useful for checking aperture geometry, focusing conventions,
waveform plumbing, and broad response scale. It is no longer the pass/fail
oracle for time-domain spatial impulse responses: `openfield` intentionally
uses exact flat-facet integration instead of Field II's far-field rectangle and
sample-window approximations.

The validation flow is intentionally separate from the normal unit tests:

1. Generate Field II reference outputs with MATLAB.
2. Generate matching `openfield` outputs with Python.
3. Compare the two output folders with configurable tolerances. Time-response
   differences are reported as reference-only expected failures unless strict
   mode is requested.

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
`validation/cases.json`. Time-response cases are also treated as Field II
reference-only by default. Use strict mode when you want a non-zero exit code
for those differences:

```powershell
python validation/compare_outputs.py --all --strict
```

List cases:

```powershell
python validation/run_openfield_cases.py --list
```

## Benchmark Plots

The benchmark utilities time a representative subset of validation cases and
turn the saved Field II/openfield CSV files into SVG plots. Timings are
end-to-end validation-script timings, including setup and output writing.

```powershell
uv run --no-project --with-editable . python validation/run_openfield_benchmarks.py --repeats 3
matlab -batch "addpath('validation'); run_fieldii_benchmarks([], 1)"
uv run --no-project --with numpy python validation/plot_benchmarks.py --all
```

Generated plots and `benchmark_summary.csv` are written to
`validation/results/plots/`.

## Convergence Studies

The convergence utilities use a fine OpenField result as the reference instead
of treating Field II as the oracle. This is useful for cases where OpenField's
exact facet kernel intentionally differs from Field II's compatibility kernel.

```powershell
uv run --no-project --with-editable . python validation/concave_convergence.py
```

By default this compares the Field II-style Cartesian concave mesh with the new
boundary-fitted polar mesh and uses a fine polar OpenField solve as the
reference. Use `--tessellations cartesian` or `--tessellations polar` to narrow
the run.

Other curved aperture families expose the same idea as opt-in geometry modes:
`Piston.adaptive(...)`, `ConcavePiston.adaptive(...)`, and
`tessellation="adaptive"` for focused/convex array constructors.

Generated CSV and SVG files are written to
`validation/results/convergence/concave_piston/`.

Kernel-mode probes that are not part of the main pass/fail matrix live in
`validation/diagnostics/`. They are useful for investigating Field II's
`fast_integration` and `accurate_time_calc` switches:

```powershell
matlab -batch "addpath('validation/diagnostics'); run_kernel_mode_diagnostics(pwd)"
uv run --no-project --with numpy python validation/diagnostics/analyze_kernel_modes.py
```

## Current Cases

- `piston_spatial_impulse`: Field II `calc_h` vs `Simulation.spatial_impulse_response`.
- `single_rectangle_spatial_impulse`: isolated one-rectangle Field II `calc_h` vs `Simulation.spatial_impulse_response`.
- `linear_array_spatial_impulse`: Field II `calc_h` on a focused linear array vs `Simulation.spatial_impulse_response`.
- `focused_linear_array_spatial_impulse`: Field II `xdc_focused_array` vs `FocusedLinearArray`.
- `linear_multirow_array_spatial_impulse`: Field II `xdc_linear_multirow` vs `LinearMultirowArray`.
- `focused_multirow_array_spatial_impulse`: Field II `xdc_focused_multirow` vs `FocusedMultirowArray`.
- `convex_array_spatial_impulse`: Field II `xdc_convex_array` vs `ConvexArray`.
- `convex_focused_array_spatial_impulse`: Field II `xdc_convex_focused_array` vs `ConvexFocusedArray`.
- `convex_focused_multirow_array_spatial_impulse`: Field II `xdc_convex_focused_multirow` vs `ConvexFocusedMultirowArray`.
- `two_dimensional_array_spatial_impulse`: Field II `xdc_2d_array` vs `TwoDimensionalArray`.
- `concave_piston_spatial_impulse`: Field II `xdc_concave` vs `ConcavePiston`.
- `rectangle_aperture_spatial_impulse`: Field II `xdc_rectangles` vs `RectangleAperture`.
- `triangle_aperture_spatial_impulse`: Field II `xdc_triangles` vs `TriangleAperture`.
- `line_bounded_aperture_spatial_impulse`: Field II `xdc_lines` vs `LineBoundedAperture`.
- `linear_array_emitted_pressure`: Field II `calc_hp` vs `Simulation.emitted_pressure`.
- `linear_array_element_waveforms`: Field II `ele_waveform`/`calc_hp` vs per-element `Waveform`s.
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
- `focused_linear_array_geometry`: Field II `xdc_get(..., 'rect')` derived geometry vs `FocusedLinearArray`.
- `focused_multirow_array_geometry`: Field II `xdc_get(..., 'rect')` derived geometry vs `FocusedMultirowArray`.
- `convex_array_geometry`: Field II `xdc_get(..., 'rect')` derived geometry vs `ConvexArray`.
- `convex_focused_array_geometry`: Field II `xdc_get(..., 'rect')` derived geometry vs `ConvexFocusedArray`.
- `convex_focused_multirow_array_geometry`: Field II `xdc_get(..., 'rect')` derived geometry vs `ConvexFocusedMultirowArray`.
- `concave_piston_geometry`: Field II `xdc_get(..., 'rect')` derived geometry vs `ConcavePiston`.
- `two_dimensional_array_geometry`: Field II `xdc_get(..., 'rect')` derived geometry vs `TwoDimensionalArray`.
- `rectangle_aperture_geometry`: Field II `xdc_rectangles` geometry vs `RectangleAperture.from_fieldii_rectangles`.

Some broader aperture-family cases are currently marked as expected failures in
`validation/cases.json`. Coverage gaps and missing oracle cases are tracked in
`docs/validation_debt.md`.
