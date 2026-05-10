# Kernel Mode Diagnostics

These scripts generate and summarize Field II reference outputs for kernel
settings that are intentionally not part of the main validation matrix yet.

Run from MATLAB:

```matlab
run_kernel_mode_diagnostics('C:\Users\fabia\PycharmProjects\openfield')
```

Then summarize from the repo root:

```powershell
uv run --no-project --with numpy python validation\diagnostics\analyze_kernel_modes.py
```

To generate every isolated curved rectangle instead of the default three
representative patches per aperture:

```matlab
run_kernel_mode_diagnostics('C:\Users\fabia\PycharmProjects\openfield', true)
```

Then compare the isolated patches against openfield:

```powershell
uv run --no-project --with numpy --with-editable . python validation\diagnostics\analyze_curved_rectangle_isolates.py
```

The isolate analyzer reports the worst patch residuals, local sample/point of
the maximum error, integer sample-shift checks, geometry descriptors, and a
small fitted half-width ratio tuple `(x_ratio, y_ratio, max_abs)` for each
listed patch. Those fitted ratios are diagnostic only; they should not be used
as a production correction unless they are consistent across aperture families.

The outputs are written under `validation/results/kernel_modes/`.

Current purpose:

- Compare Field II `fast_integration=0` and `fast_integration=1` for triangle
  and line-bounded apertures.
- Compare Field II `accurate_time_calc=0` and `accurate_time_calc=1` for the
  remaining curved rectangular aperture cases.
- Save a few isolated curved subelement responses using `ele_apodization`, so
  the rectangle projection/kernel mismatch can be fitted one patch at a time.
