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

The outputs are written under `validation/results/kernel_modes/`.

Current purpose:

- Compare Field II `fast_integration=0` and `fast_integration=1` for triangle
  and line-bounded apertures.
- Compare Field II `accurate_time_calc=0` and `accurate_time_calc=1` for the
  remaining curved rectangular aperture cases.
- Save a few isolated curved subelement responses using `ele_apodization`, so
  the rectangle projection/kernel mismatch can be fitted one patch at a time.
