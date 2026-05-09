# Validation Debt

This file tracks Field II parity gaps that are known, intentional, or still
missing dedicated validation coverage.

## Spatial Impulse Response (`calc_h`)

Status: strict Field II validation passes for flat rectangular apertures,
including linear, multirow, sparse 2D, and imported rectangle apertures.
Curved and non-rectangular primitive kernels still have known gaps.

Validated so far:

- Field II-compatible output time windows are implemented for rectangular
  subelements, focus timelines, explicit delay timelines, and subelement delays.
- The rectangle kernel uses Field II's far-field projected-rectangle rule with
  sample-centered bin integration.
- Triangle and line-bounded primitives use an event-split flat-polygon
  integration path. This keeps triangle and line descriptions of the same flat
  polygon numerically consistent, but it intentionally does not emulate Field
  II's default `fast_integration=1` approximation yet.
- Rigid and soft baffle `calc_h` cases pass strict comparison.

Strict validation now passes for:

- `focused_linear_array_geometry`
- `focused_linear_array_spatial_impulse`
- `focused_multirow_array_geometry`
- `focused_multirow_array_spatial_impulse`
- `convex_array_geometry`
- `convex_array_spatial_impulse`
- `convex_focused_array_geometry`
- `convex_focused_multirow_array_geometry`
- `concave_piston_geometry`
- `linear_array_focus_timeline_spatial_impulse`
- `linear_array_delay_timeline_spatial_impulse`
- `linear_array_apodization_spatial_impulse`
- `linear_array_soft_baffle_spatial_impulse`
- `linear_array_subelement_apodization_spatial_impulse`
- `linear_array_subelement_delay_spatial_impulse`
- `linear_array_spatial_impulse`
- `linear_multirow_array_spatial_impulse`
- `piston_spatial_impulse`
- `rectangle_aperture_spatial_impulse`
- `single_rectangle_spatial_impulse`
- `two_dimensional_array_spatial_impulse`

Broader aperture-family validation cases now exist, but are expected failures:

- `convex_focused_array_spatial_impulse`: combined convex/elevation-focused
  geometry and focus-delay parity passes; the remaining difference is the
  undocumented curved-rectangle edge-sample kernel used by `calc_h`.
- `convex_focused_multirow_array_spatial_impulse`: combined convex/elevation
  multirow geometry and focus-delay parity passes; the remaining difference is
  the undocumented curved-rectangle edge-sample kernel used by `calc_h`.
- `concave_piston_spatial_impulse`: curved-surface geometry/kernel parity is
  close but not strict.
- `triangle_aperture_spatial_impulse`: the openfield path uses an
  area-conserving, event-split flat-polygon kernel; Field II's default fast
  triangle integration intentionally trades shape accuracy for speed at low
  sampling rates.
- `line_bounded_aperture_spatial_impulse`: the openfield path uses an
  area-conserving, event-split flat-polygon kernel; Field II's default fast line
  integration intentionally trades shape accuracy for speed at low sampling
  rates.

Additional diagnostics live in `validation/diagnostics/` and write Field II
kernel-mode outputs under `validation/results/kernel_modes/`. They currently
capture `fast_integration=0/1`, `accurate_time_calc=0/1`, and selected isolated
curved subelements for fitting the remaining rectangle-kernel gap.

## Pulse-Echo And Scatterer Responses

Status: strict small-case Field II validation passes.

Initial validation goal:

- Field II and openfield produce matching result shapes and values for small
  transmit/receive/scatterer cases.
- Python unit tests verify convolution, scatterer weighting, channel ordering,
  and decimation behavior independently of Field II.

Remaining risk:

- Broader pulse-echo/scatterer coverage is still needed for more aperture
  families and nontrivial channel configurations.

## Per-Element Waveforms

Status: strict small-case Field II validation passes.

Remaining risk:

- Broader coverage is still needed for sparse element selections and
  combinations with nontrivial impulse responses.
