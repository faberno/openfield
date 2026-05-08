# Field II Coverage Overview

This document tracks the Field II files bundled in
`supplementary/field2/Field_II_ver_3_30_windows(1)` and their intended
`openfield` equivalents.

Field II is used as a feature reference, not as an API style. `openfield`
should not expose MATLAB-style global functions such as `field_init`,
`set_field`, or `xdc_*`.

## Status Legend

- **Implemented**: an `openfield` equivalent exists in the package.
- **Partial**: the core concept exists, but Field II behavior is not fully covered.
- **Planned**: no implementation yet, but it belongs in `openfield`.
- **Not applicable**: Field II runtime/UI/handle-management behavior that should not
  have a direct public equivalent in `openfield`.

## Runtime And Configuration

| Field II file | Field II functionality | `openfield` equivalent | Status | Notes |
|---|---|---|---|---|
| `field_init.m` | Initialize hidden global Field II engine state | `Simulation(...)` | Implemented | `openfield` uses explicit simulation objects instead of global state. |
| `field_end.m` | Terminate Field II and release global storage | none | Not applicable | Python object lifetime and garbage collection replace this. |
| `field_info.m` | Print current Field II engine settings | future `Simulation.info()` / object repr | Planned | Useful, but should be object-based. |
| `field_debug.m` | Enable Field II debug output | future logging configuration | Planned | Should use Python logging or diagnostics, not global debug flags. |
| `field_guide.m` | Open bundled PDF guide | documentation files | Not applicable | Runtime package should not open external PDF viewers. |
| `field_logo.m` | Show Field II logo | none | Not applicable | UI branding helper, not simulation functionality. |
| `set_field.m` | Set sound speed, sampling rate, attenuation, primitive mode, debug, nonlinear params | `Simulation`, `Medium`, future physics configs | Partial | Sound speed, sampling rate, and basic attenuation fields exist. Primitive mode/debug/nonlinear behavior is not implemented. |
| `set_sampling.m` | Set sampling frequency | `Simulation(sampling_frequency=...)` | Implemented | Explicit constructor parameter. |

## Aperture Constructors

| Field II file | Field II functionality | `openfield` equivalent | Status | Notes |
|---|---|---|---|---|
| `xdc_piston.m` | Flat circular piston aperture | `Piston` | Implemented | Geometry/tessellation implemented. |
| `xdc_concave.m` | Concave/focused circular aperture | `ConcavePiston` | Implemented | Geometry/tessellation implemented. |
| `xdc_linear_array.m` | Flat linear array | `LinearArray` | Implemented | Geometry/tessellation implemented. |
| `xdc_focused_array.m` | Linear array with mechanical elevation focus | `FocusedLinearArray` | Implemented | Geometry/tessellation implemented. |
| `xdc_linear_multirow.m` | Flat multirow linear array | `LinearMultirowArray` | Implemented | Geometry/tessellation implemented. |
| `xdc_focused_multirow.m` | Multirow linear array with mechanical elevation focus | `FocusedMultirowArray` | Implemented | Geometry/tessellation implemented. |
| `xdc_convex_array.m` | Convex linear array | `ConvexArray` | Implemented | Geometry/tessellation implemented. |
| `xdc_convex_focused_array.m` | Convex array with mechanical elevation focus | `ConvexFocusedArray` | Implemented | Geometry/tessellation implemented. |
| `xdc_convex_focused_multirow.m` | Convex multirow array with mechanical elevation focus | `ConvexFocusedMultirowArray` | Implemented | Geometry/tessellation implemented. |
| `xdc_2d_array.m` | Sparse/full 2D matrix array | `TwoDimensionalArray`, `Array2D` | Implemented | Geometry/tessellation implemented with `enabled` mask. |
| `xdc_rectangles.m` | Aperture from rectangular patches | `RectangleAperture`, `RectangleAperture.from_fieldii_rectangles()` | Implemented | Field II geometry validation passes for a custom rectangle aperture. |
| `xdc_triangles.m` | Aperture from triangular patches | `TriangleAperture`, `TriangleAperture.from_fieldii_triangles()` | Partial | Parser/exporter and triangulation exist; strict Field II geometry validation still needs a triangle-specific oracle. |
| `xdc_lines.m` | Flat aperture from line-bounded half planes | `LineBoundedAperture`, `LineBoundedAperture.from_fieldii_lines()` | Partial | Half-plane clipping and Field II-format import/export exist; needs validation against Field II examples. |

## Aperture Configuration And Manipulation

| Field II file | Field II functionality | `openfield` equivalent | Status | Notes |
|---|---|---|---|---|
| `xdc_focus.m` | Fixed focus timeline from focal points | `FixedFocus`, `FocusTimeline`, `Aperture.focused_at()`, `Aperture.with_focus_timeline()` | Partial | Fixed focus and time-indexed focal points exist; CPU reference uses the active row at time zero. |
| `xdc_center_focus.m` | Reference origin for focus delay calculation | `FixedFocus(origin=...)` | Implemented | Folded into focus object rather than aperture-global state. |
| `xdc_dynamic_focus.m` | Dynamic focusing from direction angles and start time | `DynamicFocus`, `Aperture.with_dynamic_focus()` | Partial | Far-field steering delay model exists; full receive-time dynamic focusing is not implemented. |
| `xdc_focus_times.m` | User-supplied focus delay timeline | `DelayTimeline`, `Aperture.with_delay_timeline()` | Partial | Container and CPU time-zero use exist; Field II time-axis parity still has a known mismatch. |
| `xdc_times_focus.m` | Alias/duplicate for user-supplied focus delay timeline | `DelayTimeline`, `Aperture.with_delay_timeline()` | Partial | Field II file duplicates `xdc_focus_times` behavior. |
| `xdc_apodization.m` | Physical-element apodization timeline | `Apodization`, `Aperture.with_apodization()` | Partial | Timeline container exists and CPU `calc_h` applies the active row at time zero. |
| `xdc_impulse.m` | Aperture impulse response | `Aperture.with_impulse_response()`, `Waveform`, `ToneBurst` | Partial | Storage/API exists. Pressure/scatter convolution is not implemented yet. |
| `xdc_excitation.m` | Aperture excitation pulse | `Aperture.with_excitation()`, `Waveform`, `ToneBurst` | Partial | Storage/API exists. Pressure/scatter convolution is not implemented yet. |
| `xdc_baffle.m` | Rigid or soft baffle condition | `Baffle`, `Aperture.with_baffle()` | Partial | Rigid default and soft cosine-obliquity path exist; exact Field II soft-baffle parity is unresolved. |
| `xdc_quantization.m` | Quantize focus delays | `FixedFocus(quantization=...)` | Partial | Implemented for fixed-focus delays only. |
| `xdc_convert.m` | Convert rectangular aperture description to triangles | `Aperture.to_triangles()` | Partial | Polygon fan triangulation exists; Field II-specific conversion ordering is not validated. |
| `xdc_line_convert.m` | Convert rectangular aperture to line-bounded representation | future conversion utility | Planned | Lower priority unless needed for validation/parity. |
| `xdc_get.m` | Return aperture geometry/focus/apodization data | aperture properties plus Field II-format exporters | Partial | Geometry properties and rectangle/triangle/line exporters exist; full Field II focus/apodization export is not complete. |
| `xdc_show.m` | Print aperture information | `Aperture.show_3d()`, future summary/repr | Partial | 3D visualization exists; textual summary is not implemented. |
| `xdc_free.m` | Free one aperture handle | none | Not applicable | No global handle registry in public API. |
| `ele_apodization.m` | Per-subelement apodization within physical elements | `SubElementApodization`, `Aperture.with_subelement_apodization()` | Partial | CPU `calc_h` applies one weight per subelement in aperture order; Field II value parity remains blocked by `calc_h`. |
| `ele_delay.m` | Per-subelement delay within physical elements | `SubElementDelays`, `Aperture.with_subelement_delays()` | Partial | CPU `calc_h` applies one extra delay per subelement; Field II time-axis parity has a known mismatch. |
| `ele_waveform.m` | Per-physical-element waveform | `ElementWaveforms`, `Aperture.with_element_waveforms()` | Partial | CPU emitted-pressure path supports per-element transmit waveforms; Field II validation still needs a dedicated oracle case. |

## Field And Signal Calculations

| Field II file | Field II functionality | `openfield` equivalent | Status | Notes |
|---|---|---|---|---|
| `calc_h.m` | Spatial impulse response | `Simulation.spatial_impulse_response()` / `physics.spatial_impulse_response()` | Partial | CPU far-field rectangle path exists with Field-II-compatible time windows; value parity is still in progress. |
| `calc_hp.m` | Emitted pressure field | `Simulation.emitted_pressure()` / `physics.emitted_pressure()` | Partial | CPU reference path exists with excitation and impulse-response convolution; needs validation and GPU backend. |
| `calc_hhp.m` | Pulse-echo field | `Simulation.pulse_echo_response()` | Implemented | Strict small-case Field II validation passes. |
| `calc_scat.m` | Received RF from scatterers | `Simulation.scatterer_response()` | Implemented | Strict small-case Field II validation passes. |
| `calc_scat_multi.m` | Scatterer response for each receive element | `Simulation.receive_channel_responses()` | Implemented | Strict small-case Field II validation passes. |
| `calc_scat_all.m` | Full transmit/receive synthetic aperture raw data | `Simulation.full_matrix_capture()` | Implemented | Strict small-case Field II validation passes, including Field II channel ordering and sample trimming. |

## Bundled Non-Wrapper Files

| Field II file | Purpose | `openfield` equivalent | Status | Notes |
|---|---|---|---|---|
| `Mat_field.mexw64` | Compiled Field II C/MEX engine | none | Not applicable | Not reusable for open-source Python internals. Can be used only as a validation oracle if MATLAB is available. |
| `users_guide.pdf` | Field II user guide | `docs/field2_coverage.md`, future docs | Partial | Used as feature reference. |
| `logo_field.mat` | Field II logo data | none | Not applicable | UI asset only. |

## Current High-Level Coverage

Implemented now:

- Explicit `Simulation` and `Medium` configuration.
- All Field II aperture constructor families at the geometry/tessellation level.
- Strict Field II geometry validation for linear and sparse 2D rectangular arrays.
- Strict Field II geometry validation for a custom rectangle aperture.
- `SubElement` geometry model with centers, normals, areas, physical indices, and polygon vertices.
- Fixed focus, focus timelines, explicit delay timelines, dynamic-focus container, apodization timeline, waveform container, and tone burst helper.
- CPU reference spatial impulse response shape/time-axis path.
- CPU reference emitted pressure path using excitation and impulse-response convolution.
- CPU reference pulse-echo and scatterer response paths, including receive-channel and full-matrix capture outputs.
- Baffle, per-subelement apodization/delay, and per-element waveform containers with CPU reference hooks.
- Vedo-based aperture visualization via `Aperture.show_3d()`.

Major missing areas:

- Full dynamic receive focusing semantics.
- Field II parity for soft baffles, subelement delay time windows, and per-element waveform validation.
- Attenuation and baffle physics in calculations.
- Field II-format import/export helpers for full focus/apodization data and line-bounded geometry validation.
- CUDA backend for general spatial impulse/scatterer simulation.
- Broader golden validation against Field II examples.
