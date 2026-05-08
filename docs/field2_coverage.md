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
| `xdc_rectangles.m` | Aperture from rectangular patches | `RectangleAperture` | Partial | Clean Python vertex input exists; Field II `rect` matrix parser is not implemented. |
| `xdc_triangles.m` | Aperture from triangular patches | `TriangleAperture` | Partial | Clean Python vertex input exists; Field II `data` matrix parser is not implemented. |
| `xdc_lines.m` | Flat aperture from line-bounded half planes | `LineBoundedAperture` | Partial | Half-plane clipping exists; needs validation against Field II examples. |

## Aperture Configuration And Manipulation

| Field II file | Field II functionality | `openfield` equivalent | Status | Notes |
|---|---|---|---|---|
| `xdc_focus.m` | Fixed focus timeline from focal points | `FixedFocus`, `Aperture.focused_at()`, `Aperture.with_focus()` | Partial | Fixed focus exists. Full timeline behavior is not implemented. |
| `xdc_center_focus.m` | Reference origin for focus delay calculation | `FixedFocus(origin=...)` | Implemented | Folded into focus object rather than aperture-global state. |
| `xdc_dynamic_focus.m` | Dynamic focusing from direction angles and start time | future `DynamicFocus` | Planned | No implementation yet. |
| `xdc_focus_times.m` | User-supplied focus delay timeline | future explicit delay timeline | Planned | Needed for full Field II parity. |
| `xdc_times_focus.m` | Alias/duplicate for user-supplied focus delay timeline | future explicit delay timeline | Planned | Field II file duplicates `xdc_focus_times` behavior. |
| `xdc_apodization.m` | Physical-element apodization timeline | `Apodization`, `Aperture.with_apodization()` | Partial | Timeline container exists; not yet used by all physics routines. |
| `xdc_impulse.m` | Aperture impulse response | `Aperture.with_impulse_response()`, `Waveform`, `ToneBurst` | Partial | Storage/API exists. Pressure/scatter convolution is not implemented yet. |
| `xdc_excitation.m` | Aperture excitation pulse | `Aperture.with_excitation()`, `Waveform`, `ToneBurst` | Partial | Storage/API exists. Pressure/scatter convolution is not implemented yet. |
| `xdc_baffle.m` | Rigid or soft baffle condition | future baffle config | Planned | Should likely be `Aperture.with_baffle(...)` or aperture metadata/config. |
| `xdc_quantization.m` | Quantize focus delays | `FixedFocus(quantization=...)` | Partial | Implemented for fixed-focus delays only. |
| `xdc_convert.m` | Convert rectangular aperture description to triangles | future conversion utility | Planned | Likely `aperture.to_triangles()`. |
| `xdc_line_convert.m` | Convert rectangular aperture to line-bounded representation | future conversion utility | Planned | Lower priority unless needed for validation/parity. |
| `xdc_get.m` | Return aperture geometry/focus/apodization data | aperture properties such as `centers`, `normals`, `areas`, `physical_centers` | Partial | Basic geometry properties exist. Field II-format export does not. |
| `xdc_show.m` | Print aperture information | `Aperture.show_3d()`, future summary/repr | Partial | 3D visualization exists; textual summary is not implemented. |
| `xdc_free.m` | Free one aperture handle | none | Not applicable | No global handle registry in public API. |
| `ele_apodization.m` | Per-subelement apodization within physical elements | future subelement apodization | Planned | Not implemented. |
| `ele_delay.m` | Per-subelement delay within physical elements | future subelement delay config | Planned | Not implemented. |
| `ele_waveform.m` | Per-physical-element waveform | future element waveform config | Planned | Not implemented. |

## Field And Signal Calculations

| Field II file | Field II functionality | `openfield` equivalent | Status | Notes |
|---|---|---|---|---|
| `calc_h.m` | Spatial impulse response | `Simulation.spatial_impulse_response()` / `physics.spatial_impulse_response()` | Partial | CPU reference path exists; needs stronger Field II validation and GPU backend. |
| `calc_hp.m` | Emitted pressure field | `Simulation.emitted_pressure()` / `physics.emitted_pressure()` | Partial | CPU reference path exists with excitation and impulse-response convolution; needs validation and GPU backend. |
| `calc_hhp.m` | Pulse-echo field | future `Simulation.pulse_echo_response()` | Planned | Requires transmit/receive impulse response composition. |
| `calc_scat.m` | Received RF from scatterers | future `Simulation.scatterer_response()` | Planned | Core Field II imaging workflow; not implemented yet. |
| `calc_scat_multi.m` | Scatterer response for each receive element | future `Simulation.receive_channel_responses()` | Planned | Depends on scatterer simulation. |
| `calc_scat_all.m` | Full transmit/receive synthetic aperture raw data | future `Simulation.full_matrix_capture()` | Planned | Depends on scatterer simulation and channelized Tx/Rx loops. |

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
- `SubElement` geometry model with centers, normals, areas, physical indices, and polygon vertices.
- Basic fixed focus, apodization container, waveform container, and tone burst helper.
- CPU reference spatial impulse response shape/time-axis path.
- CPU reference emitted pressure path using excitation and impulse-response convolution.
- Vedo-based aperture visualization via `Aperture.show_3d()`.

Major missing areas:

- Pulse-echo and scatterer RF calculations.
- Dynamic focusing and explicit delay timelines.
- Per-subelement apodization/delay and per-element waveforms.
- Attenuation and baffle physics in calculations.
- Field II-format import/export helpers for rectangle, triangle, line, focus, and apodization data.
- CUDA backend for general spatial impulse/scatterer simulation.
- Golden validation against Field II examples.
