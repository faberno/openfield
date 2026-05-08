# Validation Debt

This file tracks Field II parity gaps that are known, intentional, and still
blocking strict validation.

## Spatial Impulse Response (`calc_h`)

Status: unresolved.

Validated so far:

- Field II-compatible output time windows are implemented for rectangular
  subelements.
- Linear-array and sparse-2D-array geometry export now passes strict comparison
  against `xdc_get(..., 'rect')`.
- A single rectangular subelement is close in scale and timing, but not yet
  within strict value tolerances.

Current gap:

- Multi-subelement pressure fields have matching broad support but different
  sample-wise distribution from Field II.
- The current CPU path uses a far-field center approximation with support
  spreading. Field II appears to use a related rectangle approximation, but the
  exact per-sample deposition rule has not been matched.

Affected validation cases:

- `single_rectangle_spatial_impulse`
- `piston_spatial_impulse`
- `linear_array_spatial_impulse`
- `linear_array_emitted_pressure`
- `linear_array_focus_timeline_spatial_impulse`
- `linear_array_delay_timeline_spatial_impulse`
- `linear_array_apodization_spatial_impulse`
- `linear_array_subelement_apodization_spatial_impulse`
- `linear_array_subelement_delay_spatial_impulse`
- Any pulse-echo or scatterer case that depends on spatial impulse responses.

Revisit before removing expected failures from:

- `calc_h`
- `calc_hp`
- `calc_hhp`
- `calc_scat`
- `calc_scat_multi`
- `calc_scat_all`

## Pulse-Echo And Scatterer Responses

Status: strict small-case Field II validation passes.

Initial validation goal:

- Field II and openfield produce matching result shapes and values for small
  transmit/receive/scatterer cases.
- Python unit tests verify convolution, scatterer weighting, channel ordering,
  and decimation behavior independently of Field II.

Remaining risk:

- Broader pulse-echo/scatterer coverage can still move if the lower-level
  `calc_h` rectangle deposition rule changes.

## Explicit Delay Timelines

Status: unresolved.

Current gap:

- The validation case for user-supplied physical-element delays currently has a
  two-sample shape mismatch against Field II.
- This may be a delay sign/reference convention issue or another edge of the
  Field II time-window rule.

Affected validation case:

- `linear_array_delay_timeline_spatial_impulse`
- `linear_array_subelement_delay_spatial_impulse`

## Soft Baffle

Status: unresolved.

Current gap:

- `openfield` has a cosine-obliquity soft-baffle approximation, but it has not
  been matched to Field II's exact soft-baffle rule.

Affected validation case:

- `linear_array_soft_baffle_spatial_impulse`

## Per-Element Waveforms

Status: missing Field II oracle.

Current gap:

- Python unit tests cover per-physical-element transmit waveforms in the
  emitted-pressure path.
- A dedicated Field II `ele_waveform` validation case has not been added yet.
