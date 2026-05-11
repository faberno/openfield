# openfield

`openfield` is an early-stage Python package for ultrasound field simulation.

The package aims to cover the scientific scope of Field II with a Pythonic,
explicit object model and GPU acceleration. Field II is used as a reference for
feature coverage and compatibility checks, not as the numerical oracle or API
style.

The spatial impulse response kernel is built around exact flat-facet
integration. Rectangular, triangular, and line-bounded apertures are integrated
as polygon facets; non-planar polygons are triangulated. This favors a simple,
convergent model over reproducing Field II's legacy far-field rectangle
approximation.

Curved and circular apertures also provide opt-in adaptive tessellations. The
legacy Field II-style layouts remain available for compatibility, while
`tessellation="adaptive"` or the `.adaptive(...)` constructors use
boundary-fitted or curvature-refined facets.

Initial aperture geometry support includes:

- flat, focused, multirow, convex, convex-focused, and sparse 2D arrays
- flat and concave circular pistons
- arbitrary rectangle, triangle, and line-bounded apertures

3D aperture visualization is available through the optional `vedo` extra:

```bash
pip install "openfield[viz]"
```
