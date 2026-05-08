# openfield

`openfield` is an early-stage Python package for ultrasound field simulation.

The package aims to cover the scientific scope of Field II with a Pythonic,
explicit object model and GPU acceleration. Field II is used as a reference for
feature coverage and validation, not as an API style.

Initial aperture geometry support includes:

- flat, focused, multirow, convex, convex-focused, and sparse 2D arrays
- flat and concave circular pistons
- arbitrary rectangle, triangle, and line-bounded apertures

3D aperture visualization is available through the optional `vedo` extra:

```bash
pip install "openfield[viz]"
```
