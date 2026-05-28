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

Curved and circular apertures use adaptive or boundary-fitted tessellations by
default. Field II-style layouts remain available for compatibility through
explicit options such as `tessellation="cartesian"` for pistons and
`tessellation="fieldii"` for focused or convex arrays.

```python
from openfield.apertures import ConvexFocusedArray, Piston

piston = Piston(radius=5e-3, element_size=0.5e-3)

probe = ConvexFocusedArray(
    elements=64,
    width=0.3e-3,
    height=5e-3,
    kerf=0.03e-3,
    convex_radius=25e-3,
    elevation_focus=20e-3,
)
```

Initial aperture geometry support includes:

- flat, focused, multirow, convex, convex-focused, and sparse 2D arrays
- flat and concave circular pistons
- arbitrary rectangle, triangle, and line-bounded apertures

3D aperture visualization is available through the optional `vedo` extra:

```bash
pip install "openfield[viz]"
```
