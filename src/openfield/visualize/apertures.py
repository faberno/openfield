from __future__ import annotations

import importlib

import numpy as np


def aperture_mesh_data(aperture, *, element_scale: float = 1.0) -> tuple[np.ndarray, list[list[int]]]:
    """Return polygon mesh data for rendering a tessellated aperture."""

    if element_scale <= 0:
        raise ValueError("element_scale must be positive")

    vertices: list[np.ndarray] = []
    faces: list[list[int]] = []

    for element in aperture.elements:
        if element.vertices is None:
            element_vertices = _fallback_element_vertices(element)
        else:
            element_vertices = element.vertices

        if element_scale != 1.0:
            element_vertices = element.center + (element_vertices - element.center) * element_scale

        face = []
        for vertex in element_vertices:
            face.append(len(vertices))
            vertices.append(np.asarray(vertex, dtype=np.float64))
        faces.append(face)

    return np.asarray(vertices, dtype=np.float64), faces


def show_aperture(
    aperture,
    *,
    element_scale: float = 0.95,
    show_normals: bool = True,
    normal_scale: float | None = None,
    axes: bool | int = True,
    title: str | None = None,
    color: str = "lightblue",
    normal_color: str = "black",
    interactive: bool = True,
    **show_kwargs,
):
    """Render an aperture with vedo and return the vedo plotter/window object."""

    vedo = _import_vedo()
    vertices, faces = aperture_mesh_data(aperture, element_scale=element_scale)

    mesh = vedo.Mesh([vertices, faces]).c(color)
    if hasattr(mesh, "lighting"):
        mesh = mesh.lighting("plastic")

    actors = [mesh]
    if show_normals:
        centers = aperture.centers
        normals = aperture.normals
        length = _normal_length(aperture, normal_scale)
        actors.append(vedo.Arrows(centers, centers + normals * length, c=normal_color))

    return vedo.show(
        *actors,
        axes=axes,
        title=title or aperture.name,
        interactive=interactive,
        **show_kwargs,
    )


def _import_vedo():
    try:
        return importlib.import_module("vedo")
    except ModuleNotFoundError as exc:
        if exc.name != "vedo":
            raise
        raise ImportError(
            "Aperture visualization requires vedo. Install it with "
            "`pip install openfield[viz]` or `pip install vedo`."
        ) from exc


def _fallback_element_vertices(element) -> np.ndarray:
    tangent_x, tangent_y = _basis_for_normal(element.normal)
    side = np.sqrt(element.area)
    center = element.center
    return np.array(
        [
            center - side / 2 * tangent_x - side / 2 * tangent_y,
            center + side / 2 * tangent_x - side / 2 * tangent_y,
            center + side / 2 * tangent_x + side / 2 * tangent_y,
            center - side / 2 * tangent_x + side / 2 * tangent_y,
        ],
        dtype=np.float64,
    )


def _basis_for_normal(normal) -> tuple[np.ndarray, np.ndarray]:
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(reference, normal))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    tangent_x = reference - np.dot(reference, normal) * normal
    tangent_x = tangent_x / np.linalg.norm(tangent_x)
    tangent_y = np.cross(normal, tangent_x)
    return tangent_x, tangent_y


def _normal_length(aperture, normal_scale: float | None) -> float:
    if normal_scale is not None:
        if normal_scale <= 0:
            raise ValueError("normal_scale must be positive")
        return float(normal_scale)
    return float(np.sqrt(np.median(aperture.areas)))
