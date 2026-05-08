from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _vector3(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector")
    array = array.copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class SubElement:
    """Small surface element used for numerical integration."""

    center: np.ndarray
    normal: np.ndarray
    area: float
    physical_index: int
    subelement_index: int
    vertices: np.ndarray | None = None

    def __post_init__(self) -> None:
        center = _vector3(self.center, "center")
        normal = _vector3(self.normal, "normal")
        norm = float(np.linalg.norm(normal))
        if norm == 0.0:
            raise ValueError("normal must be non-zero")
        normal = (normal / norm).copy()
        normal.setflags(write=False)

        if self.area <= 0:
            raise ValueError("area must be positive")
        if self.physical_index < 0:
            raise ValueError("physical_index must be non-negative")
        if self.subelement_index < 0:
            raise ValueError("subelement_index must be non-negative")
        vertices = None
        if self.vertices is not None:
            vertices = np.asarray(self.vertices, dtype=np.float64)
            if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] < 3:
                raise ValueError("vertices must have shape (n, 3) with n >= 3")
            vertices = vertices.copy()
            vertices.setflags(write=False)

        object.__setattr__(self, "center", center)
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "vertices", vertices)
