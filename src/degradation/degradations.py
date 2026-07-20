import torch
import random
from typing import Protocol  # reduce coupling
from degradation.partial import (
    separate_point_cloud,
    sample_random_viewpoint,
    sample_equator_point,
    sample_meridian_point,
    sample_transverse_meridian_point,
)


class DegradationStrategy(Protocol):
    def __call__(
        self, points: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        "Interface for LSP type check. Returns (partial_points, mask)"
        ...


class RandomViewpointDegradation:
    """
    This strategy uses a random point on the sphere as a viewpoint.
    From this viewpoint every point is sorted by distance, removing the k% farthest points
    """

    def __init__(self, k: float = 0.15, radius: float = 1.0):
        self.k = k
        self.radius = radius

    def __call__(
        self, points: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        center = sample_random_viewpoint(points, radius=self.radius)
        self.view_point = center
        return separate_point_cloud(points, center, self.k)


class EquatorCurveDegradation:
    """Barrido de ángulo (posición en el ecuador) con k% de puntos faltantes."""

    def __call__(
        self,
        points: torch.Tensor,
        symmetry_plane: tuple[torch.Tensor, torch.Tensor],
        angle: float,
        k: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        center = sample_equator_point(symmetry_plane, angle)
        self.view_point = center
        return separate_point_cloud(points, center, k)


class MeridianCurveDegradation:
    """Barrido de latitud (0=ecuador, ±pi/2=polos) + k%."""

    def __call__(
        self,
        points: torch.Tensor,
        symmetry_plane: tuple[torch.Tensor, torch.Tensor],
        angle: float,
        k: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        center = sample_meridian_point(symmetry_plane, angle)
        self.view_point = center
        return separate_point_cloud(points, center, k)


class TransverseMeridianCurveDegradation:
    """
    Barrido de latitud (0=ecuador, ±pi/2=polos) sobre el meridiano
    transversal: tambien perpendicular al plano de simetria, pero
    ademas perpendicular al meridiano principal.
    """

    def __call__(
        self,
        points: torch.Tensor,
        symmetry_plane: list[torch.Tensor, torch.Tensor],
        angle: float,
        k: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        center = sample_transverse_meridian_point(symmetry_plane, angle)
        self.view_point = center
        return separate_point_cloud(points, center, k)
