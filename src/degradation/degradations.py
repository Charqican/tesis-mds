import torch
import random
from typing import Protocol  # Adviced to reduce coupling


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
        return separate_point_cloud(points, center, self.k)


class SymmetryPlaneDegradation:
    """
    Strategy: viewpoint on the plane of symmetry.
    """

    def __init__(
        self, k: float = 0.15, angle_deviation_deg: float = 90.0, radius: float = 1.0
    ):
        self.k = k
        self.angle_deviation_def = angle_deviation_deg
        self.radius = radius

    def __call__(
        self, points: torch.Tensor, symmetry_plane: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:

        center = sample_viewpoint_relative_to_plane(
            points,
            plane_normal=symmetry_plane,
            angle_deviation_deg=self.angle_deviation_def,
            radius=self.radius,
        )
        return separate_point_cloud(points, center, k)
