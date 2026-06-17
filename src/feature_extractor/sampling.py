import torch
import numpy as np
from pytorch3d.renderer import look_at_view_transform
from src.feature_extractor.config import FeatureConfig


def _get_camera_distance(point_cloud: torch.Tensor, fov: float) -> float:
    radius = point_cloud.norm(dim=-1).max().item()

    fov_rad = np.deg2rad(fov)
    # fov & aspect_ratio=1
    fov_h = 2 * np.arctan(np.tan(fov_rad / 2) * np.sqrt(0.5))
    fov_v = 2 * np.arctan(np.tan(fov_rad / 2) * np.sqrt(0.5))
    fov_min = min(fov_h, fov_v)

    distance = radius / np.tan(fov_min / 2)
    return distance * 1.2


def _sample_fibonacci_points(radius: float, n_point: int) -> np.ndarray:
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle
    points = []

    for i in range(n_point):
        y = 1 - (i / float(n_point - 1)) * 2  # y from 1 to -1
        radius_at_y = np.sqrt(1 - y * y)
        theta = phi * i

        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y
        points.append([x * radius, y * radius, z * radius])

    # minimun rotation to avoid nuimeric degeneration
    angle = 0.001
    axis = np.array([1.0, 0.0, 0.0])
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s, C = np.cos(angle), np.sin(angle), 1 - np.cos(angle)
    R = np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )
    points = [R @ p for p in points]

    return np.array(points)


def sample_fibonacci_views(
    point_cloud: torch.Tensor, config: FeatureConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    device = point_cloud.device

    radius = _get_camera_distance(point_cloud, config.fov)
    views = _sample_fibonacci_points(radius, config.num_views)

    R, T = look_at_view_transform(
        eye=torch.tensor(views, dtype=torch.float32), device=device
    )

    return R, T
