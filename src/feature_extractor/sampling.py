import torch
from src.feature_extractor.config import FeatureConfig


def get_camera_distance(point_cloud: torch.Tensor, fov: float) -> float:

    return 0.1


def sample_fibonacci_views(
    point_cloud: torch.Tensor, config: FeatureConfig
) -> tuple[torch.Tensor, torch.Tensor]:

    return (torch.zeros(), torch.zeros())
