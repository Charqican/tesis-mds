from src.feature_extractor.config import FeatureConfig
import torch


def render_point_cloud(
    point_cloud: torch.Tensor, R: torch.Tensor, T: torch.Tensor, config: FeatureConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros(), torch.zeros()
