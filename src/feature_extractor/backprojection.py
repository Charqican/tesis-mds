from src.feature_extractor.config import FeatureConfig
import torch


def backproject(
    mapping: torch.Tensor, point_cloud: torch.Tensor, pixel_feature: torch.Tensor
) -> torch.Tensor:
    return torch.zeros()


def interpolate_feature_map(
    features: torch.Tensor, config: FeatureConfig
) -> torch.Tensor:
    return torch.zeros()


def aggregate_features(
    model_outputs: torch.Tensor,
    mappings: torch.Tensor,
    point_cloud: torch.Tensor,
    config: FeatureConfig,
) -> torch.Tensor:
    return torch.zeros()
