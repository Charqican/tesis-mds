from feature_extractor.backprojection import aggregate_features
from feature_extractor.config import FeatureConfig
from feature_extractor.rendering import render_point_cloud
from feature_extractor.sampling import sample_fibonacci_views
from logger import pipeline_logger
import numpy as np
import torch


def extract_features(
    point_cloud: torch.Tensor,  # (N, 3)
    model: torch.nn.Module,  # DINOv2 wrapper
    config: FeatureConfig = FeatureConfig(),
) -> torch.Tensor:  # (N, emb_dim)
    """
    Extrae features DINOv2 por punto usando Fibonacci sampling y backprojection.

    Args:
        point_cloud: nube de puntos (N, 3), puede estar en CPU o GPU
        model: modelo DINOv2 que recibe (V, H, W, C) y retorna (V, num_patches, emb_dim)
        config: configuracion del pipeline, usa defaults si no se especifica

    Returns:
        features por punto (N, emb_dim)
    """
    device = point_cloud.device
    model = model.to(device)
    pipeline_logger.info(
        f"Starting feature extraction | points={len(point_cloud)} device={device}"
    )
    # Fibonacci sampling views & camera
    R, T = sample_fibonacci_views(point_cloud, config)

    # subsampling, default : 1000
    if config.max_points is not None:
        point_cloud = _subsample(point_cloud, config.max_points)
        pipeline_logger.info(f"Subsampled to {len(point_cloud)} points")

    # obtain pixel-point mappings for every rendered view
    rendered_images, mappings = point_cloud, R, T, config

    # DINOv2 features
    with torch.no_grad():
        model_outputs = model(rendered_images)

    # interpolate features to full resolution & backproject features to 3D space
    features = aggregate_features(model_outputs, mappings, point_cloud, config)
    pipeline_logger.info(f"Done | features={features.shape}")

    return features


def _subsample(point_cloud: torch.Tensor, max_points: int) -> torch.Tensor:
    """
    Subsamplea aleatoriamente la nube a max_points puntos.
    Seed fija para reproducibilidad.
    """
    if len(point_cloud) <= max_points:
        return point_cloud

    np.random.seed(0)
    indices = torch.from_numpy(
        np.random.choice(len(point_cloud), max_points, replace=False)
    )
    return point_cloud[indices]
