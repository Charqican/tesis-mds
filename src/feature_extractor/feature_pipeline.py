# src/feature_extractor/pipeline.py

import torch
from feature_extractor.config import FeatureConfig
from feature_extractor.sampling import sample_fibonacci_views
from feature_extractor.rendering import render_point_cloud
from feature_extractor.backprojection import aggregate_features


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

    # Fibonacci sampling vies & camera
    R, T = sample_fibonacci_views(point_cloud, config)

    # obtain pixel-point mappings for every rendered view
    rendered_images, mappings = render_point_cloud(point_cloud, R, T, config)

    # DINOv2 features
    with torch.no_grad():
        model_outputs = model(rendered_images)

    # interpolate features to full resolution & backproject features to 3D space
    features = aggregate_features(model_outputs, mappings, point_cloud, config)

    return features
