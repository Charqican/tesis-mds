# Original code from https://github.com/Spulp/EnhancedBackProjection/tree/main
from feature_extractor.backprojection import aggregate_features, sample_feature_mesh
from feature_extractor.config import FeatureConfig
from feature_extractor.rendering import render_point_cloud
from feature_extractor.sampling import (
    sample_fibonacci_view_positions,
    sample_fibonacci_views,
)
from feature_extractor.mesh_backprojection import (
    features_backprojection,
    _normalize_mesh_to_unit_sphere,
)
from logger import pipeline_logger
import torch
from pytorch3d.structures import Meshes
import numpy as np
import torch


def extract_features_pc(
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
    rendered_images, mappings = render_point_cloud(point_cloud, R, T, config)
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


def extract_features_fm(
    mesh: Meshes,
    model: torch.nn.Module,  # DINOv2 wrapper
    num_samples: int,
    config: FeatureConfig = FeatureConfig(),
) -> tuple[
    torch.Tensor, torch.Tensor
]:  # points (num_samples, 3), features (num_samples, emb_dim)
    """
    Extrae features DINOv2 via Feature-Mesh Sampling (FM).

    Dos etapas:
      1. RM: features por vertice del mesh original (features_backprojection).
      2. FM: propagacion de esas features a puntos muestreados sobre las
         caras del mesh, via interpolacion baricentrica (sample_feature_mesh).

    A diferencia de extract_features (PC), no hay subsampling: FM necesita
    la topologia completa del mesh para poder samplear e interpolar sobre
    las caras, asi que no se puede subsamplear vertices como con max_points
    en el pipeline de nube de puntos.

    Args:
        mesh: mesh de pytorch3d (un unico objeto)
        model: modelo DINOv2 que recibe (V, H, W, C) y retorna (V, num_patches, emb_dim)
        num_samples: numero de puntos FM a muestrear sobre la superficie
        config: configuracion del pipeline, usa defaults si no se especifica

    Returns:
        (points, features): puntos FM y sus features, ambos alineados por indice
    """

    device = mesh.device
    model = model.to(device)

    pipeline_logger.info(
        f"Starting FM feature extraction | vertices={mesh.verts_packed().shape[0]} device={device}"
    )
    # NORMALIZAR PRIMERO
    mesh = _normalize_mesh_to_unit_sphere(mesh)

    views = sample_fibonacci_view_positions(mesh.verts_packed(), config)

    vertex_features = features_backprojection(model, mesh, views, config, device=device)
    pipeline_logger.info(f"RM done | vertex_features={vertex_features.shape}")

    points, features = sample_feature_mesh(mesh, vertex_features, num_samples)
    pipeline_logger.info(f"Done | points={points.shape} features={features.shape}")

    return points, features
