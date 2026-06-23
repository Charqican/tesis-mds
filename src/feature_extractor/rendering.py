# Original code from https://github.com/Spulp/EnhancedBackProjection/tree/main

import torch
import torch.nn
import numpy as np
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    NormWeightedCompositor,
)
from pytorch3d.renderer.points.rasterizer import PointFragments
from feature_extractor.config import FeatureConfig
from logger import rendering_logger

# rot_aug=(0,1,2,3,4) → (0) VS (0,180) VS (0,90,270) VS (0,90,180,270) VS (0,0,0,0)
_ROTATION_MAPS = {
    0: [0],
    1: [0, 2],
    2: [0, 1, 3],
    3: [0, 1, 2, 3],
    4: [0, 0, 0, 0],
}


def _rotate_and_interleave_images(
    images: torch.Tensor, rotation_indices: list
) -> torch.Tensor:
    """
    Args:
        images: (V, H, W, 3)
    Returns:
        (len(rotation_indices)*V, H, W, 3)
    """
    all_rotations = [
        images,
        torch.rot90(images, k=1, dims=(1, 2)),  # 90
        torch.rot90(images, k=2, dims=(1, 2)),  # 180
        torch.rot90(images, k=3, dims=(1, 2)),  # 270
    ]
    selected = [all_rotations[i] for i in rotation_indices]
    rotations = torch.stack(selected, dim=0)  # (R, V, H, W, 3)
    rotations = rotations.permute(1, 0, 2, 3, 4)  # (V, R, H, W, 3)
    return rotations.reshape(
        len(rotation_indices) * images.shape[0], *images.shape[1:]
    ).contiguous()


def _rotate_and_interleave_mappings(
    tensor: torch.Tensor, rotation_indices: list
) -> torch.Tensor:
    """
    Args:
        tensor: (V, H, W)
    Returns:
        (len(rotation_indices)*V, H, W)
    """
    all_rotations = [
        tensor,
        torch.rot90(tensor, k=1, dims=(1, 2)),  # 90
        torch.rot90(tensor, k=2, dims=(1, 2)),  # 180
        torch.rot90(tensor, k=3, dims=(1, 2)),  # 270
    ]
    selected = [all_rotations[i] for i in rotation_indices]
    rotations = torch.stack(selected, dim=0)  # (R, V, H, W)
    rotations = rotations.permute(1, 0, 2, 3)  # (V, R, H, W)
    return rotations.reshape(len(rotation_indices) * tensor.shape[0], *tensor.shape[1:])


def _render(
    pcd_stacked: Pointclouds,
    rasterizer: PointsRasterizer,
    compositor: NormWeightedCompositor,
) -> tuple[torch.Tensor, PointFragments]:

    fragments: PointFragments = rasterizer(pcd_stacked)
    r = rasterizer.raster_settings.radius
    dists2 = fragments.dists.permute(0, 3, 1, 2)
    weights = 1 - dists2 / (r * r)
    images = compositor(
        fragments.idx.long().permute(0, 3, 1, 2),
        weights,
        pcd_stacked.features_packed().permute(1, 0),
    )
    return images.permute(0, 2, 3, 1), fragments


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


def render_point_cloud(
    point_cloud: torch.Tensor,  # (N, 3)
    R: torch.Tensor,  # (V, 3, 3)
    T: torch.Tensor,  # (V, 3)
    config: FeatureConfig,
) -> tuple[torch.Tensor, torch.Tensor]:  # (rendered_images, mappings)
    """
    Renderiza la nube de puntos desde cada vista Fibonacci y retorna
    las imagenes RGB y los mappings pixel→punto para backprojection.

    Returns:
        rendered_images: (V*rot_aug, H, W, 3)
        mappings:        (V*rot_aug, H, W)  — indice de punto por pixel, -1 si vacio
    """
    device = point_cloud.device

    # normalizar nube: centrar en origen y escalar a esfera unitaria
    verts = point_cloud - point_cloud.mean(dim=0)
    verts = verts / verts.norm(dim=-1).max()
    rgb = point_cloud / 255

    # crear objeto Pointclouds y replicar para batch de vistas
    pcd = Pointclouds(points=[verts], features=[rgb])
    pcd_stacked = pcd.extend(len(R))

    # instanciar camaras perspectiva con las matrices de Fibonacci
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=0.01)

    # configurar rasterizador
    raster_settings = PointsRasterizationSettings(
        image_size=(config.resolution, config.resolution),
        radius=config.point_size,
        points_per_pixel=config.points_per_pixel,
        bin_size=0,
    )
    rendered_images, fragments = _render(
        pcd_stacked,  # Pointclouds
        PointsRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
        ),  # Rasterized
        NormWeightedCompositor(background_color=(1.0, 1.0, 1.0)),  # Compositor
    )
    rendering_logger.debug(
        f"Rendered | images={rendered_images.shape} fragments.idx={fragments.idx.shape}"
    )
    # corregir indices globales → locales en fragments.idx
    # pytorch3d concatena las nubes en batch, por lo que los indices
    # de la vista i apuntan a i*N + punto_local, hay que restar i*N
    num_points = len(verts)
    for i in range(len(R)):
        mask = fragments.idx[i] != -1
        fragments.idx[i, mask] -= i * num_points

    # aplicar rot_aug: rotar y duplicar imagenes y mappings
    rotation_indices = _ROTATION_MAPS.get(config.rot_aug, [0])
    rendered_images = _rotate_and_interleave_images(
        rendered_images[..., :3], rotation_indices
    )
    mappings = _rotate_and_interleave_mappings(fragments.idx[..., 0], rotation_indices)
    rendering_logger.info(
        f"Done | rendered_images={rendered_images.shape} mappings={mappings.shape}"
    )
    return rendered_images, mappings
