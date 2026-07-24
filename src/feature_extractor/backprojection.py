# Original code from https://github.com/Spulp/EnhancedBackProjection/tree/main
from typing import cast
from feature_extractor.config import FeatureConfig
from logger import backprojection_logger
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import knn
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.renderer import TexturesVertex


def _backproject(
    mapping: torch.Tensor,  # (H, W)
    point_cloud: torch.Tensor,  # (N, 3)
    pixel_features: torch.Tensor,  # (H, W, emb_dim)
) -> torch.Tensor:  # (N, emb_dim)
    """
    Back-project features to points in the point cloud using the given mapping.

    :param mapping: tensor with shape (CANVAS_HEIGHT, CANVAS_WIDTH)
    :param point_cloud: tensor, points in the point cloud, with shape (#points, 3)
    :param pixel_features: features extracted with a backbone model. Shape is
        (CANVAS_HEIGHT, CANVAS_WIDTH, feature_dimensionality)

    :return: a new tensor of shape (#points, feature_dimensionality) that associates
        to each point in the point cloud a feature vector. Feature vector is all 0
        if no feature is being associated to a point.
        It can be indexed as `point_cloud`, so the i-th feature vector is associated
        to the i-th point in `point_cloud`.
    """
    assert mapping.shape == pixel_features.shape[:2], (
        f"mapping {mapping.shape} y pixel_features {pixel_features.shape[:2]} incompatibles"
    )

    device = point_cloud.device
    mapping = mapping.to(device)
    pixel_features = pixel_features.to(device)

    features = torch.zeros(
        (len(point_cloud), pixel_features.shape[-1]), dtype=torch.float, device=device
    )
    yx_coords = (mapping != -1).nonzero()

    # mapping[pixel] → indice del punto → features[punto]
    # si un punto ocupa multiples pixeles vecinos se queda con el ultimo
    # (pixeles vecinos tienen features muy similares, no es un problema practico)
    features[mapping[mapping != -1]] = pixel_features[yx_coords[:, 0], yx_coords[:, 1]]

    return features


def _interpolate_feature_map(
    features: torch.Tensor,  # (1, num_patches, emb_dim)
    config: FeatureConfig,
) -> torch.Tensor:  # (H, W, emb_dim)
    """
    Upscalea el feature map de resolucion de patches a resolucion de pixels
    usando interpolacion bicubica.
    """
    R, L, _ = features.shape
    W = H = np.sqrt(L).astype(int)

    with torch.no_grad():
        interpolated = F.interpolate(
            features.view(R, W, H, -1).permute(3, 0, 1, 2),  # (emb_dim, 1, W, H)
            size=(config.resolution, config.resolution),
            mode="bicubic",
            align_corners=False,
        )
    backprojection_logger.debug(
        f"Interpolated feature map | input={features.shape} output={interpolated.shape}"
    )
    return interpolated.permute(1, 2, 3, 0).squeeze(0)  # (H, W, emb_dim)


def _interpolate_occluded_points(
    point_cloud: torch.Tensor,  # (N, 3)
    features: torch.Tensor,  # (N, emb_dim)
    neighbors: int = 20,
) -> torch.Tensor:  # (N, emb_dim)
    """
    Interpola features de puntos ocluidos (vector cero) usando el promedio
    de sus k vecinos mas cercanos que si tienen features.
    """
    missing = torch.all(features == 0, dim=-1).nonzero().view(-1)

    if len(missing) == 0:
        return features

    knn_result = knn(point_cloud, point_cloud, neighbors + 1)
    knn_indices = knn_result[1].view(len(point_cloud), neighbors + 1)

    neighbor_features = features[knn_indices[missing].view(-1)].view(
        len(missing), neighbors + 1, -1
    )
    # marcar vecinos sin features como NaN para excluirlos del promedio
    neighbor_features[torch.all(neighbor_features == 0, dim=-1)] = float("nan")
    features[missing] = neighbor_features.nanmean(dim=1)

    backprojection_logger.debug(
        f"Interpolated occluded points | missing={len(missing)}/{len(point_cloud)}"
    )

    return torch.nan_to_num(features)


def aggregate_features(
    model_outputs: torch.Tensor,  # (V*rot, num_patches, emb_dim)
    mappings: torch.Tensor,  # (V*rot, H, W)
    point_cloud: torch.Tensor,  # (N, 3)
    config: FeatureConfig,
) -> torch.Tensor:  # (N, emb_dim)
    """
    Interpola features de patches a pixels, backprojecta a espacio 3D,
    agrega sobre todas las vistas y resuelve puntos ocluidos con kNN.
    """
    device = point_cloud.device
    num_views = model_outputs.shape[0]
    emb_dim = model_outputs.shape[-1]
    num_points = len(point_cloud)

    features_acc = torch.zeros((num_points, emb_dim), device=device, dtype=torch.double)
    count = torch.zeros(num_points, device=device)

    for i in range(0, num_views, config.batch_size):
        batch_outputs = model_outputs[
            i : i + config.batch_size
        ]  # (B, num_patches, emb_dim)
        batch_mappings = mappings[i : i + config.batch_size]  # (B, H, W)

        for j in range(len(batch_outputs)):
            # patches → pixels
            interpolated = _interpolate_feature_map(
                batch_outputs[j : j + 1], config
            )  # (H, W, emb_dim)

            # pixels → puntos 3D
            feature_pcd = _backproject(
                batch_mappings[j], point_cloud, interpolated
            )  # (N, emb_dim)

            # acumular solo puntos visibles
            visible = ~torch.all(feature_pcd == 0.0, dim=-1)
            features_acc[visible] += feature_pcd[visible]
            count[visible] += 1

            del interpolated, feature_pcd

        del batch_outputs, batch_mappings
        if device.type != "cpu":
            torch.cuda.empty_cache()

    # normalizar por numero de vistas en que fue visible cada punto
    count[count == 0] = 1
    features_acc /= count.unsqueeze(-1)

    # interpolar puntos ocluidos con kNN
    features_acc = _interpolate_occluded_points(point_cloud[:, :3], features_acc)
    backprojection_logger.info(f"Done | features={features_acc.shape}")
    return features_acc


def sample_feature_mesh(
    mesh: Meshes,
    vertex_features: torch.Tensor,  # (N, emb_dim) features ya calculadas sobre los vertices del mesh (RM)
    num_samples: int,
) -> tuple[
    torch.Tensor, torch.Tensor
]:  # points (num_samples, 3), features (num_samples, emb_dim)
    """
    Feature-Mesh Sampling (FM). Muestrea puntos uniformemente sobre la
    superficie del mesh y les asigna features por interpolacion baricentrica
    de las features ya calculadas en los vertices (RM), sin volver a
    renderizar ni back-projectar.

    A diferencia de PC (`aggregate_features`, donde cada punto muestreado
    se back-projecta independientemente vista por vista), FM reutiliza
    las features de vertice ya computadas y las propaga usando la
    topologia del mesh: para un punto en una cara con vertices
    (va, vb, vc) y coordenadas baricentricas (a, b, c), su feature es
    a*f_va + b*f_vb + c*f_vc.

    param mesh: Meshes de pytorch3d con un unico objeto
    param vertex_features: features por vertice, alineadas con
        `mesh.verts_packed()` (mismo orden, mismo N)
    param num_samples: numero de puntos a muestrear sobre la superficie

    return: (points, features) muestreados sobre la superficie del mesh
    """
    num_verts = mesh.verts_packed().shape[0]
    assert vertex_features.shape[0] == num_verts, (
        f"vertex_features tiene {vertex_features.shape[0]} filas pero el mesh "
        f"tiene {num_verts} vertices"
    )

    device = vertex_features.device
    mesh = mesh.to(device)

    # las features se cargan como "textura" por vertice para que pytorch3d
    # las interpole baricentricamente al samplear puntos sobre las caras
    mesh_with_features = mesh.clone()
    mesh_with_features.textures = TexturesVertex(
        verts_features=vertex_features.unsqueeze(0)  # (1, N, emb_dim)
    )

    # return_textures=True hace que la firma de tipo de sample_points_from_meshes
    # sea Union; con este flag en runtime siempre retorna (points, textures)
    result = sample_points_from_meshes(
        mesh_with_features, num_samples, return_textures=True
    )
    points, features = cast(tuple[torch.Tensor, torch.Tensor], result)

    backprojection_logger.info(
        f"FM sampling done | points={points.shape} features={features.shape}"
    )

    return points.squeeze(0), features.squeeze(
        0
    )  # (num_samples, 3), (num_samples, emb_dim)
