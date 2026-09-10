from typing import cast

import numpy as np
from scipy.spatial import KDTree
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes, sample_farthest_points
import torch
# Collection of geometric function used by other modules.


def transform_points(pts: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Aplica pose model->camera: X_cam = R @ X + t."""
    return pts @ R.T + t.reshape(1, 3)


def backproject_depth(
    depth_raw: np.ndarray,
    K: np.ndarray,
    depth_scale: float,
    stride: int = 1,
) -> np.ndarray:
    H, W = depth_raw.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    vs, us = np.mgrid[0:H:stride, 0:W:stride]
    z = depth_raw[vs, us].astype(np.float64) * depth_scale
    valid = z > 0
    us, vs, z = us[valid], vs[valid], z[valid]

    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def nn_residuals(query: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Para cada punto en query, distancia al vecino más cercano en target.
    """
    d, _ = KDTree(target).query(query, k=1)
    return d


def isolate_object_points(
    depth: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    depth_scale: float,
) -> np.ndarray:
    d_obj = depth.copy()
    d_obj[~mask] = 0
    return backproject_depth(d_obj, K, depth_scale, stride=1)


def knn_propagate(
    model_points: np.ndarray,
    point_features: np.ndarray,
    gt_points: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    model_points_cam = transform_points(model_points, R, t)
    tree = KDTree(model_points_cam)
    # ids : array of indexes corresponding with closest model point
    _, ids = tree.query(gt_points, k=1)
    # reorders point_features to correctly assign each feature to each gt point
    propagated_features = point_features[ids]
    return propagated_features


def propagate_symmetry_to_target(
    mesh_points: np.ndarray,
    symmetry_scalar: np.ndarray,
    target_points: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    symmetry_scalar = np.asarray(symmetry_scalar).reshape(-1)  # enforces 1D
    propagated = knn_propagate(mesh_points, symmetry_scalar, target_points, R, t)
    return propagated.reshape(-1)  # just in case!


def subsample_points(
    points: np.ndarray, max_points: int, seed: int = 123
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def sample_mesh_fps(
    mesh: Meshes,
    num_points_dense: int,
    num_points_fps: int,
) -> torch.Tensor:

    dense_points = cast(torch.Tensor, sample_points_from_meshes(mesh, num_points_dense))
    fps_points, _ = sample_farthest_points(dense_points, K=num_points_fps)

    return fps_points.squeeze(0)
