from typing import cast

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes, sample_farthest_points
import torch
# Collection of geometric function used by other modules.


def transform_points(pts: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Applies pose model->camera: X_cam = R @ X + t."""
    return pts @ R.T + t.reshape(1, 3)


def to_object_frame(pts: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Inverse of transform_points, camera->model: X = R^T @ (X_cam - t)."""
    return (pts - t.reshape(1, 3)) @ R


### Point segmentation


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


def isolate_object_points(
    depth: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    depth_scale: float,
) -> np.ndarray:
    d_obj = depth.copy()  # may be unnecesary
    d_obj[~mask] = 0
    return backproject_depth(d_obj, K, depth_scale, stride=1)


### Feature propagation
# Same as compute_diameter in external/dgedi/utils.py: max distance between two points
def point_cloud_diameter(points: np.ndarray) -> float:
    return float(pdist(points).max())


def nn_residuals(query: np.ndarray, target: np.ndarray) -> np.ndarray:
    d, _ = KDTree(target).query(query, k=1)
    return d


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


### Metrics


def regression_recall(
    pred: np.ndarray, target: np.ndarray, field_range: np.ndarray | float, pct: float
) -> np.ndarray:
    """
    Fraction of points of each instance with |pred - target| < pct * field_range.
    pred, target: (n_instances, n_points) in mm. field_range: max of the canonical
    symmetry field of the object (half extent of the CAD along the plane normal) in mm,
    one per instance (n_instances,) or a scalar. Returns (n_instances,).
    """
    eps = pct * np.asarray(field_range, dtype=np.float64).reshape(-1, 1)
    return (np.abs(pred - target) < eps).mean(axis=1)


### Symmetry planes from the BOP discrete symmetries (rotations, det R = +1)


def rotation_axis_and_angle(R: np.ndarray) -> tuple[np.ndarray, float]:
    """Axis (eigenvector with eigenvalue 1, R a = a) and angle in degrees of a rotation."""
    eigenvalues, eigenvectors = np.linalg.eig(R)
    axis = eigenvectors[:, np.argmin(np.abs(eigenvalues - 1))].real
    angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    return axis / np.linalg.norm(axis), float(angle)


def candidate_plane_normals(
    symmetries: list[np.ndarray], parallel_cos: float = 0.99
) -> list[np.ndarray]:
    """
    Normals of the candidate mirror planes of an object: the axes of its (4, 4) BOP
    symmetries, then the CAD axes x, y, z, dropping near-parallel duplicates. The order is
    fixed, so a candidate index is reproducible. The planes pass through the CAD origin
    (BOP symmetry translations are < 1 mm). Whether a candidate is a real mirror of the CAD
    is checked in notebooks/per_object_models.py.
    """
    rotation_axes = [rotation_axis_and_angle(T[:3, :3])[0] for T in symmetries]
    normals = []
    for normal in [*rotation_axes, *np.eye(3)]:
        if all(abs(normal @ kept) < parallel_cos for kept in normals):
            normals.append(normal)
    return normals


### Symmetry available in a partial cloud
# labels of symmetric_correspondence
BAND, MATCHED, UNMATCHED = 0, 1, 2


def median_spacing(points: np.ndarray) -> float:
    """Median distance from each point to its nearest neighbour (FPS keeps it uniform)."""
    d, _ = KDTree(points).query(points, k=2)
    return float(np.median(d[:, 1]))


def symmetric_correspondence(
    points_obj: np.ndarray,
    normal: np.ndarray,
    plane_point: np.ndarray,
    band: float,
    tol: float,
) -> np.ndarray:
    """
    Label of each point of a pT in the object frame (mm):
      BAND      |distance to the plane| < band, close to the plane, little information
      MATCHED   its reflection on the plane has a pT point closer than tol
      UNMATCHED otherwise
    band comes from the CAD (pct * field_range) and tol from the pT (k * median_spacing).
    Keep band >= tol / 2, below that a point matches its own neighbourhood.
    """
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    h = (points_obj - np.asarray(plane_point, dtype=np.float64)) @ normal
    reflected = points_obj - 2 * h[:, None] * normal
    d, _ = KDTree(points_obj).query(reflected, k=1)

    labels = np.where(d < tol, MATCHED, UNMATCHED)
    labels[np.abs(h) < band] = BAND
    return labels


### subsampling
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


### Outlier removal


def remove_outlier_points(pcs: torch.Tensor):
    pass


### Angle between views
# Geodesic distance in SO(3): theta = arccos((tr(R_a R_b^T) - 1) / 2) [Huynh 2009, Hartley et al. 2013].
# With unit quaternions it is theta = 2 arccos(|q_a . q_b|), and the chordal distance |q_a - q_b| grows
# with theta, so a KDTree on the quaternions (q and -q, both represent R) finds the nearest view.


def _quaternions(rotations: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(rotations).as_quat()


def _chord_to_deg(chord: np.ndarray) -> np.ndarray:
    return np.degrees(2 * np.arccos(np.clip(1 - chord**2 / 2, -1, 1)))


def _deg_to_chord(angle_deg: float) -> float:
    return float(np.sqrt(2 - 2 * np.cos(np.radians(angle_deg) / 2)))


def rotation_angle_deg(rotations_a: np.ndarray, rotations_b: np.ndarray) -> np.ndarray:
    """Angle in degrees of R_a R_b^T for every pair (len(a) x len(b)). For small sets."""
    cos_angle = (np.einsum("aij,bij->ab", rotations_a, rotations_b) - 1) / 2
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))


def nearest_view_angle(query: np.ndarray, reference: np.ndarray, exclude_self: bool = False) -> np.ndarray:
    """Angle (deg) from each query rotation to its closest reference rotation, (n_query,).
    exclude_self: query is reference, skip each view itself."""
    q_ref = _quaternions(reference)
    tree = KDTree(np.concatenate([q_ref, -q_ref]))
    chord, _ = tree.query(_quaternions(query), k=2 if exclude_self else 1)
    return _chord_to_deg(chord[:, 1] if exclude_self else chord)


def distinct_views(rotations: np.ndarray, min_angle_deg: float) -> int:
    """Greedy count of views at least min_angle_deg apart from each other, in the given order."""
    quats = _quaternions(rotations)
    tree = KDTree(np.concatenate([quats, -quats]))
    n = len(quats)
    free = np.ones(n, dtype=bool)
    n_views = 0
    for i in range(n):
        if free[i]:
            n_views += 1
            close = np.asarray(tree.query_ball_point(quats[i], _deg_to_chord(min_angle_deg)))
            free[close % n] = False
    return n_views
