import numpy as np
from scipy.spatial import KDTree


def transform_points(pts: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Aplica pose model->camera: X_cam = R @ X + t."""
    return pts @ R.T + t.reshape(1, 3)


def backproject_depth(
    depth_raw: np.ndarray,
    K: np.ndarray,
    depth_scale: float,
    stride: int = 1,
) -> np.ndarray:
    """
    Retroproyecta un mapa de profundidad 16-bit a nube de puntos 3D (mm).
    Píxeles con valor 0 (sin lectura) se descartan.
    """
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
    """
    Aplica máscara visible al mapa de profundidad y retroproyecta.
    stride=1 para máxima resolución del objeto.
    """
    d_obj = depth.copy()
    d_obj[~mask] = 0
    return backproject_depth(d_obj, K, depth_scale, stride=1)
