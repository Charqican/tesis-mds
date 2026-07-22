from dataclasses import dataclass

import numpy as np
import trimesh  # may change to torch

from .config import LMOConfig
from .data_loader import LMOLoader, InstanceData
from .geometry_utils import backproject_depth, transform_points, isolate_object_points


# dataclass que contiene scene cloud, object cloud y posed mesh.
# Se favorece sobre diccionarios para un workflow idieomatico + mejor LSP compat
@dataclass(frozen=True)
class RegistrationResult:
    scene_pts: np.ndarray  # (N, 3) nube completa de la escena
    object_pts: (
        np.ndarray | None
    )  # (M, 3) puntos del objeto visibles, o None sin máscara
    posed_mesh_pts: np.ndarray  # (S, 3) muestras de superficie del mesh posado
    pose_R: np.ndarray  # (3, 3)
    pose_t: np.ndarray  # (3,)
    obj_id: int
    visib_fract: float | None


def load_mesh_samples(
    models_dir: str | object, obj_id: int, n_samples: int
) -> np.ndarray:
    """Carga malla y muestrea n_samples puntos de su superficie."""
    path = f"{models_dir}/obj_{obj_id:06d}.ply"
    mesh = trimesh.load(path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    # Muestreo area-weighted
    tri = vertices[faces]  # (F, 3, 3)
    a = tri[:, 1] - tri[:, 0]
    b = tri[:, 2] - tri[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(a, b), axis=1)
    probs = areas / areas.sum()

    rng = np.random.default_rng(0)
    idx = rng.choice(len(faces), size=n_samples, p=probs)
    u = rng.random((n_samples, 1))
    w = rng.random((n_samples, 1))
    over = (u + w) > 1
    u[over] = 1 - u[over]
    w[over] = 1 - w[over]

    return tri[idx, 0] + u * a[idx] + w * b[idx]


def register_instance(
    loader: LMOLoader,
    config: LMOConfig,
    scene_id: int,
    img_id: int,
    inst_idx: int,
) -> RegistrationResult:
    """
    Registra una instancia de objeto en una imagen del dataset.

    1. Carga camara, depth, instancia
    2. Construye nube de escena (submuestreada por load_mesh_samples)
    3. Aísla puntos del objeto con máscara visible
    4. Muestrea mesh y aplica pose GT
    """
    # Metadatos
    K, depth_scale = loader.load_camera(scene_id, img_id)
    instances = loader.load_instances(scene_id, img_id)
    instance: InstanceData = instances[inst_idx]

    # Nube de escena
    depth = loader.load_depth(scene_id, img_id)
    scene_pts = backproject_depth(depth, K, depth_scale, stride=config.depth_stride)

    # Puntos del objeto (con máscara visible)
    mask = loader.load_mask_visib(scene_id, img_id, inst_idx)
    object_pts = None
    if mask is not None:
        object_pts = isolate_object_points(depth, mask, K, depth_scale)

    # Mesh posado
    mesh_samples = load_mesh_samples(
        config.paths.models_dir, instance.obj_id, config.mesh_samples
    )
    posed_mesh_pts = transform_points(mesh_samples, instance.R, instance.t)

    return RegistrationResult(
        scene_pts=scene_pts,
        object_pts=object_pts,
        posed_mesh_pts=posed_mesh_pts,
        pose_R=instance.R,
        pose_t=instance.t,
        obj_id=instance.obj_id,
        visib_fract=instance.visible_fract,
    )
