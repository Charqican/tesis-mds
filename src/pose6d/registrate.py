from dataclasses import dataclass

import torch
import numpy as np
import trimesh  # may change to torch

from pose6d.config import LMOConfig
from pose6d.data_loader import LMOLoader, InstanceData, SymmetryData
from pose6d.features import MeshFeatureExtractor
from pose6d.geometry_utils import (
    backproject_depth,
    transform_points,
    isolate_object_points,
    knn_propagate,
)
from model_wrappers import DINOWrapper
from logger import registrate_logger


# dataclass que contiene scene cloud, object cloud y posed mesh.
# Se favorece sobre diccionarios para un workflow idieomatico + mejor LSP compat
# Ademas se expone como interfaz para otros modulos
@dataclass(frozen=True)
class RegistratedData:
    scene_pts: np.ndarray  # (N, 3) nube completa de la escena
    masked_points: (
        np.ndarray | None
    )  # (M, 3) puntos del objeto visibles, o None sin mascara
    posed_mesh_pts: np.ndarray  # (S, 3) muestras de superficie del mesh posado
    pose_R: np.ndarray  # (3, 3)
    pose_t: np.ndarray  # (3,)
    obj_id: int
    visib_fract: float | None  # fraccion visible del objeto
    propagated_features: (
        np.ndarray | None
    )  # features propagadas desde el mesh a puntos visibles


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


# WARGNING: object path resolver is currently hardcoded. It should change in case of a different naming convention
# TODO: Handle errors
def register_instance(
    loader: LMOLoader,
    config: LMOConfig,
    scene_id: int,
    img_id: int,
    inst_idx: int,
    feature_mode: str = "distance",
) -> RegistratedData:
    """
    Registra una instancia de objeto en una imagen del dataset.

    1. Carga camara, depth, instancia
    2. Construye nube de escena (submuestreada por load_mesh_samples)
    3. Aísla puntos del objeto con máscara visible
    4. Muestrea mesh y aplica pose GT
    5. Obtiene features del mesh GT
    6. Porpaga features del mesh a pose GT

    """
    # Metadatos
    registrate_logger.info("Registrating Metadata")
    K, depth_scale = loader.load_camera(scene_id, img_id)
    instances = loader.load_instances(scene_id, img_id)
    instance: InstanceData = instances[inst_idx]

    # Nube de escena
    registrate_logger.info("Loading scene")
    depth = loader.load_depth(scene_id, img_id)
    scene_pts = backproject_depth(depth, K, depth_scale, stride=config.depth_stride)

    # Puntos del objeto (con mascara visible)
    registrate_logger.info("Building ocludded pointcloud")
    mask = loader.load_mask_visib(scene_id, img_id, inst_idx)
    masked_points = None
    if mask is not None:
        # Obtener puntos visibles
        masked_points = isolate_object_points(depth, mask, K, depth_scale)

    # Mesh posado
    registrate_logger.info("Sampling Mesh")
    mesh_samples = load_mesh_samples(
        config.paths.models_dir, instance.obj_id, config.mesh_samples
    )
    posed_mesh_pts = transform_points(mesh_samples, instance.R, instance.t)

    # Features extraidos de mesh
    registrate_logger.info("Extracting & propagating features")
    propagated_features = None

    if masked_points is not None and len(masked_points) > 0:
        mesh_path = config.paths.models_dir / f"obj_{instance.obj_id:06d}.ply"
        extractor = MeshFeatureExtractor(mesh_path, feature_mode, config.mesh_samples)

        # Extract Symmetry data to calculate distnace fields
        if feature_mode == "distance":  # get symmdata
            symmetry_data = loader.load_symmetry_plane(instance.obj_id)
            if symmetry_data is not None:
                mesh_points, mesh_features = extractor.extract(plane=symmetry_data)
            else:
                ...  # handle error: symmetry data not found
                raise ValueError("No symetry planes metadata found")

        # for now, using backprojected features as fallback
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = DINOWrapper(device)
            mesh_points, mesh_features = extractor.extract(model=model)

    else:
        raise ValueError("No object points found")  # handle error: no object points

    # Propagar features al gt del escaner
    features = (
        mesh_features.cpu().numpy() if torch.is_tensor(mesh_features) else mesh_features
    )
    mesh_points = (
        mesh_points.cpu().numpy() if torch.is_tensor(mesh_points) else mesh_points
    )

    propagated_features = knn_propagate(
        mesh_points, features, masked_points, instance.R, instance.t
    )

    return RegistratedData(
        scene_pts=scene_pts,
        masked_points=masked_points,
        posed_mesh_pts=posed_mesh_pts,
        pose_R=instance.R,
        pose_t=instance.t,
        obj_id=instance.obj_id,
        visib_fract=instance.visible_fract,
        propagated_features=propagated_features,
    )
