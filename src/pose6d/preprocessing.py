from pathlib import Path
from collections.abc import Iterator
import numpy as np

from pose6d.config import LMOConfig
from pose6d.loader import InstanceData, LMOLoader, instance_uid
from pose6d.geometry_utils import isolate_object_points


# Gives an iterator of every instance pointcloud in a scene
# TODO: change this to return a list
def extract_frame_instances_pcs(
    loader: LMOLoader,
    scene_id: int,
    img_id: int,
    target_obj_ids: list[int],
    min_visib_fract: float = 0.05,
) -> Iterator[tuple[str, np.ndarray]]:
    """
    Extrae nubes de puntos de instancias visibles en un frame específico.
    """
    K, depth_scale = loader.load_camera(scene_id, img_id)
    instances = loader.load_instances(scene_id, img_id)
    depth = loader.load_depth(scene_id, img_id)

    for inst_idx, instance in enumerate(instances):
        if instance.obj_id not in target_obj_ids:
            continue
        if (
            instance.visible_fract is not None
            and instance.visible_fract < min_visib_fract
        ):
            continue

        mask = loader.load_mask_visib(scene_id, img_id, inst_idx)
        if mask is None:
            continue

        pts = isolate_object_points(depth, mask, K, depth_scale)
        if pts.shape[0] == 0:
            continue

        uid = instance_uid(scene_id, img_id, instance.obj_id, inst_idx)
        yield uid, pts


def extract_scene_instances_pcs(
    loader: LMOLoader,
    scene_id: int,
    target_obj_ids: list[int],
    min_visib_fract: float = 0.05,
) -> Iterator[tuple[str, np.ndarray]]:
    """
    Extrae nubes de puntos de instancias visibles en toda una escena.
    """
    img_ids = loader.list_image_ids(scene_id)
    for img_id in img_ids:
        yield from extract_frame_instances_pcs(
            loader, scene_id, img_id, target_obj_ids, min_visib_fract
        )


def save_pointclouds(
    instances: Iterator[tuple[str, np.ndarray]], out_dir: Path
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for uid, pts in instances:
        path = out_dir / f"{uid}.npz"
        np.savez(path, points=pts.astype(np.float32))
        saved.append(path)
    return saved
