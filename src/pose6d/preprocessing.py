from pathlib import Path
from collections.abc import Iterator
import numpy as np

from pose6d.config import LMOConfig
from pose6d.loader import LMOLoader, instance_uid
from pose6d.geometry_utils import isolate_object_points


# Gives an iterator of every instance pointcloud in a scene
def extract_scene_instances_pcs(
    loader: LMOLoader,
    scene_id: int,
    target_obj_ids: list[int],
    min_visib_fract: float = 0.05,
) -> Iterator[tuple[str, np.ndarray]]:

    img_ids = loader.list_image_ids(scene_id)
    for img_id in img_ids:
        K, depth_scale = loader.load_camera(scene_id, img_id)
        # IO happens here
        instances = loader.load_instances(scene_id, img_id)
        depth = loader.load_depth(scene_id, img_id)

        for inst_idx, instance in enumerate(instances):
            # Ignore all instances of objects not in the set of ids
            if instance.obj_id not in target_obj_ids:
                continue
            # Ignore all instances with minimal visibility
            if (
                instance.visible_fract is not None
                and instance.visible_fract < min_visib_fract
            ):
                continue
            # IO for mask
            mask = loader.load_mask_visib(scene_id, img_id, inst_idx)
            if mask is None:
                continue

            # segment pointcloud
            pts = isolate_object_points(depth, mask, K, depth_scale)
            if pts.shape[0] == 0:
                continue

            uid = instance_uid(scene_id, img_id, instance.obj_id, inst_idx)
            yield uid, pts


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
