from pathlib import Path
from collections.abc import Iterator
import numpy as np
import torch

import logger
from pose6d.config import LMOConfig
from pose6d.loader import InstanceData, LMOLoader, instance_uid
from pose6d.geometry_utils import (
    isolate_object_points,
    backproject_depth,
    subsample_points,
    sample_farthest_points,
)

from logger import pose6d_preprocessing_logger as log


# Gives an iterator of every instance pointcloud in a scene
# TODO: maybe change this to return a list
def extract_instances_pcs(
    loader: LMOLoader,
    scene_id: int,
    img_id: int,
    target_obj_ids: list[int],
    min_visib_fract: float = 0.05,
) -> Iterator[tuple[str, np.ndarray]]:
    """
    Extrae nubes de puntos de instancias visibles en un frame específico.
    """
    # pose6d_preprocessing_logger.info(
    #     f"Targeting scene direcory: {loader.paths.scene_dir(scene_id)}"
    # )
    K, depth_scale = loader.load_camera(scene_id, img_id)
    instances = loader.load_instances(scene_id, img_id)
    depth = loader.load_depth(scene_id, img_id)
    config = loader.cfg

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
        if pts.shape[0] == 0 or (len(pts) < config.sample_points):
            continue
        pts_sampled, _ = sample_farthest_points(
            torch.from_numpy(pts), K=config.sample_points
        )

        uid = instance_uid(scene_id, img_id, instance.obj_id, inst_idx)
        yield uid, pts_sampled.numpy()


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
    log.info(f"Imgs in scene: {len(img_ids)}")
    for img_id in img_ids:
        yield from extract_instances_pcs(
            loader, scene_id, img_id, target_obj_ids, min_visib_fract
        )


def save_instance_pcs(
    instances: Iterator[tuple[str, np.ndarray]], out_dir: Path
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for uid, pts in instances:
        path = out_dir / f"{uid}.npz"
        np.savez(path, points=pts.astype(np.float32))
        saved.append(path)
        log.info(f"Saved {pts.shape[0]} points from instance {uid}")
    return saved


def extract_frames_pcs(loader, scene_id, img_id) -> tuple[str, np.ndarray]:
    config = loader.cfg
    K, depth_scale = loader.load_camera(scene_id, img_id)
    depth_image = loader.load_depth(scene_id, img_id)
    frame_point_cloud = backproject_depth(
        depth_image, K, depth_scale, stride=config.depth_stride
    )
    return f"scene{scene_id:06d}_img{img_id}", frame_point_cloud


def extract_scene_frames_pcs(loader: LMOLoader, scene_id: int):
    img_ids = loader.list_image_ids(scene_id)
    log.info(f"Imgs in scene: {len(img_ids)}")

    for img_id in img_ids:
        yield extract_frames_pcs(loader, scene_id, img_id)


def save_frame_pcs(frames: Iterator[tuple[str, np.ndarray]], out_dir: str | Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Pointclouds will be saved in : {out_dir}")
    saved = []
    for uid, pts in frames:
        # pose6d_preprocessing_logger.info(f"Saving uid: {uid}")
        path = out_dir / f"{uid}.npz"
        np.savez(path, points=pts.astype(np.float32))
        saved.append(path)
    return saved
