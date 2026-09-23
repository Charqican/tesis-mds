from pathlib import Path
from collections.abc import Iterator
from collections import Counter
import json
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
import open3d as o3d

from logger import pose6d_preprocessing_logger as log


# Gives an iterator of every instance pointcloud in a scene
# TODO: maybe change this to return a list
def extract_instances_pcs(
    loader: LMOLoader,
    scene_id: int,
    img_id: int,
    target_obj_ids: list[int],
    min_visib_fract: float = 0.05,
    **kwargs,
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
    nb_neighbors, std = kwargs.get("nb_neighbors"), kwargs.get("std")

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
            log.info(f"Skipping instance, n points: {len(pts)}")
            continue

        ratio = 0
        # if params are passed, remove outliers
        if nb_neighbors is not None and std is not None:
            outlier_mask = get_outliers_mask(pts, nb_neighbors, std)
            ratio = outlier_mask.sum() / pts.shape[0]
            pts = pts[~outlier_mask]
            log.info(f"Removing outliers, ratio: {ratio}")

        pts_sampled, _ = sample_farthest_points(
            torch.from_numpy(pts).unsqueeze(0), K=config.sample_points
        )

        uid = instance_uid(scene_id, img_id, instance.obj_id, inst_idx)

        if ratio > 0.2:
            log.warning(f"Outlier ratio too high, uid: {uid}, ratio: {ratio}")
        yield uid, pts_sampled.squeeze(0).numpy()


def extract_scene_instances_pcs(
    loader: LMOLoader,
    scene_id: int,
    target_obj_ids: list[int],
    min_visib_fract: float = 0.05,
    **kwargs,
) -> Iterator[tuple[str, np.ndarray]]:
    """
    Extrae nubes de puntos de instancias visibles en toda una escena.
    """
    img_ids = loader.list_image_ids(scene_id)
    log.info(f"Imgs in scene: {len(img_ids)}")
    for img_id in img_ids:
        yield from extract_instances_pcs(
            loader,
            scene_id,
            img_id,
            target_obj_ids,
            min_visib_fract,
            **kwargs,
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
        # log.info(f"Saved {} points from instance {uid}")
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


def save_pT_version(
    loader: LMOLoader,
    scene_id: int | list[int],
    version_name: str,
    total_objects: int,
    uids: list[str],
    out_dir: Path,
    **kwargs,
) -> dict:
    objects_tuples = [tuple(loader.parse_instance_uid(u)[2:4]) for u in uids]
    obj_counts = Counter(obj_id for obj_id, _ in objects_tuples)

    metadata = {
        "version_name": version_name,
        "scene_id": scene_id,
        "total_objects": total_objects,
        "n_saved": len(uids),
        "n_discarded": total_objects - len(uids),
        "objects_per_id": dict(obj_counts),
        "uids": uids,
        **kwargs,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{version_name}.json"
    out_path.write_text(json.dumps(metadata, indent=2))

    return metadata


def get_outliers_mask(pts: np.ndarray, nb_neighbors: float, std: float):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    pcd_clean, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std
    )
    ind_idx = np.asarray(ind)
    outlier_mask = np.ones(pts.shape[0], dtype=bool)
    outlier_mask[ind_idx] = False
    return outlier_mask
