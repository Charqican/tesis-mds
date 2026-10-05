from dataclasses import dataclass
from pathlib import Path
from collections import Counter, defaultdict
from collections.abc import Iterator
import numpy as np
import torch

from pose6d.loader import BOPLoader, loader_from_env
from pose6d.features import compute_canonical_symmetry_field
from pose6d.geometry_utils import (
    isolate_object_points,
    backproject_depth,
    sample_farthest_points,
    propagate_symmetry_to_target,
)
import open3d as o3d

from logger import pose6d_preprocessing_logger as log


HIGH_OUTLIER_RATIO = 0.2

"""
Preprocessing extract a pointcloud from BOP dataset using depth images. Points are filtered using statistical outliers, then FPS is applied to obtain a n_points sized pointcloud
"""


# Extraction parameters of a pT version. Parameters from toml configuration.
@dataclass(frozen=True)
class ExtractParams:
    n_points: int
    min_visib: float
    outliers: dict | None = None  # {"nb_neighbors": int, "std": float}


# status is "ok" when points were extracted, otherwise the discard reason:
# "low_visib", "no_mask" or "few_points"
@dataclass(frozen=True)
class ExtractedInstance:
    uid: str
    points: np.ndarray | None
    status: str
    outlier_ratio: float = 0.0


def extract_frame_instances(
    loader: BOPLoader,
    scene_id: int,
    img_id: int,
    obj_ids: set[int],
    params: ExtractParams,
    inst_ids: set[int] | None = None,
) -> list[ExtractedInstance]:
    """
    Extracts the partial pointcloud (pT) of every instance of obj_ids in a frame.
    inst_ids restricts the extraction to specific instances of the frame.
    Discarded instances are returned too (points=None).
    """
    K, depth_scale = loader.load_camera(scene_id, img_id)
    instances = loader.load_instances(scene_id, img_id)
    depth = None

    results = []
    for inst_idx, instance in enumerate(instances):
        if instance.obj_id not in obj_ids:
            continue
        if inst_ids is not None and inst_idx not in inst_ids:
            continue
        uid = loader.instance_uid(scene_id, img_id, instance.obj_id, inst_idx)

        if (
            instance.visible_fract is not None
            and instance.visible_fract < params.min_visib
        ):
            results.append(ExtractedInstance(uid, None, "low_visib"))
            continue

        mask = loader.load_mask_visib(scene_id, img_id, inst_idx)
        if mask is None:
            results.append(ExtractedInstance(uid, None, "no_mask"))
            continue

        if depth is None:
            depth = loader.load_depth(scene_id, img_id)
        pts = isolate_object_points(depth, mask, K, depth_scale)

        ratio = 0.0
        if params.outliers and len(pts) >= params.n_points:
            outlier_mask = get_outliers_mask(
                pts, params.outliers["nb_neighbors"], params.outliers["std"]
            )
            ratio = float(outlier_mask.sum() / pts.shape[0])
            pts = pts[~outlier_mask]

        # checked after outlier removal, FPS can't sample more points than available
        if len(pts) < params.n_points:
            results.append(ExtractedInstance(uid, None, "few_points"))
            continue

        pts_sampled, _ = sample_farthest_points(
            torch.from_numpy(pts).unsqueeze(0), K=params.n_points
        )
        results.append(
            ExtractedInstance(uid, pts_sampled.squeeze(0).numpy(), "ok", ratio)
        )
    return results


# Kept for notebook_plots: yields (uid, points) of the extracted instances of a frame
def extract_instances_pcs(
    loader: BOPLoader,
    scene_id: int,
    img_id: int,
    target_obj_ids: list[int],
    min_visib_fract: float = 0.05,
    **kwargs,
) -> Iterator[tuple[str, np.ndarray]]:
    nb_neighbors, std = kwargs.get("nb_neighbors"), kwargs.get("std")
    outliers = (
        {"nb_neighbors": nb_neighbors, "std": std}
        if nb_neighbors is not None and std is not None
        else None
    )
    params = ExtractParams(loader.cfg.sample_points, min_visib_fract, outliers)
    for inst in extract_frame_instances(
        loader, scene_id, img_id, set(target_obj_ids), params
    ):
        if inst.points is not None:
            yield inst.uid, inst.points


# ------ pT extraction (scene / specific uids), used by scripts/pose6d_prepare_data.py
def _save_points(out_dir: Path, uid: str, points: np.ndarray) -> None:
    np.savez(out_dir / f"{uid}.npz", points=points.astype(np.float32))


def extract_scene(
    loader: BOPLoader,
    scene_id: int,
    obj_ids: set[int],
    params: ExtractParams,
    out_dir: Path,
) -> dict:
    """Saves the pT of every instance in a scene. Returns saved/discarded counts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    status = Counter()
    n_high_outliers = 0  # saved, but worth knowing
    for img_id in loader.list_image_ids(scene_id):
        for inst in extract_frame_instances(loader, scene_id, img_id, obj_ids, params):
            status[inst.status] += 1
            if inst.points is not None:
                _save_points(out_dir, inst.uid, inst.points)
                n_high_outliers += inst.outlier_ratio > HIGH_OUTLIER_RATIO
    return {
        "n_saved": status.pop("ok", 0),
        "n_high_outlier_ratio": n_high_outliers,
        "discarded": dict(status),
    }


def extract_uids(
    loader: BOPLoader,
    uids: list[str],
    obj_ids: set[int],
    params: ExtractParams,
    out_dir: Path,
) -> list[str]:
    """Saves the pT of specific instances. Returns the uids that were saved."""
    out_dir.mkdir(parents=True, exist_ok=True)
    by_frame: dict[tuple[int, int], dict[str, int]] = defaultdict(dict)
    for uid in uids:
        _, scene_id, img_id, _, inst_idx = loader.parse_instance_uid(uid)
        by_frame[(scene_id, img_id)][uid] = inst_idx

    saved = []
    for (scene_id, img_id), frame_uids in by_frame.items():
        found = {
            inst.uid: inst
            for inst in extract_frame_instances(
                loader, scene_id, img_id, obj_ids, params, set(frame_uids.values())
            )
        }
        for uid in frame_uids:
            inst = found.get(uid)
            if inst is None:
                log.warning(f"{uid}: no such instance (or object is not symmetric)")
            elif inst.points is None:
                log.warning(f"{uid}: discarded by the pT parameters ({inst.status})")
            else:
                _save_points(out_dir, uid, inst.points)
                saved.append(uid)
    return saved


# ------ symmetry field target
def compute_targets(
    points_paths: dict[str, Path],
    target_dir: Path,
    canonical_samples: int = 20000,
    seed: int = 0,
) -> None:
    """
    Propagates the canonical symmetry field of each object to its pT (uid -> pT path)
    and saves it in target_dir. The mesh sampling is seeded so targets are reproducible.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    loaders: dict[str, BOPLoader] = {}
    canonical: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for i, (uid, points_path) in enumerate(points_paths.items(), start=1):
        dataset, scene_id, img_id, obj_id, inst_idx = BOPLoader.parse_instance_uid(uid)
        if dataset not in loaders:
            loaders[dataset] = loader_from_env(dataset)
        loader = loaders[dataset]

        # every dataset uses the lmo models, so the canonical field is shared
        if obj_id not in canonical:
            torch.manual_seed(seed)
            canonical[obj_id] = compute_canonical_symmetry_field(
                loader, obj_id, canonical_samples
            )
        mesh_points, symmetry_scalar = canonical[obj_id]

        points = np.load(points_path)["points"]
        instance = loader.load_instances(scene_id, img_id)[inst_idx]
        target = propagate_symmetry_to_target(
            mesh_points, symmetry_scalar, points, instance.R, instance.t
        )
        np.savez(target_dir / f"{uid}.npz", target=target.astype(np.float32))
        if i % 1000 == 0:
            log.info(f"targets: {i}/{len(points_paths)}")


def extract_frames_pcs(loader: BOPLoader, scene_id, img_id) -> tuple[str, np.ndarray]:
    config = loader.cfg
    K, depth_scale = loader.load_camera(scene_id, img_id)
    depth_image = loader.load_depth(scene_id, img_id)
    frame_point_cloud = backproject_depth(
        depth_image, K, depth_scale, stride=config.depth_stride
    )
    return f"scene{scene_id:06d}_img{img_id}", frame_point_cloud


def extract_scene_frames_pcs(loader: BOPLoader, scene_id: int):
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
