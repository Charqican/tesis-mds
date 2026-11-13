from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from collections import Counter, defaultdict
from collections.abc import Iterator
import numpy as np
import torch

from pose6d.loader import BOPLoader, loader_from_env
from pose6d.features import compute_canonical_symmetry_field
from pose6d.geometry_utils import (
    BAND,
    MATCHED,
    UNMATCHED,
    isolate_object_points,
    backproject_depth,
    median_spacing,
    sample_farthest_points,
    propagate_symmetry_to_target,
    point_cloud_diameter,
    symmetric_correspondence,
    to_object_frame,
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
    cluster: dict | None = None  # {"eps": mm, "min_points": int}, keeps the largest DBSCAN cluster
    sphere: dict | None = None  # {"k": float}, keeps ||p - t|| <= k * CAD radius (GT pose)


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

        # outlier removal before FPS: CAD sphere, statistical, then the largest cluster
        ratio = 0.0
        if len(pts) >= params.n_points and (
            params.sphere or params.outliers or params.cluster
        ):
            n_raw = len(pts)
            if params.sphere:
                radius = cad_radius(loader.paths.model_path(instance.obj_id))
                pts = pts[
                    ~object_sphere_outliers_mask(
                        pts, instance.t, params.sphere["k"] * radius
                    )
                ]
            if params.outliers:
                pts = pts[
                    ~get_outliers_mask(
                        pts, params.outliers["nb_neighbors"], params.outliers["std"]
                    )
                ]
            if params.cluster:
                pts = pts[
                    ~largest_cluster_outliers_mask(
                        pts, params.cluster["eps"], params.cluster["min_points"]
                    )
                ]
            ratio = 1 - len(pts) / n_raw

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
# Scale (mm) that divides the target of one instance:
#   "pT"    diameter of the partial cloud, like dGeDi does with its input (changes with
#           the view and with the outliers left in the pT)
#   "model" diameter of the object mesh (models_info.json), the same for every instance
TARGET_SCALES = ("pT", "model")


def target_scale(
    normalize_by: str, points: np.ndarray, loader: BOPLoader, obj_id: int
) -> float:
    if normalize_by == "pT":
        return point_cloud_diameter(points)
    if normalize_by == "model":
        return loader.load_models_info()[obj_id].diameter
    raise ValueError(f"normalize_by must be one of {TARGET_SCALES}, got {normalize_by!r}")


def model_diameter(uid: str) -> float:
    """Diameter (mm, models_info.json) of the object of an instance uid."""
    dataset, _, _, obj_id, _ = BOPLoader.parse_instance_uid(uid)
    return _models_info(dataset)[obj_id].diameter


@lru_cache
def _models_info(dataset: str) -> dict:
    return loader_from_env(dataset).load_models_info()


def compute_targets(
    points_paths: dict[str, Path],
    target_dir: Path,
    canonical_samples: int = 20000,
    seed: int = 0,
    normalize_target: bool = True,
    normalize_by: str = "pT",
    symmetry_planes: dict[str, dict[int, int]] | None = None,
) -> None:
    """
    Propagates the canonical symmetry field of each object to its pT (uid -> pT path)
    and saves it in target_dir. The mesh sampling is seeded so targets are reproducible.
    Saves `scale` too (1.0 when not normalized): target * scale is in mm.
    normalize_by picks the diameter used as scale (see target_scale).
    symmetry_planes: dataset -> obj_id -> candidate plane index (BOPLoader.load_symmetry_plane),
    objects not listed keep the legacy plane.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    loaders: dict[str, BOPLoader] = {}
    canonical: dict[tuple[Path, int, int | None], tuple[np.ndarray, np.ndarray]] = {}

    for i, (uid, points_path) in enumerate(points_paths.items(), start=1):
        dataset, scene_id, img_id, obj_id, inst_idx = BOPLoader.parse_instance_uid(uid)
        if dataset not in loaders:
            loaders[dataset] = loader_from_env(dataset)
        loader = loaders[dataset]
        plane_index = (symmetry_planes or {}).get(dataset, {}).get(obj_id)

        # keyed by models dir: lmo/pbr share models, but ycbv obj ids mean other objects
        key = (loader.paths.models_dir, obj_id, plane_index)
        if key not in canonical:
            torch.manual_seed(seed)
            canonical[key] = compute_canonical_symmetry_field(
                loader, obj_id, canonical_samples, plane_index
            )
        mesh_points, symmetry_scalar = canonical[key]

        points = np.load(points_path)["points"]
        instance = loader.load_instances(scene_id, img_id)[inst_idx]
        target = propagate_symmetry_to_target(
            mesh_points, symmetry_scalar, points, instance.R, instance.t
        )
        scale = (
            target_scale(normalize_by, points, loader, obj_id)
            if normalize_target
            else 1.0
        )
        np.savez(
            target_dir / f"{uid}.npz",
            target=(target / scale).astype(np.float32),
            scale=np.float32(scale),
        )
        if i % 1000 == 0:
            log.info(f"targets: {i}/{len(points_paths)}")


# ------ symmetry available in the pT
def available_symmetry(
    points_paths: dict[str, Path],
    band_pct: float = 0.10,
    k_tol: float = 2.0,
    canonical_samples: int = 20000,
    seed: int = 0,
    symmetry_planes: dict[str, dict[int, int]] | None = None,
) -> list[dict]:
    """
    One row per uid (uid -> pT path) with the symmetric_correspondence summary, on the
    same plane as the target (BOPLoader.load_symmetry_plane, symmetry_planes as in compute_targets):
      band = band_pct * field_range (CAD), tol = k_tol * median spacing of the pT
      available_symm = matched / (matched + unmatched), points outside the band only
    """
    loaders: dict[str, BOPLoader] = {}
    planes: dict[tuple[Path, int, int | None], tuple[np.ndarray, np.ndarray, float]] = {}
    rows = []
    for uid, points_path in points_paths.items():
        dataset, scene_id, img_id, obj_id, inst_idx = BOPLoader.parse_instance_uid(uid)
        if dataset not in loaders:
            loaders[dataset] = loader_from_env(dataset)
        loader = loaders[dataset]
        plane_index = (symmetry_planes or {}).get(dataset, {}).get(obj_id)

        # field_range: max of the canonical field, half extent of the CAD along the normal
        key = (loader.paths.models_dir, obj_id, plane_index)
        if key not in planes:
            torch.manual_seed(seed)
            _, field = compute_canonical_symmetry_field(loader, obj_id, canonical_samples, plane_index)
            plane = loader.load_symmetry_plane(obj_id, plane_index)
            planes[key] = (plane.normal.numpy(), plane.plane_point.numpy(), float(field.max()))
        normal, plane_point, field_range = planes[key]

        instance = loader.load_instances(scene_id, img_id)[inst_idx]
        points = to_object_frame(np.load(points_path)["points"], instance.R, instance.t)
        band, tol = band_pct * field_range, k_tol * median_spacing(points)
        labels = symmetric_correspondence(points, normal, plane_point, band, tol)
        n_matched, n_unmatched = (labels == MATCHED).sum(), (labels == UNMATCHED).sum()
        rows.append(
            {
                "uid": uid,
                "field_range": field_range,
                "band": band,
                "tol": tol,
                "frac_band": float((labels == BAND).mean()),
                "available_symm": n_matched / max(n_matched + n_unmatched, 1),
            }
        )
    return rows


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


# ------ alternative outlier masks (compared in notebooks/pose6d_outlier_removal.py)
# All of them return a boolean mask, True = outlier, like get_outliers_mask.
def radius_outliers_mask(pts: np.ndarray, nb_points: int, radius: float) -> np.ndarray:
    """Outlier: fewer than nb_points neighbors within radius (mm)."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    _, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    outlier_mask = np.ones(pts.shape[0], dtype=bool)
    outlier_mask[np.asarray(ind, dtype=int)] = False
    return outlier_mask


def largest_cluster_outliers_mask(pts: np.ndarray, eps: float, min_points: int) -> np.ndarray:
    """Outlier: not in the largest DBSCAN cluster (eps in mm). Removes groups of points
    separated from the object, which the statistical removal keeps."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    if (labels < 0).all():
        return np.zeros(pts.shape[0], dtype=bool)
    return labels != np.bincount(labels[labels >= 0]).argmax()


@lru_cache
def cad_radius(model_path: Path) -> float:
    """Radius (mm) of the CAD around its origin: max ||v|| of the mesh vertices. The GT
    translation t is that origin in the camera frame. Not diameter / 2: the BOP diameter
    doesn't depend on the origin (eggbox 83.0 vs 82.3, glue 91.8 vs 87.9, up to 17% more on
    the other lmo objects)."""
    vertices = np.asarray(o3d.io.read_triangle_mesh(str(model_path)).vertices)
    return float(np.linalg.norm(vertices, axis=1).max())


def object_sphere_outliers_mask(
    pts: np.ndarray, center: np.ndarray, radius: float
) -> np.ndarray:
    """Outlier: farther than radius (mm) from center, the GT translation of the object.
    Like SAM-6D training (radius = 1.2 * object radius), but with the CAD radius."""
    return np.linalg.norm(pts - np.asarray(center).reshape(1, 3), axis=1) > radius


def depth_outliers_mask(pts: np.ndarray, max_deviation: float) -> np.ndarray:
    """Outlier: depth (camera z) farther than max_deviation (mm) from the median depth of
    the instance. With max_deviation ~ the model diameter it drops mask pixels that hit the
    background."""
    return np.abs(pts[:, 2] - np.median(pts[:, 2])) > max_deviation
