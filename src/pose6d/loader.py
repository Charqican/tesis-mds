from dataclasses import dataclass
from .config import LMOConfig

import numpy as np
import torch
import cv2

from pathlib import Path
from typing import Iterator
import json

"""
1. This file contains dataclasses as interfaces to decouple the path resolver & dataloader from the dataset and its schema. 
2. The implementation of LMOLoader uses the LMOConfig (which contains LMOPath as a field) to resolve access to the data. 
"""


# TODO: if needed a 'parser' can be abstracted for differnet datasets
# TODO: missing axial symmetry implementation and getters


# InstanceData abstracts the concept of specific object in a frame.
@dataclass(frozen=True)
class InstanceData:
    obj_id: int
    R: np.ndarray
    t: np.ndarray
    visible_fract: float | None
    px_count_visible: float | None


# FrameData abstracts the concept of a given image with objects in a particular scene.
# it contains a list of instanceData
@dataclass(frozen=True)
class FrameData:
    scene_id: int
    img_id: int
    K: np.ndarray
    depth_scale: float
    instances: list[InstanceData]


# SymmetryData gives a convention for symetry planes (used to extract a symmetry field)
@dataclass(frozen=True)
class SymmetryData:
    normal: torch.Tensor
    plane_point: torch.Tensor


# Atributtes of a canonical object. Symmetries in here represents a matrix of translation + rotation
@dataclass(frozen=True)
class ModelInfo:
    obj_id: int
    diameter: float
    min_xyz: np.ndarray  # (3,)
    size_xyz: np.ndarray  # (3,)
    symmetries_discrete: list[np.ndarray] | None  # cada uno (4,4), o None


# WARNING: in case of multiple symmetries only one is returned
class LMOLoader:
    """
    helper dataclass that resolves, loads and parses data from the raw dataset.

    ## Fields

    config: LMOConfig
        A config dataclass that containts the necessary information to load the dataset

    ## Methods

    load_camera(scene_id, img_id):
        returns (K, depth_scale) for a given image.

    load_instances(scene_id, img_id):
        returns a list of instances (partial objects) of a given image.

    load_depth(scene_id, img_id):
        returns the depth image for a given frame using its img_id.

    load_rgb(scene_id, img_id):
        returns the rgb image for a given frame by its image_id.

    load_mask_visib(scene_id, img_id, inst_idx):
        return the visible mask for a given instance inside a frame.

    load_models_info:
        returns the parsed metadata of all objects

    list_img_ids(scene_id):
        returns every frame id (img_id) inside a scene

    iter_frames(scene_id):
        returns an iterator of FrameData of every image inside a scene

    load_symmetry_data(obj_id):
        returns a SymmetryData if the object has symmetries, None if not. It only returns one plane.

    symmetric_obj_ids():
        returns every symmetric object id.

    parse_instance_uid(uid).
        returns every id that composes an uid. An uid identifies a partial pointcloud of a given instance of an object in a particular frame inside a scene.
    """

    def __init__(self, config: LMOConfig) -> None:
        self.cfg = config
        self.paths = config.paths

    @classmethod
    def from_root(cls, root: str | Path) -> "LMOLoader":
        return cls(LMOConfig.from_root(root))

    def load_camera(self, scene_id: int, img_id: int) -> tuple[np.ndarray, float]:
        data = self._load_json_int_keys(self.paths.scene_camera_path(scene_id))
        cam = data[img_id]
        K = np.asarray(cam["cam_K"], dtype=np.float64).reshape(3, 3)
        depth_scale = float(cam.get("depth_scale", 1.0))
        return K, depth_scale

    def load_instances(self, scene_id: int, img_id: int) -> list[InstanceData]:
        "Returns all instances of an image as a list of InstanceData"
        gt = self._load_json_int_keys(self.paths.scene_gt_path(scene_id))
        info = self._load_json_int_keys(self.paths.scene_gt_info_path(scene_id))

        inst_list = gt.get(img_id, [])
        info_list = info.get(img_id, [{}] * len(inst_list))

        parsed = []  # positions are the same as they are saved the same order they appear in the json
        for inst, meta in zip(inst_list, info_list):
            parsed.append(
                InstanceData(
                    obj_id=int(inst["obj_id"]),
                    R=np.asarray(inst["cam_R_m2c"], dtype=np.float64).reshape(3, 3),
                    t=np.asarray(inst["cam_t_m2c"], dtype=np.float64).reshape(3),
                    visible_fract=meta.get("visib_fract"),
                    px_count_visible=meta.get("px_count_visib"),
                )
            )
        return parsed

    def load_models_info(self) -> dict[int, ModelInfo]:
        "Returns a dictionary with every object ModelInfo as a value and id as a key"
        raw = json.loads(self.cfg.paths.models_info.read_text())
        result = {}
        for obj_id_str, info in raw.items():
            obj_id = int(obj_id_str)
            sym = info.get("symmetries_discrete")
            result[obj_id] = ModelInfo(
                obj_id=obj_id,
                diameter=info["diameter"],
                min_xyz=np.array([info["min_x"], info["min_y"], info["min_z"]]),
                size_xyz=np.array([info["size_x"], info["size_y"], info["size_z"]]),
                symmetries_discrete=(
                    [np.array(m).reshape(4, 4) for m in sym]
                    if sym is not None
                    else None
                ),
            )
        return result

    # WARNING:only one symmetry
    def load_symmetry_plane(self, obj_id: int) -> SymmetryData | None:

        with open(self.paths.models_info) as f:
            info = json.load(f)

        obj_data = info.get(str(obj_id))
        if not obj_data:
            return None

        symmetries = obj_data.get("symmetries_discrete")
        if not symmetries:
            return None

        return self._parse_symmetry_matrix(
            symmetries[0]
        )  # harcoded to 1 plane of symmetry

    # --- Image loading ---
    def load_depth(self, scene_id: int, img_id: int) -> np.ndarray:
        path = self.paths.depth_path(scene_id, img_id)
        d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if d is None:
            raise FileNotFoundError(path)
        if d.ndim == 3:
            d = d[..., 0]
        return d

    def load_rgb(self, scene_id: int, img_id: int) -> np.ndarray:
        path = self.paths.rgb_path(scene_id, img_id)
        d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if d is None:
            raise FileNotFoundError(path)
        if d.ndim == 3 and d.shape[2] == 3:
            d = cv2.cvtColor(d, cv2.COLOR_BGR2RGB)
        return d

    def load_mask_visib(
        self, scene_id: int, img_id: int, inst_idx: int
    ) -> np.ndarray | None:
        path = self.paths.mask_visible_path(scene_id, img_id, inst_idx)
        if not path.exists():
            return None
        m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if m is None:
            return None
        return m > 0

    # --- iterator ---
    # return the img_ids found in the GT, ignoring orphan frames
    def list_image_ids(self, scene_id: int) -> list[int]:
        gt = self._load_json_int_keys(self.paths.scene_gt_path(scene_id))
        return sorted(gt.keys())

    # returns an iterator of frames containing the instances of each frame
    def iter_frames(self, scene_id: int) -> Iterator[FrameData]:
        for img_id in self.list_image_ids(scene_id):
            K, depth_scale = self.load_camera(scene_id, img_id)
            instances = self.load_instances(scene_id, img_id)
            yield FrameData(
                scene_id=scene_id,
                img_id=img_id,
                K=K,
                depth_scale=depth_scale,
                instances=instances,
            )

    # iterate over object metadata to filter non discrete symmetries
    def symmetric_obj_ids(self) -> set[int]:
        return {
            obj_id
            for obj_id, info in self.load_models_info().items()
            if info.symmetries_discrete is not None
        }

    def parse_instance_uid(self, uid: str) -> tuple[int, int, int, int]:
        """retrieves the instance ids of a given uid: uid -> (scene_id, img_id, obj_id, inst_idx)."""
        parts = uid.split("_")
        scene_id = int(parts[0].removeprefix("scene"))
        img_id = int(parts[1].removeprefix("img"))
        obj_id = int(parts[2].removeprefix("obj"))
        inst_idx = int(parts[3].removeprefix("inst"))
        return scene_id, img_id, obj_id, inst_idx

    # --- internal parsers ---
    def _load_json_int_keys(self, path: Path) -> dict:
        with open(path, "r") as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}

    def _parse_symmetry_matrix(self, symm_matrix: list[float]) -> SymmetryData:
        M = np.array(symm_matrix, dtype=np.float64).reshape(4, 4)
        R = M[:3, :3]
        t = M[:3, 3]

        eigvals, eigvecs = np.linalg.eig(R)
        normal = eigvecs[:, np.argmin(eigvals)].real
        normal = normal / np.linalg.norm(normal)
        point = t / 2.0

        return SymmetryData(
            torch.tensor(normal, dtype=torch.float32),
            torch.tensor(point, dtype=torch.float32),
        )

    # --- utilities ---
    def get_best_instance(
        self, scene_id: int, img_id: int, min_visib: float = 0.5
    ) -> tuple[int, InstanceData] | None:
        """Retorna (inst_idx, instance) con mayor visib_fract >= min_visib."""
        instances = self.load_instances(scene_id, img_id)
        best_idx = None
        best_vf = -1.0

        for i, inst in enumerate(instances):
            if inst.visible_fract is None:
                continue
            if inst.visible_fract >= min_visib and inst.visible_fract > best_vf:
                best_vf = inst.visible_fract
                best_idx = i

        if best_idx is None:
            return None
        return best_idx, instances[best_idx]


# uid generator
# REFACTOR: maybe this should go inside its dataclass
def instance_uid(scene_id: int, img_id: int, obj_id: int, inst_idx: int) -> str:
    return f"scene{scene_id:06d}_img{img_id:06d}_obj{obj_id:06d}_inst{inst_idx:02d}"
