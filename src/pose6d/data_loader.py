from dataclasses import dataclass
from .config import LMOConfig

import numpy as np
from pathlib import Path
from typing import Iterator

import json
import cv2


# firma para instancias de objetos y su data asociada. (also LSP friendly :))
@dataclass(frozen=True)
class InstanceData:
    obj_id: int
    R: np.ndarray
    t: np.ndarray
    visible_fract: float | None
    px_count_visible: float | None


@dataclass(frozen=True)
class SceneData:
    scene_id: int
    img_id: int
    K: np.ndarray
    depth_scale: float
    instances: list[InstanceData]


class LMOLoader:
    """Carga metadatos e imágenes del dataset LM-O (test split)."""

    def __init__(self, config: LMOConfig) -> None:
        self.cfg = config
        self.paths = config.paths

    @classmethod
    def from_root(cls, root: str | Path) -> "LMOLoader":
        return cls(LMOConfig.from_root(root))

    # --- carga de JSONs ---

    def _load_json_int_keys(self, path: Path) -> dict:
        with open(path, "r") as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}

    def load_camera(self, scene_id: int, img_id: int) -> tuple[np.ndarray, float]:
        """Retorna (K, depth_scale) para una imagen."""
        data = self._load_json_int_keys(self.paths.scene_camera_path(scene_id))
        cam = data[img_id]
        K = np.asarray(cam["cam_K"], dtype=np.float64).reshape(3, 3)
        depth_scale = float(cam.get("depth_scale", 1.0))
        return K, depth_scale

    def load_instances(self, scene_id: int, img_id: int) -> list[InstanceData]:
        """Retorna todas las instancias de una imagen."""
        gt = self._load_json_int_keys(self.paths.scene_gt_path(scene_id))
        info = self._load_json_int_keys(self.paths.scene_gt_info_path(scene_id))

        inst_list = gt.get(img_id, [])
        info_list = info.get(img_id, [{}] * len(inst_list))

        parsed = []
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

    # --- carga de imágenes ---

    def load_depth(self, scene_id: int, img_id: int) -> np.ndarray:
        path = self.paths.depth_path(scene_id, img_id)
        d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if d is None:
            raise FileNotFoundError(path)
        if d.ndim == 3:
            d = d[..., 0]
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
    def list_image_ids(self, scene_id: int) -> list[int]:
        gt = self._load_json_int_keys(self.paths.scene_gt_path(scene_id))
        return sorted(gt.keys())

    def iter_images(self, scene_id: int) -> Iterator[SceneData]:
        for img_id in self.list_image_ids(scene_id):
            K, depth_scale = self.load_camera(scene_id, img_id)
            instances = self.load_instances(scene_id, img_id)
            yield SceneData(
                scene_id=scene_id,
                img_id=img_id,
                K=K,
                depth_scale=depth_scale,
                instances=instances,
            )

    # --- utilidades ---

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
