from dataclasses import dataclass
from pathlib import Path
import numpy as np
import os

# Self Contained configurations & paths class.


@dataclass(frozen=True)
class BOPPath:
    """
    Path resolver template class. Expects the followint scene structure
    - depth/
    - mask/
    - mask_visib/
    - rgb/
    - scene_camera.json
    - scene_gt.json
    - scene_gt_info.json
    """

    def scene_dir(self, scene_id: int) -> Path:
        raise NotImplementedError

    def rgb_path(self, scene_id: int, img_id: int) -> Path:
        return self.scene_dir(scene_id) / "rgb" / f"{img_id:06d}.png"

    def depth_path(self, scene_id: int, img_id: int) -> Path:
        return self.scene_dir(scene_id) / "depth" / f"{img_id:06d}.png"

    def mask_visible_path(self, scene_id: int, img_id: int, instance_id: int) -> Path:
        return (
            self.scene_dir(scene_id)
            / "mask_visib"
            / f"{img_id:06d}_{instance_id:06d}.png"
        )

    def scene_camera_path(self, scene_id: int) -> Path:
        return self.scene_dir(scene_id) / "scene_camera.json"

    def scene_gt_path(self, scene_id: int) -> Path:
        return self.scene_dir(scene_id) / "scene_gt.json"

    def scene_gt_info_path(self, scene_id: int) -> Path:
        return self.scene_dir(scene_id) / "scene_gt_info.json"


@dataclass(frozen=True)
class LMOPath(BOPPath):
    """
    Expected file structure lmo/
    - models/
    - models_eval/
    - test/
    - train/
    - camera.json
    - test_targets_bop19.sjon

    test & train are BOP scenes.
    """

    root: Path

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def models_info(self) -> Path:
        return self.models_dir / "models_info.json"

    def model_path(self, obj_id: int) -> Path:
        return self.models_dir / f"obj_{obj_id:06d}.ply"

    def scene_dir(self, scene_id: int) -> Path:
        path_test = self.root / "test" / f"{scene_id:06d}"
        path_train = self.root / "train" / f"{scene_id:06d}"
        return path_test if path_test.exists() else path_train

    @classmethod
    def from_env(cls, env_var: str = "LMO_ROOT") -> "LMOPath":
        root = os.environ.get(env_var)
        if root is None:
            raise ValueError(f"Environment variable {env_var} not set")
        return cls(root=Path(root))

    @classmethod
    def from_root(cls, root: str | Path) -> "LMOPath":
        return cls(root=Path(root))


@dataclass(frozen=True)
class PBRPath(BOPPath):
    """
    root/
        - scene_xxxx1
        - scene_xxxx2
        - ...
    model_root/
        - models/
    """

    root: Path
    model_root: Path

    def scene_dir(self, scene_id: int) -> Path:
        return self.root / f"scene_{scene_id:06d}"

    @property
    def models_dir(self) -> Path:
        return self.model_root / "models"

    def model_path(self, obj_id: int) -> Path:
        return self.models_dir / f"obj_{obj_id:06d}.ply"


# TODO: configuration should be decoupled from path model. we should abstrct LMOPath (eg. PathResolver) to make future dataset implementations easier
# TODO: Track if mesh_samples actually makes it to the implementation or if it ends up lost
@dataclass(frozen=True)
class LMOConfig:
    """Parámetros del dataset y del pipeline."""

    paths: LMOPath
    # using test if none is given at the fun call
    default_scene: int = 2
    # configuration option for data loader
    depth_stride: int = 2
    # any backprojected pointcloud should have a min viable pointcloud
    sample_points: int = 1024

    @classmethod
    def from_root(cls, root: str | Path) -> "LMOConfig":
        return cls(paths=LMOPath.from_root(root))

    @classmethod
    def from_env(cls, env_var: str = "LMO_ROOT") -> "LMOConfig":
        return cls(paths=LMOPath.from_env(env_var))
