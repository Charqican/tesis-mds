from dataclasses import dataclass
from pathlib import Path
import numpy as np
import os

# Self Contained configurations & paths class.


@dataclass(frozen=True)
class BOPPath:
    """
    Path resolver class. Expects the following scene structure
    - depth/
    - mask/
    - mask_visib/
    - rgb/
    - scene_camera.json
    - scene_gt.json
    - scene_gt_info.json

    Also resolves object model files, rooted at model_root/models/. Every
    BOP-format dataset points model_root somewhere. Scene_dir is dataset-specific.
    """

    model_root: Path

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

    @property
    def models_dir(self) -> Path:
        return self.model_root / "models"

    @property
    def models_info(self) -> Path:
        return self.models_dir / "models_info.json"

    def model_path(self, obj_id: int) -> Path:
        return self.models_dir / f"obj_{obj_id:06d}.ply"


@dataclass(frozen=True)
class LMOPath(BOPPath):
    """
    Expected file structure lmo
    - models/
    - models_eval/
    - test/
    - train/
    - camera.json
    - test_targets_bop19.sjon

    test & train are BOP scenes. Models are its own tree (model_root == root).
    """

    root: Path

    def scene_dir(self, scene_id: int) -> Path:
        path_test = self.root / "test" / f"{scene_id:06d}"
        path_train = self.root / "train" / f"{scene_id:06d}"
        return path_test if path_test.exists() else path_train

    @classmethod
    def from_env(cls, env_var: str = "LMO_ROOT") -> "LMOPath":
        root = os.environ.get(env_var)
        if root is None:
            raise ValueError(f"Environment variable {env_var} not set")
        return cls.from_root(root)

    @classmethod
    def from_root(cls, root: str | Path) -> "LMOPath":
        root = Path(root)
        return cls(root=root, model_root=root)


@dataclass(frozen=True)
class PBRPath(BOPPath):
    """
    root/
        - scene_xxxx1
        - scene_xxxx2
        - ...

    model_root points to wherever the object models live (e.g. lmo/ root).
    """

    root: Path

    def scene_dir(self, scene_id: int) -> Path:
        return self.root / f"{scene_id:06d}"

    @classmethod
    def from_roots(cls, root: str | Path, model_root: str | Path) -> "PBRPath":
        return cls(root=Path(root), model_root=Path(model_root))


# TODO: Track if mesh_samples actually makes it to the implementation or if it ends up lost
@dataclass(frozen=True)
class LMOConfig:
    """Parámetros del dataset y del pipeline."""

    paths: LMOPath
    default_scene: int = 2
    depth_stride: int = 2
    sample_points: int = 1024

    @classmethod
    def from_root(cls, root: str | Path) -> "LMOConfig":
        return cls(paths=LMOPath.from_root(root))

    @classmethod
    def from_env(cls, env_var: str = "LMO_ROOT") -> "LMOConfig":
        return cls(paths=LMOPath.from_env(env_var))


@dataclass(frozen=True)
class PBRConfig:
    """Config for BOP PBR (train_pbr) scenes; model info resolved from an external root."""

    paths: PBRPath
    depth_stride: int = 2
    sample_points: int = 1024

    @classmethod
    def from_roots(cls, root: str | Path, model_root: str | Path) -> "PBRConfig":
        return cls(paths=PBRPath.from_roots(root, model_root))
