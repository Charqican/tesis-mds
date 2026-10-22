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


@dataclass(frozen=True)
class YCBVPath(BOPPath):
    """
    Expected file structure ycbv
    - test/
    - ycbv_models/ (models/, models_eval/, models_fine/)
    - test_targets_bop19.json

    test holds BOP scenes. Models live in root/ycbv_models/models/.
    """

    root: Path

    def scene_dir(self, scene_id: int) -> Path:
        return self.root / "test" / f"{scene_id:06d}"

    @classmethod
    def from_root(cls, root: str | Path) -> "YCBVPath":
        root = Path(root)
        return cls(root=root, model_root=root / "ycbv_models")


@dataclass(frozen=True)
class TLESSPath(BOPPath):
    """
    Expected file structure tless
    - test_primesense/   (real, BOP19 test)
    - train_pbr/         (synthetic)
    - tless_models/ (models_cad/, models_eval/, models_reconst/)
    - test_targets_bop19.json

    scenes_subdir picks the BOP scenes (test_primesense or train_pbr). Both use the
    CAD models in root/tless_models/models_cad/.
    """

    root: Path
    scenes_subdir: str = "test_primesense"

    def scene_dir(self, scene_id: int) -> Path:
        return self.root / self.scenes_subdir / f"{scene_id:06d}"

    @property
    def models_dir(self) -> Path:
        return self.model_root / "models_cad"

    @classmethod
    def from_root(
        cls, root: str | Path, scenes_subdir: str = "test_primesense"
    ) -> "TLESSPath":
        root = Path(root)
        return cls(
            root=root, model_root=root / "tless_models", scenes_subdir=scenes_subdir
        )


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


@dataclass(frozen=True)
class YCBVConfig:
    """Config for YCB-V test scenes (BOP19 subset)."""

    paths: YCBVPath
    depth_stride: int = 2
    sample_points: int = 1024

    @classmethod
    def from_root(cls, root: str | Path) -> "YCBVConfig":
        return cls(paths=YCBVPath.from_root(root))


@dataclass(frozen=True)
class TLESSConfig:
    """Config for T-LESS scenes: real test (primesense) or synthetic train_pbr."""

    paths: TLESSPath
    depth_stride: int = 2
    sample_points: int = 1024

    @classmethod
    def from_root(
        cls, root: str | Path, scenes_subdir: str = "test_primesense"
    ) -> "TLESSConfig":
        return cls(paths=TLESSPath.from_root(root, scenes_subdir))
