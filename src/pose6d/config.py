from dataclasses import dataclass
from pathlib import Path
import numpy as np
import os

# Self Contained configurations & paths class.


# dataclass to decouple path access to other implementations. This uses the lmo-bop structure, but a different one can be implemented.
@dataclass(frozen=True)
class LMOPath:
    """
    File structure expected:
    lmo/
    - models/
    - models_eval/
    - test/
    - train/
    - camera.json
    - test_targets_bop19.sjon
    """

    root: Path

    # point to folder containing .obj objects
    @property
    def models_dir(self) -> Path:
        """
        Estructure expected:
        models/
        - models_info.json
        - obj_0000xx.json
        """
        return self.root / "models"

    # return models/models_info.json
    @property
    def models_info(self) -> Path:
        return self.models_dir / "models_info.json"

    # points to a scene in train or split using the scene_id
    # TODO: changing the split is to cumbersome, we should look into other options
    # WARNING: use only test split
    def scene_dir(self, scene_id: int, split: str = "test") -> Path:
        """
        Expected structure
        split/
        - scene_id/
            - depth/
            - mask/
            - mask_visib/
            - rgb/
            - scene_camera.json
            - scene_gt.json
            - scene_gt_info.json
        """
        return self.root / split / f"{scene_id:06d}"

    #  points to the rgb images of a scene
    def rgb_path(self, scene_id: int, img_id: int, split: str = "test") -> Path:
        return self.scene_dir(scene_id, split=split) / "rgb" / f"{img_id:06d}.png"

    # points to the depth images of a scene
    def depth_path(self, scene_id: int, img_id: int, split: str = "test") -> Path:
        return self.scene_dir(scene_id, split) / "depth" / f"{img_id:06d}.png"

    # gets a mask using the img_id and the 'instance_id', the latter refering to the order in which it appers in the corresponding scene_gt.json line
    def mask_visible_path(
        self, scene_id: int, img_id: int, instance_id: int, split: str = "test"
    ) -> Path:
        return (
            self.scene_dir(scene_id, split=split)
            / "mask_visib"
            / f"{img_id:06d}_{instance_id:06d}.png"
        )

    def scene_camera_path(self, scene_id: int, split: str = "test") -> Path:
        return self.scene_dir(scene_id, split=split) / "scene_camera.json"

    def scene_gt_path(self, scene_id: int, split: str = "test") -> Path:
        return self.scene_dir(scene_id, split=split) / "scene_gt.json"

    def scene_gt_info_path(self, scene_id: int, split="test") -> Path:
        return self.scene_dir(scene_id, split=split) / "scene_gt_info.json"

    # setup for this dataclass using .env file. It searchs for a LMO_ROOT key
    @classmethod
    def from_env(cls, env_var: str = "LMO_ROOT") -> "LMOPath":
        root = os.environ.get(env_var)
        if root is None:
            raise ValueError(f"Environment variable {env_var} not set")
        return cls(root=Path(root))

    # setup for this dataclass using a path to the lmo dataset
    @classmethod
    def from_root(cls, root: str | Path) -> "LMOPath":
        return cls(root=Path(root))


# TODO: configuration should be decoupled from path model. we should abstrct LMOPath to make future implementations easier
@dataclass(frozen=True)
class LMOConfig:
    """Parámetros del dataset y del pipeline."""

    paths: LMOPath
    # using test if none is given at the fun call
    default_scene: int = 2
    # configuration option for data loader
    depth_stride: int = 2
    mesh_samples: int = 20000

    @classmethod
    def from_root(cls, root: str | Path) -> "LMOConfig":
        return cls(paths=LMOPath.from_root(root))

    @classmethod
    def from_env(cls, env_var: str = "LMO_ROOT") -> "LMOConfig":
        return cls(paths=LMOPath.from_env(env_var))
