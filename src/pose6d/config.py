from dataclasses import dataclass
from pathlib import Path
import numpy as np
import os

# Self Contained configurations & paths class.


# dataclass to decouple path access to other implementations. This uses the lmo-bop structure, but a different one can be implemented.
@dataclass(frozen=True)
class LMOPath:
    """
    Path resolver dataclass. Used internally by loader.
    Expected file structure lmo/
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
        Expected file structure:
        models/
        - models_info.json
        - obj_0000xx.json
        """
        return self.root / "models"

    # return models/models_info.json
    @property
    def models_info(self) -> Path:
        return self.models_dir / "models_info.json"

    def model_path(self, obj_id: int):
        return self.models_dir / f"obj_{obj_id:06d}.ply"

    # points to a scene in train or split using the scene_id
    def scene_dir(self, scene_id: int) -> Path:
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
        try:
            path = self.root / "test" / f"{scene_id:06d}"

        except FileNotFoundError:
            path = self.root / "train" / f"{scene_id:06d}"

        return path

    #  points to the rgb images of a scene
    def rgb_path(self, scene_id: int, img_id: int) -> Path:
        return self.scene_dir(scene_id) / "rgb" / f"{img_id:06d}.png"

    # points to the depth images of a scene
    def depth_path(self, scene_id: int, img_id: int) -> Path:
        return self.scene_dir(scene_id) / "depth" / f"{img_id:06d}.png"

    # gets a mask using the img_id and the 'instance_id', the latter refering to the order in which it appers in the corresponding scene_gt.json line
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
    mesh_samples: int = 20000

    @classmethod
    def from_root(cls, root: str | Path) -> "LMOConfig":
        return cls(paths=LMOPath.from_root(root))

    @classmethod
    def from_env(cls, env_var: str = "LMO_ROOT") -> "LMOConfig":
        return cls(paths=LMOPath.from_env(env_var))
