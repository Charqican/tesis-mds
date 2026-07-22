from dataclasses import dataclass
from pathlib import Path
import numpy as np
# Self Contained configurations & paths class. 6dpose is a separate problem using dependencies from backprojection.
# dependencies are managed separatedly by uv as a dependency group.


# Dataclass para definir y acceder programaticamente a los paths. Una forma simlpe de implementar funciones auxiliares para resolver paths
@dataclass(frozen=True)
class LMOPath:
    root: Path

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def models_info(self) -> Path:
        return self.models_dir / "models_info.json"

    def scene_dir(self, scene_id: int) -> Path:
        return self.root / "test" / f"{scene_id:06d}"

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

    @classmethod
    def from_env(cls, env_var: str = "LMO_ROOT") -> "LMOPath": ...


@dataclass(frozen=True)
class LMOConfig:
    """Parámetros del dataset y del pipeline."""

    paths: LMOPath
    default_scene: int = 2
    depth_stride: int = 2
    mesh_samples: int = 20000

    @classmethod
    def from_root(cls, root: str | Path) -> "LMOConfig": ...
