from pipelines.backprojected_features import extract_features_fm
from pose6d.data_loader import SymmetryData
from model_wrappers import DINOWrapper
from feature_extractor.config import FeatureConfig

from pytorch3d.structures import Meshes
from pytorch3d.io import IO
from typing import Literal

from pathlib import Path
import torch

# class for feature extraction.
# exposes .extract() as its main method


class MeshFeatureExtractor:
    def __init__(
        self,
        mesh_path: Path | str,
        feature_mode: Literal["distance", "backprojection"],
        num_samples: int = 8192,
    ):
        self.mode = feature_mode
        self.mesh_path = Path(mesh_path)
        self.num_samples = num_samples

        self._mesh = IO().load_mesh(str(self.mesh_path))
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._mesh = self._mesh.to(self._device)

    def extract(
        self,
        plane: SymmetryData | None = None,
        model: DINOWrapper | None = None,
        extract_settings: FeatureConfig | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "distance":
            if plane is None:
                raise ValueError("plane required for distance mode")
            return _extract_distance(plane)

        elif self.mode == "backprojection":
            return extract_features_fm(self._mesh, model, extract_settings)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")


def _extract_distance(
    plane: SymmetryData,
) -> tuple[torch.Tensor, torch.Tensor]:
    points = _sample_mesh_points(self._mesh, self.num_samples)
    field = _compute_distance_field(points, plane)
    return points, field
