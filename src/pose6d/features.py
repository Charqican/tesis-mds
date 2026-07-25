from pipelines.backprojected_features import extract_features_fm
from pose6d.data_loader import SymmetryData
from model_wrappers import DINOWrapper
from feature_extractor.config import FeatureConfig

from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from pytorch3d.io import IO
from typing import Literal

from pathlib import Path
import torch

# class for feature extraction.
# exposes .extract() as its main method


# TODO : implement distance_feature_field &
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

        self._mesh: Meshes = IO().load_mesh(str(self.mesh_path))
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
            return _distance_field(plane)

        elif self.mode == "backprojection":
            return extract_features_fm(self._mesh, model, 10000, extract_settings, "")

        else:
            raise ValueError(f"Unknown mode: {self.mode}")


def _sample_mesh_points(mesh: Meshes, num_samples: int) -> torch.Tensor:
    """Samplea puntos uniformemente sobre la superficie del mesh."""
    points = sample_points_from_meshes(mesh, num_samples)  # (1, N, 3)
    return points.squeeze(0)  # (N, 3)


def _compute_distance_field(
    points: torch.Tensor,
    plane: SymmetryData,
) -> torch.Tensor:
    """Calcula distancia con signo al plano de simetría."""
    device = points.device
    normal = plane.normal.to(device)
    point = plane.plane_point.to(device)

    diff = points - point  # (N, 3)
    distances = diff @ normal  # (N,)
    return distances.unsqueeze(-1)  # (N, 1)


def _distance_field(self, plane: SymmetryData) -> tuple[torch.Tensor, torch.Tensor]:
    points = _sample_mesh_points(self._mesh, self.num_samples)
    field = _compute_distance_field(points, plane)
    return points, field
