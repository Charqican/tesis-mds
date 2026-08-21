from pipelines.backprojected_features import extract_features_fm
from pose6d.loader import SymmetryData
from pose6d.config import LMOConfig, LMOPath
from pose6d.loader import LMOLoader
from model_wrappers import DINOWrapper
from feature_extractor.config import FeatureConfig

from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from pytorch3d.io import IO
from typing import Literal

from pathlib import Path
import numpy as np
import torch

# TODO: this works but implementation is all over the place


# WARNING: This was used mainly for DINOv2 features in the registration mockup, a refactor is planned as it is used in the pT extraction script at the propagate stage


class MeshFeatureExtractor:
    """Feature extractor class, exposes .extract() as its main entrypoint"""

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
        extract_settings: FeatureConfig | None = FeatureConfig(
            batch_size=1, max_points=10000
        ),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "distance":
            if plane is None:
                raise ValueError("plane required for distance mode")
            return self._distance_field(plane)

        elif self.mode == "backprojection":
            return extract_features_fm(
                self._mesh, model, self.num_samples, extract_settings, ""
            )

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _distance_field(self, plane: SymmetryData) -> tuple[torch.Tensor, torch.Tensor]:

        points = self._sample_mesh_points(self._mesh, self.num_samples)
        field = self._compute_distance_field(points, plane)
        return points, field

    # Sample points using pytorch3d utilities
    def _sample_mesh_points(self, mesh: Meshes, num_samples: int) -> torch.Tensor:
        """Samplea puntos uniformemente sobre la superficie del mesh."""
        points = sample_points_from_meshes(mesh, num_samples)  # (1, N, 3)
        return points.squeeze(0)  # (N, 3)

    # INFO: this function uses abs value!
    def _compute_distance_field(
        self,
        points: torch.Tensor,
        plane: SymmetryData,
    ) -> torch.Tensor:
        """Helper function to compute distance field from the symmety plane using the internal SymmetryData dataclass"""
        device = points.device
        normal = plane.normal.to(device)
        point = plane.plane_point.to(device)

        diff = points - point  # (N, 3)
        distances = diff @ normal  # (N,)
        return distances.unsqueeze(-1).abs()  # (N, 1)


# function used by the pT extractor script. It directly uses the MeshFeatureExtractor
# WARNING: this function is coupled to the LMODataset, but can be easily be decoupled in the future
def compute_canonical_symmetry_field(
    config: LMOConfig, loader: LMOLoader, obj_id: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Helper function that calls MeshFeatureExtractor from this module.

    params: (config: LMOConfig, loader. LMOLoader, obj_id: int)
    Returns (mesh_points, symmetry_scalar).
    """
    symmetry_data = loader.load_symmetry_plane(obj_id)
    if symmetry_data is None:
        raise ValueError(f"No symmetry plane metadata for obj_id={obj_id}")

    mesh_path = config.paths.models_dir / f"obj_{obj_id:06d}.ply"
    extractor = MeshFeatureExtractor(mesh_path, "distance", config.mesh_samples)
    mesh_points, symmetry_scalar = extractor.extract(plane=symmetry_data)

    mesh_points = (
        mesh_points.cpu().numpy() if torch.is_tensor(mesh_points) else mesh_points
    )
    symmetry_scalar = (
        symmetry_scalar.cpu().numpy()
        if torch.is_tensor(symmetry_scalar)
        else symmetry_scalar
    )
    return mesh_points, symmetry_scalar


# DEPRECATED: this function is no longer in use after changes in the file system format
def cache_canonical_symmetry_field(
    config: LMOConfig, loader: LMOLoader, obj_id: int, out_dir: Path
) -> Path:
    mesh_points, symmetry_scalar = compute_canonical_symmetry_field(
        config, loader, obj_id
    )
    out_path = out_dir / f"obj_{obj_id:06d}.npz"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        mesh_points=mesh_points.astype(np.float32),
        symmetry_scalar=symmetry_scalar.astype(np.float32),
    )
    return out_path
