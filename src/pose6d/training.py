from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from pose6d.config import LMOConfig
from pose6d.data_loader import LMOLoader
from pose6d.features import compute_canonical_symmetry_field
from pose6d.geometry_utils import propagate_symmetry_to_target


# TODO: Necesitamos soportar Batching para futuros datasets o parsear para utilizar dataloaders


class SymmetryPointDataset(Dataset):
    def __init__(
        self,
        config: LMOConfig,
        loader: LMOLoader,
        points_pT_dir: Path,
        features_dgedi_dir: Path,
    ):
        # we only need to calculate 2 objects, so we're going to only calculate once at the start
        canonical_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        # features extracted from every instance (~330 instances) using dGedi
        all_features = []
        # features from target (extracted & propagated)
        all_targets = []
        # we save access to every pT saved in memory, only for analysis (few mb)
        all_points = []
        # id of every object related to each instance
        all_uids = []
        self.uid_list = []  # which ones exists!

        for npz_path in sorted(points_pT_dir.glob("*.npz")):
            uid = npz_path.stem
            pt_data = np.load(npz_path)
            obj_id = int(pt_data["obj_id"])

            feat_path = features_dgedi_dir / npz_path.name
            if not feat_path.exists():
                continue
            feats = np.load(feat_path)["features"]

            if obj_id not in canonical_cache:
                canonical_cache[obj_id] = compute_canonical_symmetry_field(
                    config, loader, obj_id
                )
            mesh_points, symmetry_scalar = canonical_cache[obj_id]

            target = propagate_symmetry_to_target(
                mesh_points,
                symmetry_scalar,
                pt_data["points"],
                pt_data["R"],
                pt_data["t"],
            )
            assert target.ndim == 1, (
                f"target debería ser 1D, llegó con shape {target.shape}"
            )
            assert feats.shape[0] == target.shape[0] == pt_data["points"].shape[0], (
                f"{npz_path.name}: desalineamiento features/target/points"
            )
            n_pts = feats.shape[0]
            all_features.append(feats)
            all_targets.append(target)
            all_points.append(pt_data["points"])
            all_uids.append(np.full(n_pts, len(self.uid_list)))
            self.uid_list.append(uid)

        self.features = torch.from_numpy(np.concatenate(all_features, axis=0)).float()
        self.points = torch.from_numpy(np.concatenate(all_points, axis=0)).float()
        self.instance_idx = torch.from_numpy(np.concatenate(all_uids, axis=0)).long()
        targets_raw = torch.from_numpy(np.concatenate(all_targets, axis=0)).float()

        # Saved for normalization
        self.target_mean = targets_raw.mean()
        self.target_std = targets_raw.std()
        self.targets = (targets_raw - self.target_mean) / self.target_std

        self.targets_raw = targets_raw

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        return self.features[idx], self.targets[idx]

    def denormalize(self, pred_normalized: torch.Tensor) -> torch.Tensor:
        return pred_normalized * self.target_std + self.target_mean

    def get_instance(self, uid: str):
        idx = self.uid_list.index(uid)
        mask = self.instance_idx == idx
        return self.points[mask], self.features[mask], self.targets_raw[mask]
