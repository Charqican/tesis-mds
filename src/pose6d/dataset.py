from pathlib import Path
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from pose6d.loader import LMOLoader
from logger import pose6d_preprocessing_logger

"""
pytorch implementation of dataset to be passed to a dataloader to handle batching.
"""


# INFO : this implementation is eager, Lazy loading can be implementated in the future to reduce memory usage
class SymmetryFieldPointDataset(Dataset):
    """
    This Dataset samples points from the point cloud instances.
    """

    def __init__(
        self,
        points_dir: Path,
        input_dir: Path,
        target_dir: Path,
        max_instances: int | None = None,
        seed: int = 1234,
        include_all_test: bool = False,
        test_scenes: list[int] | None = None,
    ):
        uids = sorted(p.stem for p in input_dir.rglob("*.npz"))
        if not uids:
            raise FileNotFoundError(f"No .npz file found in {input_dir}")

        if max_instances is not None and max_instances < len(uids):
            rng = random.Random(seed)
            if include_all_test and test_scenes:
                test_uids_all = [
                    u
                    for u in uids
                    if LMOLoader.parse_instance_uid_(u)[0] in set(test_scenes)
                ]
                other_uids = [
                    u
                    for u in uids
                    if LMOLoader.parse_instance_uid_(u)[0] not in set(test_scenes)
                ]
                n_remaining = max(0, max_instances - len(test_uids_all))
                sampled_others = rng.sample(
                    other_uids, min(n_remaining, len(other_uids))
                )
                uids = sorted(test_uids_all + sampled_others)
                print(f"train | val : {len(other_uids)}test: {len(test_uids_all)}")
            else:
                uids = sorted(rng.sample(uids, max_instances))

        all_features, all_targets, all_points = [], [], []
        all_instance_idx = []
        self.uid_list: list[str] = []
        for i, uid in enumerate(uids):
            points_path = points_dir / f"{uid}.npz"
            target_path = target_dir / f"{uid}.npz"
            input_path = input_dir / f"{uid}.npz"
            # TODO: this requieres a user warning
            if not (
                points_path.exists() and target_path.exists() and input_path.exists()
            ):
                raise FileNotFoundError(
                    f"Missing files for {uid}: points={points_path.exists()}, target={target_path.exists()}, inputs={input_path.exists()}"
                )
            points = np.load(points_path)["points"]
            feats = np.load(input_path)["features"]
            target = np.load(target_path)["target"]
            # Sanity checks
            assert feats.shape[0] == target.shape[0] == points.shape[0], (
                f"{uid}: features - target - points missalignment:\n"
                f"- input :{feats.shape[0]}\n"
                f"- target :{target.shape[0]}\n"
                f"- points: {points.shape[0]}"
            )
            n_pts = feats.shape[0]
            all_features.append(feats)
            all_targets.append(target)
            all_points.append(points)
            all_instance_idx.append(np.full(n_pts, i))
            self.uid_list.append(uid)

        self.features = torch.from_numpy(np.concatenate(all_features, axis=0)).float()
        self.points = torch.from_numpy(np.concatenate(all_points, axis=0)).float()
        self.instance_idx = torch.from_numpy(
            np.concatenate(all_instance_idx, axis=0)
        ).long()  # this is an index
        self.targets_raw = torch.from_numpy(np.concatenate(all_targets, axis=0)).float()
        # WARNING: CHECK THIS LINE
        self.split_per_instance = torch.full(
            (len(self.uid_list),), -1, dtype=torch.long
        )
        self.split = self.split_per_instance[self.instance_idx]

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        return self.features[idx], self.targets[idx]

    def denormalize(self, pred_normalized: torch.Tensor) -> torch.Tensor:
        return pred_normalized * self.target_std + self.target_mean

    def denormalize_features(self, pred_normalized: torch.Tensor) -> torch.Tensor:
        return pred_normalized * self.feature_std + self.feature_mean

    def assign_split(
        self,
        train_uids: set[str],
        val_uids: set[str],
        test_uids: set[str],
        normalize: bool = True,
    ) -> None:
        for i, uid in enumerate(self.uid_list):
            if uid in train_uids:
                self.split_per_instance[i] = 0
            elif uid in val_uids:
                self.split_per_instance[i] = 1
            elif uid in test_uids:
                self.split_per_instance[i] = 2
        self.split = self.split_per_instance[self.instance_idx]
        train_mask = self.split == 0
        if normalize:
            self.target_mean = self.targets_raw[train_mask].mean()
            self.target_std = self.targets_raw[train_mask].std(unbiased=False)
            self.target_std = torch.clamp(self.target_std, min=1e-6)
            self.targets = (self.targets_raw - self.target_mean) / self.target_std
            # normalizing features
            self.feature_mean = self.features[train_mask].mean(dim=0)
            self.feature_std = self.features[train_mask].std(dim=0, unbiased=False)
            self.feature_std = torch.clamp(self.feature_std, min=1e-6)
            self.features = (self.features - self.feature_mean) / self.feature_std
        else:
            self.targets = self.targets_raw

    # May be possible to obtain with only the uid assigned to the point in an index vector
    def get_instance(self, uid: str):
        idx = self.uid_list.index(uid)
        mask = self.instance_idx == idx
        return self.points[mask], self.features[mask], self.targets_raw[mask]


def split_by_scene(
    dataset: SymmetryFieldPointDataset,
    test_scenes: set[int],
    val_frac: float = 0.2,
    seed: int = 123,
) -> tuple[set[str], set[str], set[str]]:
    rng = random.Random(seed)

    test_uids = set()
    remaining_uids = []
    for uid in dataset.uid_list:
        scene_id, _, _, _ = LMOLoader.parse_instance_uid_(uid)
        if scene_id in test_scenes:
            test_uids.add(uid)
        else:
            remaining_uids.append(uid)

    remaining_uids = sorted(remaining_uids)
    rng.shuffle(remaining_uids)
    n_val = int(len(remaining_uids) * val_frac)

    val_uids = set(remaining_uids[:n_val])
    train_uids = set(remaining_uids[n_val:])

    return train_uids, val_uids, test_uids


class SymmetryFieldInstanceDataset(Dataset):
    def __init__(self, points_dir: Path, input_dir: Path, target_dir: Path):
        uids = sorted(p.stem for p in input_dir.glob("*.npz"))
        if not uids:
            raise FileNotFoundError(f"Sin archivos .npz en {input_dir}")

        self.uid_list: list[str] = []
        self._features: list[torch.Tensor] = []
        self._targets_raw: list[torch.Tensor] = []
        self._points: list[torch.Tensor] = []

        for uid in uids:
            points_path = points_dir / f"{uid}.npz"
            target_path = target_dir / f"{uid}.npz"
            input_path = input_dir / f"{uid}.npz"
            if not (points_path.exists() and target_path.exists()):
                continue

            points = np.load(points_path)["points"]
            feats = np.load(input_path)["features"]
            target = np.load(target_path)["target"]

            assert target.ndim == 1, f"{uid}: target debería ser 1D"
            assert feats.shape[0] == target.shape[0] == points.shape[0], (
                f"{uid}: desalineamiento features/target/points"
            )

            self.uid_list.append(uid)
            self._features.append(torch.from_numpy(feats).float())
            self._targets_raw.append(torch.from_numpy(target).float())
            self._points.append(torch.from_numpy(points).float())

        # normalización se calcula en assign_split, sólo con train -- evita leakage
        self.target_mean: torch.Tensor | None = None
        self.target_std: torch.Tensor | None = None
        self.split_per_instance = torch.full(
            (len(self.uid_list),), -1, dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.uid_list)

    def __getitem__(self, idx: int):
        if self.target_mean is None:
            raise RuntimeError("Llamar assign_split() antes de usar el dataset")
        features = self._features[idx]
        target_raw = self._targets_raw[idx]
        target_norm = (target_raw - self.target_mean) / self.target_std
        return features, target_norm

    def denormalize(self, pred_normalized: torch.Tensor) -> torch.Tensor:
        return pred_normalized * self.target_std + self.target_mean

    def assign_split(
        self,
        train_uids: set[str],
        val_uids: set[str],
        test_uids: set[str],
        normalize: bool = True,
    ) -> None:
        for i, uid in enumerate(self.uid_list):
            if uid in train_uids:
                self.split_per_instance[i] = 0
            elif uid in val_uids:
                self.split_per_instance[i] = 1
            elif uid in test_uids:
                self.split_per_instance[i] = 2

        train_targets = torch.cat(
            [
                self._targets_raw[i]
                for i in range(len(self.uid_list))
                if self.split_per_instance[i] == 0
            ]
        )

        if normalize:
            self.target_mean = train_targets.mean()
            self.target_std = train_targets.std()
        else:
            self.target_mean = torch.tensor(0.0)
            self.target_std = torch.tensor(1.0)

    def get_instance(self, uid: str):
        idx = self.uid_list.index(uid)
        return self._points[idx], self._features[idx], self._targets_raw[idx]
