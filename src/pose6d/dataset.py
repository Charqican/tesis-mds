from pathlib import Path
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from pose6d.loader import LMOLoader
from logger import pose6d_dataset_logger as log


# TODO: Change this to use instance batching as default, lazy loading can be considered for better memory managmente and scalability.
class SymmetryFieldPointDataset(Dataset):
    """
    Loads from disk input and target pointclouds. At the moment splits are not explicitly separated in disk,
    instead every file has an unique identifier using the metadata asociated (scene id, frame id, obj id, instance id)
    so it can be determined at load, by deafault it uses the scene_id.

    This class handles preprocessing related to training, meanwhile geometrical preprocessing is done offline.

    Batches generaly have (B, K, f) dimensions:
    - B is the number of instances
    - K the number of points of every instance. We use fps subsampling to be able to structure batches efficiently
    - f the dimension of the features. In cases of training with pure points this dimension can be missing.

    """

    def __init__(
        self,
        points_dir: Path,
        input_dir: Path,
        target_dir: Path,
        uids: list[str] | None = None,
    ):
        all_uids = sorted(p.stem for p in input_dir.rglob("*.npz"))
        if not all_uids:
            raise FileNotFoundError(f"No .npz file found in {input_dir}")

        uids = uids if uids is not None else all_uids

        all_features, all_targets, all_points = [], [], []
        self.uid_list: list[str] = []

        expected_k: int | None = None
        for uid in uids:
            points_path = points_dir / f"{uid}.npz"
            target_path = target_dir / f"{uid}.npz"
            input_path = input_dir / f"{uid}.npz"
            if not (
                points_path.exists() and target_path.exists() and input_path.exists()
            ):
                raise FileNotFoundError(
                    f"Missing files for {uid}: points={points_path.exists()}, "
                    f"target={target_path.exists()}, inputs={input_path.exists()}"
                )

            points = np.load(points_path)["points"]
            feats = np.load(input_path)["features"]
            target = np.load(target_path)["target"]

            assert feats.shape[0] == target.shape[0] == points.shape[0], (
                f"{uid}: features - target - points missalignment:\n"
                f"- input :{feats.shape[0]}\n"
                f"- target :{target.shape[0]}\n"
                f"- points: {points.shape[0]}"
            )

            if expected_k is None:
                expected_k = points.shape[0]
            elif points.shape[0] != expected_k:
                raise ValueError(
                    f"{uid}: tiene {points.shape[0]} puntos, se esperaban {expected_k} "
                    f"(K debe ser fijo entre instancias -- verificar que el subsampling "
                    f"FPS haya corrido para todas)"
                )

            all_features.append(feats)
            all_targets.append(target)
            all_points.append(points)
            self.uid_list.append(uid)

        self.points = torch.from_numpy(np.stack(all_points, axis=0)).float()

        self.features_raw = torch.from_numpy(np.stack(all_features, axis=0)).float()
        self.targets_raw = torch.from_numpy(np.stack(all_targets, axis=0)).float()
        self.features = self.features_raw
        self.targets = self.targets_raw

        self.target_mean = torch.tensor(0.0)
        self.target_std = torch.tensor(1.0)
        self.feature_mean = torch.zeros(self.features_raw.shape[-1])
        self.feature_std = torch.ones(self.features_raw.shape[-1])

        self.split = torch.full((len(self.uid_list),), -1, dtype=torch.long)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        return self.features[idx], self.targets[idx]

    def denormalize(self, pred_normalized: torch.Tensor) -> torch.Tensor:
        return pred_normalized * self.target_std + self.target_mean

    def denormalize_features(self, feats_normalized: torch.Tensor) -> torch.Tensor:
        return feats_normalized * self.feature_std + self.feature_mean

    def assign_split(
        self,
        train_uids: set[str],
        val_uids: set[str],
        test_uids: set[str],
        normalize: bool = True,
    ) -> None:
        for i, uid in enumerate(self.uid_list):
            if uid in train_uids:
                self.split[i] = 0
            elif uid in val_uids:
                self.split[i] = 1
            elif uid in test_uids:
                self.split[i] = 2

        train_mask = self.split == 0

        if normalize:
            train_targets = self.targets_raw[train_mask]
            self.target_mean = train_targets.mean()
            self.target_std = train_targets.std(unbiased=False)
            self.target_std = torch.clamp(self.target_std, min=1e-6)
            self.targets = (self.targets_raw - self.target_mean) / self.target_std

            train_features = self.features_raw[train_mask]
            flat_train_features = train_features.reshape(-1, train_features.shape[-1])
            self.feature_mean = flat_train_features.mean(dim=0)
            self.feature_std = flat_train_features.std(dim=0, unbiased=False)
            self.feature_std = torch.clamp(self.feature_std, min=1e-6)
            self.features = (self.features_raw - self.feature_mean) / self.feature_std
        else:
            self.target_mean = torch.tensor(0.0)
            self.target_std = torch.tensor(1.0)
            self.feature_mean = torch.zeros(self.features_raw.shape[-1])
            self.feature_std = torch.ones(self.features_raw.shape[-1])
            self.targets = self.targets_raw
            self.features = self.features_raw

    def get_instance(self, uid: str):
        """
        (points, features, target_raw) de UNA instancia.
        Warning: `features` viene normalizada (lista para el modelo);
        `target_raw` viene SIN normalizar -- denormalizar la predicción
        del modelo con `denormalize` antes de comparar contra este target.
        """
        idx = self.uid_list.index(uid)
        return self.points[idx], self.features[idx], self.targets_raw[idx]


def select_uids(
    all_uids: list[str],
    max_instances: int | None = None,
    test_scenes: set[int] | None = None,
    seed: int = 1234,
) -> list[str]:
    """
    This functions should be use to give a subset of instances to the dataset without
    having to load them all. test_scenes
    """
    if max_instances is None or max_instances >= len(all_uids):
        return sorted(all_uids)

    rng = random.Random(seed)

    if test_scenes:
        test_uids = [
            u for u in all_uids if LMOLoader.parse_instance_uid_(u)[0] in test_scenes
        ]
        other_uids = [
            u
            for u in all_uids
            if LMOLoader.parse_instance_uid_(u)[0] not in test_scenes
        ]
        n_remaining = max(0, max_instances - len(test_uids))
        sampled_others = rng.sample(other_uids, min(n_remaining, len(other_uids)))
        return sorted(test_uids + sampled_others)

    return sorted(rng.sample(all_uids, max_instances))


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
