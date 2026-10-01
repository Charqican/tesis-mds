from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import random
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from logger import pose6d_dataset_logger as log


# WARNING: inputs loading uses 'features', so this naming convention is implicit
def load_instance_npz(
    uid: str,
    points_dir: Path,
    input_dir: Path,
    target_dir: Path,
    expected_k: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    points_path = points_dir / f"{uid}.npz"
    target_path = target_dir / f"{uid}.npz"
    input_path = input_dir / f"{uid}.npz"

    if not (points_path.exists() and target_path.exists() and input_path.exists()):
        raise FileNotFoundError(
            f"Missing files for {uid}: points={points_path.exists()}, "
            f"target={target_path.exists()}, inputs={input_path.exists()}"
        )

    points = np.load(points_path)["points"]
    inp = np.load(input_path)["features"]
    target = np.load(target_path)["target"]

    if not (inp.shape[0] == target.shape[0] == points.shape[0]):
        raise ValueError(
            f"{uid}: input - target - points missalignment:\n"
            f"- input :{inp.shape[0]}\n"
            f"- target :{target.shape[0]}\n"
            f"- points: {points.shape[0]}"
        )

    k = points.shape[0]
    if expected_k is not None and k != expected_k:
        raise ValueError(f"{uid}: has {k} points, expected {expected_k}")

    return points, inp, target, k


# One dataset's worth of file locations. uids=None discovers every uid under
# input_dir; an explicit list restricts to a subset (e.g. one scene for a
# small local trial).
@dataclass(frozen=True)
class DatasetSource:
    points_dir: Path
    input_dir: Path
    target_dir: Path
    uids: list[str] | None = None


# INFO: a lot of raise blocks as this implementation is essentialy a state machine.
class SymmetryFieldInstanceDataset(Dataset):
    _SPLIT_IDS = {"train": 0, "val": 1, "test": 2}

    @classmethod
    def from_dirs_by_uid(
        cls, dirs_by_uid: dict[str, tuple[Path, Path, Path]]
    ) -> "SymmetryFieldInstanceDataset":
        self = cls.__new__(cls)
        self._init_common(dirs_by_uid)
        return self

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

        uid_list = uids if uids is not None else all_uids
        dirs_by_uid = {uid: (points_dir, input_dir, target_dir) for uid in uid_list}
        self._init_common(dirs_by_uid)

    @classmethod
    def from_sources(
        cls, sources: list[DatasetSource]
    ) -> "SymmetryFieldInstanceDataset":
        dirs_by_uid: dict[str, tuple[Path, Path, Path]] = {}
        for source in sources:
            uids = source.uids
            if uids is None:
                uids = sorted(p.stem for p in source.input_dir.rglob("*.npz"))
                if not uids:
                    raise FileNotFoundError(f"No .npz file found in {source.input_dir}")
            for uid in uids:
                if uid in dirs_by_uid:
                    raise ValueError(f"duplicate uid across sources: {uid}")
                dirs_by_uid[uid] = (
                    source.points_dir,
                    source.input_dir,
                    source.target_dir,
                )

        self = cls.__new__(cls)
        self._init_common(dirs_by_uid)
        return self

    def _init_common(self, dirs_by_uid: dict[str, tuple[Path, Path, Path]]) -> None:
        self._dirs_by_uid = dirs_by_uid
        self.uid_list = list(dirs_by_uid.keys())
        self.split = torch.full((len(self.uid_list),), -1, dtype=torch.long)

        self._splits_assigned = False
        self._loaded = False

        self.points = None
        self.input_raw = None
        self.targets_raw = None
        self.input = None
        self.targets = None
        self.target_mean = torch.tensor(0.0)
        self.target_std = torch.tensor(1.0)
        self.input_mean = None
        self.input_std = None

    def __len__(self) -> int:
        return len(self.uid_list)

    def __getitem__(self, idx: int):
        if not self._loaded:
            raise RuntimeError("call load() first")
        return self.input[idx], self.targets[idx]

    def assign_splits_explicit(
        self,
        test_uids: set[str],
        val_uids: set[str] | None = None,
    ) -> None:
        val_uids = val_uids or set()
        for i, uid in enumerate(self.uid_list):
            if uid in test_uids:
                self.split[i] = 2
            elif uid in val_uids:
                self.split[i] = 1
            else:
                self.split[i] = 0
        self._splits_assigned = True

    def assign_splits_random(
        self,
        fractions: tuple[float, float] | tuple[float, float, float],
        seed: int = 1234,
    ) -> None:
        rng = random.Random(seed)
        uids = list(self.uid_list)
        rng.shuffle(uids)
        n = len(uids)

        if len(fractions) == 2:
            _, test_frac = fractions
            n_test = int(n * test_frac)
            test_uids = set(uids[:n_test])
            val_uids = set()
        else:
            _, val_frac, test_frac = fractions
            n_test = int(n * test_frac)
            n_val = int(n * val_frac)
            test_uids = set(uids[:n_test])
            val_uids = set(uids[n_test : n_test + n_val])

        self.assign_splits_explicit(test_uids, val_uids)

    def load(self) -> None:
        if not self._splits_assigned:
            raise RuntimeError(
                "call assign_splits_explicit or assign_splits_random before load()"
            )
        if self._loaded:
            return

        all_points, all_input, all_targets = [], [], []
        expected_k = None
        for uid in self.uid_list:
            points_dir, input_dir, target_dir = self._dirs_by_uid[uid]
            points, inp, target, expected_k = load_instance_npz(
                uid, points_dir, input_dir, target_dir, expected_k
            )
            all_points.append(points)
            all_input.append(inp)
            all_targets.append(target)

        self.points = torch.from_numpy(np.stack(all_points, axis=0)).float()
        self.input_raw = torch.from_numpy(np.stack(all_input, axis=0)).float()
        self.targets_raw = torch.from_numpy(np.stack(all_targets, axis=0)).float()
        self.input = self.input_raw
        self.targets = self.targets_raw
        self.input_mean = torch.zeros(self.input_raw.shape[-1])
        self.input_std = torch.ones(self.input_raw.shape[-1])

        self._normalize()
        self._loaded = True

    def _normalize(self) -> None:
        train_mask = self.split == 0
        if train_mask.sum() == 0:
            return

        train_targets = self.targets_raw[train_mask]
        self.target_mean = train_targets.mean()
        self.target_std = torch.clamp(train_targets.std(unbiased=False), min=1e-6)
        self.targets = (self.targets_raw - self.target_mean) / self.target_std

        train_input = self.input_raw[train_mask]
        flat = train_input.reshape(-1, train_input.shape[-1])
        self.input_mean = flat.mean(dim=0)
        self.input_std = torch.clamp(flat.std(dim=0, unbiased=False), min=1e-6)
        self.input = (self.input_raw - self.input_mean) / self.input_std

    def denormalize(self, pred_normalized: torch.Tensor) -> torch.Tensor:
        return pred_normalized * self.target_std + self.target_mean

    def denormalize_input(self, input_normalized: torch.Tensor) -> torch.Tensor:
        return input_normalized * self.input_std + self.input_mean

    def get_instance(self, uid: str):
        if not self._loaded:
            raise RuntimeError("call load() first")
        idx = self.uid_list.index(uid)
        return self.points[idx], self.input[idx], self.targets_raw[idx]

    def indices_for(self, name: str) -> list[int]:
        if name not in self._SPLIT_IDS:
            raise ValueError(f"unknown split name: {name}")
        return (self.split == self._SPLIT_IDS[name]).nonzero(as_tuple=True)[0].tolist()

    def uids_for(self, name: str) -> list[str]:
        return [self.uid_list[i] for i in self.indices_for(name)]

    def get_split(self, name: str) -> Subset:
        if not self._loaded:
            raise RuntimeError("call load() first")
        return Subset(self, self.indices_for(name))

    def uids_for_filtered(self, predicate: Callable[[str, int], bool]) -> list[str]:
        return [uid for i, uid in enumerate(self.uid_list) if predicate(uid, i)]

    def get_partition(self, predicate: Callable[[str, int], bool]) -> Subset:
        if not self._loaded:
            raise RuntimeError("call load() first")
        indices = [i for i, uid in enumerate(self.uid_list) if predicate(uid, i)]
        return Subset(self, indices)

    def make_partition(
        self, predicate: Callable[[str, int], bool]
    ) -> "SymmetryFieldInstanceDataset":
        if self._loaded:
            raise RuntimeError("dataset already loaded, use get_partition instead")
        uids = self.uids_for_filtered(predicate)
        dirs_by_uid = {uid: self._dirs_by_uid[uid] for uid in uids}
        new = SymmetryFieldInstanceDataset.__new__(SymmetryFieldInstanceDataset)
        new._init_common(dirs_by_uid)
        return new
