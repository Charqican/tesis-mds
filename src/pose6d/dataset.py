import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from pose6d.layout import DataLayout, uid_dataset
from pose6d.loader import BOPLoader
from logger import pose6d_dataset_logger as log

"""
Training data of an experiment ({root}/experiments/{experiment}/{input,target}).
SymmetryFieldDataset is the entry point: it lists the uids on creation and loads the
data when a split is assigned, fitting the normalizer on train only.

    # training
    ds = SymmetryFieldDataset(layout, "scalarfield_exp3", obj_ids={10, 11})
    ds.split_by_dataset("lmo", val_frac=0.15)         # or split_random / assign_splits
    DataLoader(ds.train, batch_size=1024, shuffle=True)
    ds.splits.to_dict(), ds.normalizer.to_dict()      # save them with the model

    # analysis of a trained model on partial local data
    ds = SymmetryFieldDataset(layout, "scalarfield_exp3", normalizer=Normalizer.from_dict(saved))
    ds.assign_splits(Splits.from_dict(saved_splits), skip_missing=True)
    x, y = ds.test.get(uid)
"""
SPLIT_NAMES = ("train", "val", "test")


@dataclass
class Splits:
    train: list[str] = field(default_factory=list)
    val: list[str] = field(default_factory=list)
    test: list[str] = field(default_factory=list)

    def __post_init__(self):
        sets = [set(self.train), set(self.val), set(self.test)]
        if sum(map(len, sets)) != len(set.union(*sets)):
            raise ValueError("train/val/test splits overlap")

    @property
    def is_empty(self) -> bool:
        return not (self.train or self.val or self.test)

    @staticmethod
    def _shuffle_split(
        uids: list[str], frac: float, seed: int
    ) -> tuple[list[str], list[str]]:
        shuffled = sorted(uids)
        random.Random(seed).shuffle(shuffled)
        n = int(len(shuffled) * frac)
        return sorted(shuffled[n:]), sorted(shuffled[:n])

    @classmethod
    def random(
        cls, uids: list[str], val_frac: float, test_frac: float, seed: int = 123
    ) -> "Splits":
        rest, test = cls._shuffle_split(uids, test_frac, seed)
        train, val = cls._shuffle_split(rest, val_frac / (1 - test_frac), seed)
        return cls(train=train, val=val, test=test)

    # e.g. train on pbr, test on every lmo instance
    @classmethod
    def by_dataset(
        cls, uids: list[str], test_dataset: str, val_frac: float, seed: int = 123
    ) -> "Splits":
        test = [u for u in uids if uid_dataset(u) == test_dataset]
        rest = [u for u in uids if uid_dataset(u) != test_dataset]
        train, val = cls._shuffle_split(rest, val_frac, seed)
        return cls(train=train, val=val, test=test)

    # same number of test uids for every object (test_frac of the smallest one).
    # Each object is shuffled on its own, so filtering by objects keeps the split nested:
    # the obj 10 split is the obj 10 part of the obj 10+11 split.
    @classmethod
    def stratified(
        cls, uids: list[str], test_frac: float, val_frac: float, seed: int = 123
    ) -> "Splits":
        by_obj: dict[int, list[str]] = {}
        for uid in sorted(uids):
            by_obj.setdefault(BOPLoader.parse_instance_uid(uid)[3], []).append(uid)
        n_test = int(min(map(len, by_obj.values())) * test_frac)

        splits = cls()
        for obj_id, obj_uids in sorted(by_obj.items()):
            random.Random(seed * 1000 + obj_id).shuffle(obj_uids)
            n_val = int(len(obj_uids) * val_frac)
            splits.test += obj_uids[:n_test]
            splits.val += obj_uids[n_test : n_test + n_val]
            splits.train += obj_uids[n_test + n_val :]
        return cls(*(sorted(s) for s in (splits.train, splits.val, splits.test)))

    def subset(self, uids: list[str]) -> "Splits":
        keep = set(uids)
        return Splits(
            **{n: [u for u in getattr(self, n) if u in keep] for n in SPLIT_NAMES}
        )

    def to_dict(self) -> dict[str, list[str]]:
        return {"train": self.train, "val": self.val, "test": self.test}

    @classmethod
    def from_dict(cls, d: dict[str, list[str]]) -> "Splits":
        return cls(train=d["train"], val=d.get("val", []), test=d["test"])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path: Path) -> "Splits":
        return cls.from_dict(json.loads(path.read_text()))


class Normalizer:
    """Per channel input and scalar target standardization. Unfitted until fit()."""

    def __init__(self):
        self.input_mean: torch.Tensor | None = None  # (D,)
        self.input_std: torch.Tensor | None = None
        self.target_mean: float | None = None
        self.target_std: float | None = None

    @property
    def is_fitted(self) -> bool:
        return self.input_mean is not None

    def fit(self, inputs: torch.Tensor, targets: torch.Tensor) -> "Normalizer":
        flat = inputs.reshape(-1, inputs.shape[-1])
        self.input_mean = flat.mean(dim=0)
        self.input_std = flat.std(dim=0, unbiased=False).clamp(min=1e-6)
        self.target_mean = targets.mean().item()
        self.target_std = max(targets.std(unbiased=False).item(), 1e-6)
        return self

    def inputs(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.input_mean) / self.input_std

    def targets(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.target_mean) / self.target_std

    def denormalize_targets(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.target_std + self.target_mean

    def to_dict(self) -> dict:
        return {
            "input_mean": self.input_mean.tolist(),
            "input_std": self.input_std.tolist(),
            "target_mean": self.target_mean,
            "target_std": self.target_std,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizer":
        norm = cls()
        norm.input_mean = torch.tensor(d["input_mean"])
        norm.input_std = torch.tensor(d["input_std"])
        norm.target_mean = float(d["target_mean"])
        norm.target_std = float(d["target_std"])
        return norm


class SplitData(Dataset):
    """Normalized (input, target) pairs of one split, what DataLoader iterates."""

    def __init__(self, uids: list[str], inputs: torch.Tensor, targets: torch.Tensor):
        self.uids = uids
        self.inputs = inputs  # (N, K, D)
        self.targets = targets  # (N, K)
        self._index = {u: i for i, u in enumerate(uids)}

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]

    def get(self, uid: str) -> tuple[torch.Tensor, torch.Tensor]:
        return self[self._index[uid]]


class SymmetryFieldDataset:
    """
    uids of an experiment, optionally filtered by object and {dataset: scenes}.
    Nothing is loaded until a split is assigned; train/val/test fail before that.
    A given normalizer (e.g. a trained model's) is used as is, otherwise one is
    fitted on the train split every time splits are assigned.
    """

    def __init__(
        self,
        layout: DataLayout,
        experiment: str,
        obj_ids: set[int] | None = None,
        scenes: dict[str, set[int]] | None = None,
        normalizer: Normalizer | None = None,
    ):
        self.input_dir = layout.input_dir(experiment)
        self.target_dir = layout.target_dir(experiment)
        self.uids = []
        for path in sorted(self.target_dir.glob("*.npz")):
            dataset, scene_id, _, obj_id, _ = BOPLoader.parse_instance_uid(path.stem)
            if obj_ids is not None and obj_id not in obj_ids:
                continue
            if scenes is not None and scene_id not in scenes.get(dataset, ()):
                continue
            self.uids.append(path.stem)

        self._fit_normalizer = normalizer is None
        self.normalizer = normalizer or Normalizer()
        self.splits = Splits()
        self._data: dict[str, SplitData] = {}

    # --- split assignment (loads the data)
    def split_random(self, val_frac: float, test_frac: float, seed: int = 123) -> None:
        self.assign_splits(Splits.random(self.uids, val_frac, test_frac, seed))

    def split_by_dataset(
        self, test_dataset: str, val_frac: float, seed: int = 123
    ) -> None:
        self.assign_splits(Splits.by_dataset(self.uids, test_dataset, val_frac, seed))

    def assign_splits(self, splits: Splits, skip_missing: bool = False) -> None:
        """skip_missing drops uids not on disk (e.g. a server split on partial local data)."""
        if skip_missing:
            splits = Splits(
                **{n: self._present(getattr(splits, n)) for n in SPLIT_NAMES}
            )
        if splits.is_empty:
            raise ValueError("splits are empty")

        raw = {
            n: self._load(getattr(splits, n)) for n in SPLIT_NAMES if getattr(splits, n)
        }
        if self._fit_normalizer:
            if "train" not in raw:
                raise ValueError(
                    "no train split to fit the normalizer, pass one instead"
                )
            self.normalizer = Normalizer().fit(*raw["train"])

        self.splits = splits
        self._data = {
            n: SplitData(
                getattr(splits, n),
                self.normalizer.inputs(x),
                self.normalizer.targets(y),
            )
            for n, (x, y) in raw.items()
        }

    # --- access
    @property
    def train(self) -> SplitData:
        return self._split("train")

    @property
    def val(self) -> SplitData:
        return self._split("val")

    @property
    def test(self) -> SplitData:
        return self._split("test")

    def _split(self, name: str) -> SplitData:
        if self.splits.is_empty:
            raise RuntimeError(
                "no split assigned: call split_random, split_by_dataset or assign_splits"
            )
        if name not in self._data:
            raise KeyError(f"split '{name}' is empty")
        return self._data[name]

    # --- io
    # WARNING: inputs are saved under 'features' and targets under 'target'
    def _load(self, uids: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = [], []
        for uid in uids:
            inp = np.load(self.input_dir / f"{uid}.npz")["features"]
            target = np.load(self.target_dir / f"{uid}.npz")["target"]
            if inp.shape[0] != target.shape[0]:
                raise ValueError(
                    f"{uid}: {inp.shape[0]} input points vs {target.shape[0]} target points"
                )
            inputs.append(inp)
            targets.append(target)
        return (
            torch.from_numpy(np.stack(inputs)).float(),
            torch.from_numpy(np.stack(targets)).float(),
        )

    def _present(self, uids: list[str]) -> list[str]:
        present = [
            u
            for u in uids
            if (self.input_dir / f"{u}.npz").exists()
            and (self.target_dir / f"{u}.npz").exists()
        ]
        if len(present) < len(uids):
            log.warning(
                f"{len(uids) - len(present)} of {len(uids)} uids not on disk, skipped"
            )
        return present
