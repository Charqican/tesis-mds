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
Training data of a preprocessing version ({root}/versions/{version}/{input,target}).
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


def _representative_block(img_ids, visib_fract, n_frames, free) -> np.ndarray:
    """Indices of the n_frames consecutive free frames (img_ids sorted) whose visib_fract mean
    and p10 are closest to those of all the frames."""
    target = np.array([visib_fract.mean(), np.percentile(visib_fract, 10)])
    best, best_score = None, np.inf
    for start in range(len(img_ids) - n_frames + 1):
        window = np.arange(start, start + n_frames)
        if not free[window].all() or img_ids[window[-1]] - img_ids[start] != n_frames - 1:
            continue
        stats = np.array([visib_fract[window].mean(), np.percentile(visib_fract[window], 10)])
        score = np.abs(stats - target).sum()
        if score < best_score:
            best, best_score = window, score
    return best


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

    # n_test test uids per object, the rest split into train/val. Objects with uids in
    # test_pool take their test from it (the pool uids left over are dropped, so an old test
    # uid never ends up in train). The others take n_test random uids, or with block=True
    # n_test consecutive frames of one scene: in a video the neighbours of a random frame
    # are almost the same view. Each object is shuffled on its own, like stratified.
    @classmethod
    def per_object(
        cls,
        uids: list[str],
        n_test: int,
        val_frac: float,
        test_pool: set[str] | None = None,
        block: bool = False,
        seed: int = 0,
    ) -> "Splits":
        by_obj: dict[tuple[str, int], list[str]] = {}
        for uid in sorted(uids):
            dataset, _, _, obj_id, _ = BOPLoader.parse_instance_uid(uid)
            by_obj.setdefault((dataset, obj_id), []).append(uid)

        splits = cls()
        for (_, obj_id), obj_uids in sorted(by_obj.items()):
            rng = random.Random(seed * 1000 + obj_id)
            pool = sorted(set(obj_uids) & (test_pool or set()))
            if pool:
                test = rng.sample(pool, n_test)
                rest = sorted(set(obj_uids) - set(pool))
            elif block:
                scene_of = [BOPLoader.parse_instance_uid(u)[1] for u in obj_uids]
                starts = [
                    i
                    for i in range(len(obj_uids) - n_test + 1)
                    if scene_of[i] == scene_of[i + n_test - 1]
                ]
                start = rng.choice(starts)
                test = obj_uids[start : start + n_test]
                rest = obj_uids[:start] + obj_uids[start + n_test :]
            else:
                test = rng.sample(obj_uids, n_test)
                rest = sorted(set(obj_uids) - set(test))
            rng.shuffle(rest)
            n_val = int(len(rest) * val_frac)
            splits.test += test
            splits.val += rest[:n_val]
            splits.train += rest[n_val:]
        return cls(*(sorted(s) for s in (splits.train, splits.val, splits.test)))

    # Same train/val/test sizes for every object, the smallest one fixes them: n_test test,
    # val_frac of the rest val, the rest train. lmo test and val are random. Other datasets
    # (videos) take test, then val, as the block of consecutive frames whose visib_fract mean and
    # p10 are closest to those of all the frames, with `gap` frames around each block left out.
    # Train is evenly spaced in frame order among the rest (the camera path is kept), the rest
    # is unused. uids: one scene per object. Written in notebooks/per_object_models.py (5.6),
    # the random draws are the same as there (pandas sample with random_state=seed).
    @classmethod
    def per_object_base(
        cls,
        uids: list[str],
        visib_fract: dict[str, float],
        n_test: int,
        val_frac: float,
        gap: int,
        seed: int = 0,
        extended: bool = False,
    ) -> "Splits":
        """extended: same val/test, train = every remaining uid instead of n_train evenly spaced."""
        by_obj: dict[tuple[str, int], list[str]] = {}
        for uid in sorted(uids):
            dataset, _, _, obj_id, _ = BOPLoader.parse_instance_uid(uid)
            by_obj.setdefault((dataset, obj_id), []).append(uid)
        n_rest = min(map(len, by_obj.values())) - n_test
        n_val = round(val_frac * n_rest)
        n_train = n_rest - n_val

        test_rng = np.random.RandomState(seed)  # one for every object, in this order
        splits = cls()
        for (dataset, _), obj_uids in sorted(by_obj.items()):
            img_ids = np.array([BOPLoader.parse_instance_uid(u)[2] for u in obj_uids])
            visib = np.array([visib_fract[u] for u in obj_uids])
            part = np.full(len(obj_uids), "train", dtype=object)
            if dataset == "lmo":
                part[test_rng.permutation(len(obj_uids))[:n_test]] = "test"
                rest = np.flatnonzero(part == "train")
                part[rest[np.random.RandomState(seed).permutation(len(rest))[:n_val]]] = "val"
            else:
                for name, n_frames in [("test", n_test), ("val", n_val)]:
                    block = _representative_block(img_ids, visib, n_frames, part == "train")
                    near = (img_ids >= img_ids[block[0]] - gap) & (img_ids <= img_ids[block[-1]] + gap)
                    part[near & (part == "train")] = "gap"
                    part[block] = name
            rest = np.flatnonzero(part == "train")
            kept = rest if extended else rest[np.round(np.linspace(0, len(rest) - 1, n_train)).astype(int)]
            splits.train += [obj_uids[i] for i in kept]
            splits.val += [u for u, p in zip(obj_uids, part) if p == "val"]
            splits.test += [u for u, p in zip(obj_uids, part) if p == "test"]
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
    uids of a version, optionally filtered by object and {dataset: scenes}.
    Nothing is loaded until a split is assigned; train/val/test fail before that.
    A given normalizer (e.g. a trained model's) is used as is, otherwise one is
    fitted on the train split every time splits are assigned.
    """

    def __init__(
        self,
        layout: DataLayout,
        version: str,
        obj_ids: set[int] | None = None,
        scenes: dict[str, set[int]] | None = None,
        normalizer: Normalizer | None = None,
    ):
        self.input_dir = layout.input_dir(version)
        self.target_dir = layout.target_dir(version)
        # uids with both target and input: a version can have targets whose dgedi input was
        # never computed (e.g. objects dropped from the dgedi work list)
        input_uids = {path.stem for path in self.input_dir.glob("*.npz")}
        self.uids = []
        for path in sorted(self.target_dir.glob("*.npz")):
            if path.stem not in input_uids:
                continue
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

    # target * scale is in mm. Targets saved before normalization have no scale (mm already)
    def scales(self, uids: list[str]) -> np.ndarray:
        scales = []
        for uid in uids:
            data = np.load(self.target_dir / f"{uid}.npz")
            scales.append(float(data["scale"]) if "scale" in data else 1.0)
        return np.array(scales)

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
