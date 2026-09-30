from dataclasses import dataclass
from pathlib import Path
import random
from typing import FrozenSet
import numpy as np
from sqlalchemy.sql.compiler import RESERVED_WORDS
import torch
from torch.utils.data import DataLoader, Subset
from pose6d.dataset import SymmetryFieldInstanceDataset
from pose6d.loader import LMOLoader, InstanceData
from pose6d.selection import (
    uids_by_visib_max,
    uids_by_visib_min,
    uids_by_visib_percentile,
)


# This dataclass is only a formality for every experiment
@dataclass
class TrainData:
    model: torch.nn.Module
    batch_size: int
    input_dim: int
    dataset: SymmetryFieldInstanceDataset


# ------ Aux funcctions
def reserve_test_uids(
    dataset: SymmetryFieldInstanceDataset,
    loader: LMOLoader,
    sel_obj_ids: frozenset[int],
    n_per_obj: int = 2,
    seed: int = 123,
) -> set[str]:
    rng = random.Random(seed)
    uids_by_obj: dict[int, list[str]] = {}
    for uid in dataset.uid_list:
        _, _, obj_id, _ = loader.parse_instance_uid(uid)
        if obj_id not in sel_obj_ids:
            continue
        uids_by_obj.setdefault(obj_id, []).append(uid)
    test_uids = set()
    for obj_id, uids in uids_by_obj.items():
        test_uids.update(rng.sample(uids, min(n_per_obj, len(uids))))
    return test_uids


# WARNING: implicit assumption "features" in .npz
def get_input_dim(input_dir: Path | str) -> int:
    input_dir = Path(input_dir)
    path = next(input_dir.rglob("*.npz"))
    return np.load(path)["features"].shape[-1]


# ------ EXPERIMENTS SETUP
# Every experiment may have different data processing


### --- Experiment 1 ---
## Specifications :
# Only 1 object
# Training in BOP-lmo test dataset (scene id 2)
# Seg + backproj -> N points -> FPS -> 1024 points (low res)
## Notes:
# This experiments uses a custom dataset (pose6d.dataset) which expect point clouds
# to be already available at a speific directory & ready for training.
# Data preoprocessing is a previus step.
def setup_exp1(
    loader: LMOLoader,
    points_pt_dir: Path,
    input_dir: Path,
    target_dir: Path,
    model_cls,
    model_kwargs: dict | None = None,
    sel_obj_ids: frozenset[int] = frozenset({10}),
    batch_size=32,
    n_per_obj=10,
) -> TrainData:
    dataset = SymmetryFieldInstanceDataset(points_pt_dir, input_dir, target_dir)

    obj_condition = lambda uid, i: LMOLoader.parse_instance_uid_(uid)[2] in sel_obj_ids
    obj_dataset: SymmetryFieldInstanceDataset = dataset.make_partition(obj_condition)

    test_uid = reserve_test_uids(obj_dataset, loader, sel_obj_ids, n_per_obj=n_per_obj)
    obj_dataset.assign_splits_explicit(test_uids=test_uid)
    input_dim = get_input_dim(input_dir)
    model = model_cls(in_dim=input_dim, **(model_kwargs or {}))

    return TrainData(
        model=model, dataset=obj_dataset, batch_size=batch_size, input_dim=input_dim
    )
