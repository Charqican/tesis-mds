from dataclasses import dataclass
import random
import torch
from torch.utils.data import DataLoader, Subset
from pose6d.dataset import SymmetryFieldInstanceDataset
from pose6d.loader import LMOLoader


# This dataclass is only a formality for every experiment
@dataclass
class TrainData:
    model: torch.nn.Module
    train_loader: DataLoader
    test_features: torch.Tensor
    test_targets: torch.Tensor
    feature_dim: int
    dataset: object = None


# ------ Aux funcctions
def reserve_test_uids(
    dataset: SymmetryFieldInstanceDataset,
    loader: LMOLoader,
    sel_obj_id,
    n_per_obj: int = 2,
    seed: int = 123,
) -> set[str]:
    rng = random.Random(seed)
    uids_by_obj: dict[int, list[str]] = {}
    for uid in dataset.uid_list:
        _, _, obj_id, _ = loader.parse_instance_uid(uid)  # extract obj id
        if obj_id != sel_obj_id:
            continue
        uids_by_obj.setdefault(obj_id, []).append(
            uid
        )  # populate the dictionary with obj_id: []
    test_uids = set()
    for obj_id, uids in uids_by_obj.items():
        test_uids.update(
            rng.sample(uids, min(n_per_obj, len(uids)))
        )  # populate the set with random uids
    return test_uids


# ------ EXPERIMENTS SETUP
# Every experiment may have different data processing


### --- Experiment 1 ---
## Specifications :
# Only 1 object
# Training in BOP-lmo test dataset (scene id 2)
# Using segmentation mask to obtain partial pointcloud
# Seg + backproj -> N points -> FPS -> 1024 points (low res)
## Notes:
# This experiments uses a custom dataset (pose6d.dataset) which expect point clouds
# to be already available at a speific directory & ready for training.
# Data preoprocessing is a previus step.
def setup_exp1(
    loader,
    points_pt_dir,
    features_input_di,
    target_dir,
    model_cls,
    model_kwargs=None,
    sel_obj_id=10,
    batch_size=32,
    device=None,
) -> TrainData:
    dataset = SymmetryFieldInstanceDataset(points_pt_dir, features_input_di, target_dir)

    # we select only the requested object using a simple in statement
    test_uids = reserve_test_uids(dataset, loader, sel_obj_id, n_per_obj=5)
    obj_target = f"obj{sel_obj_id:06d}"
    train_uids = set(x for x in dataset.uid_list if obj_target in x) - test_uids
    dataset.assign_split(train_uids=train_uids, val_uids=set(), test_uids=test_uids)

    # TODO: check if this is correct
    train_idx = (dataset.split == 0).nonzero(as_tuple=True)[0].tolist()
    test_idx = (dataset.split == 2).nonzero(as_tuple=True)[0].tolist()
    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=batch_size, shuffle=True
    )

    feature_dim = dataset.features.shape[-1]
    model = model_cls(in_dim=feature_dim, **(model_kwargs or {}))

    return TrainData(
        model=model,
        train_loader=train_loader,
        test_features=dataset.features[test_idx],
        test_targets=dataset.targets[test_idx],
        feature_dim=feature_dim,
        dataset=dataset,
    )
