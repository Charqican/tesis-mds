import torch

from experiments.registry import SPLITS
from pose6d.dataset import Splits, SymmetryFieldDataset
from pose6d.layout import CONFIGS_DIR, DataLayout
from logger import experiments_logger as log

SPLITS_DIR = CONFIGS_DIR / "splits"


def load_splits(layout: DataLayout, experiment: str, name: str) -> Splits:
    """Saved split of a data experiment, made from all its uids the first time."""
    path = SPLITS_DIR / experiment / f"{name}.json"
    if path.exists():
        return Splits.load(path)

    splits = SPLITS[name](SymmetryFieldDataset(layout, experiment).uids)
    if not (splits.train and splits.val and splits.test):
        raise ValueError(f"split {name!r} has an empty part, not saving it")
    splits.save(path)
    log.info(f"Created split file {path}")
    return splits


def setup(
    layout: DataLayout, data: dict, model: dict, seed: int
) -> tuple[SymmetryFieldDataset, torch.nn.Module]:
    """Loads the config's data (split filtered by objects/scenes) and builds the model."""
    torch.manual_seed(seed)
    scenes = data["scenes"]
    ds = SymmetryFieldDataset(
        layout,
        data["experiment"],
        obj_ids=set(data["obj_ids"]),
        scenes={d: set(s) for d, s in scenes.items()} if scenes else None,
    )
    splits = load_splits(layout, data["experiment"], data["split"])
    ds.assign_splits(splits.subset(ds.uids))

    in_dim = ds.train.inputs.shape[-1]
    return ds, model["cls"](in_dim, **(model["kwargs"] or {}))
