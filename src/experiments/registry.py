from dataclasses import dataclass
from itertools import product

import torch

from pose6d.dataset import Splits
from pose6d.layout import uid_dataset
from pose6d.model import (
    SymmetryFieldMLP1,
    SymmetryFieldMLP2,
    SymmetryFieldMLP3,
    SymmetryFieldMLP4,
    SymmetryFieldMLP5,
)

"""
Experiments for the CLI (scripts/run_experiments.py). An Experiment is a list of run
configs, each one a plain dict:
    data:  preprocessing version ({root}/versions/{version}), split name, obj_ids, scenes
    model: model class + kwargs
    train: batch size, optimizer, scheduler, early stopping, seed

"""

DATA_VERSION = "scalarfield_exp3"

# Splits are static to compare them between runs and experiments
SPLITS = {
    "lmo-strat": lambda uids: Splits.stratified(
        [u for u in uids if uid_dataset(u) == "lmo"], test_frac=0.15, val_frac=0.15
    ),
    "pbr-train_lmo-test": lambda uids: Splits.by_dataset(uids, "lmo", val_frac=0.15),
}


# slim dataclass, confiug can vary so a dict is passed
@dataclass
class Experiment:
    name: str
    configs: list[dict]


def _tag(values) -> str:
    return "-".join(str(v) for v in sorted(values))


def build_configs(
    experiment_name: str,
    split: str,
    obj_ids: set[int],
    model_specs: list[tuple[type, dict | None]],
    batch_sizes: list[int],
    optimizer_specs: list[tuple[type, dict | None, type | None, dict | None]],
    scenes: dict[str, set[int]] | None = None,
    data_version: str = DATA_VERSION,
    seed: int = 0,
    **train_kwargs,
) -> list[dict]:
    """One config per (batch size, optimizer, model). seed sets model init and shuffling."""
    scene_tag = "".join(f"_{d}-sc{_tag(s)}" for d, s in sorted((scenes or {}).items()))
    configs = []
    for batch_size, (opt_cls, opt_kwargs, sched_cls, sched_kwargs), (
        model_cls,
        model_kwargs,
    ) in product(batch_sizes, optimizer_specs, model_specs):  # cartesian product
        model_tag = model_cls.__name__.removeprefix("SymmetryField")
        if model_kwargs:
            model_tag += "-" + "-".join(
                f"{k}{v}" for k, v in sorted(model_kwargs.items())
            )
        sched_tag = sched_cls.__name__ if sched_cls else "noshd"
        configs.append(
            {
                "experiment_name": experiment_name,
                "run_name": (
                    f"obj{_tag(obj_ids)}_{split}{scene_tag}_{model_tag}"
                    f"_bs{batch_size}_{opt_cls.__name__}_{sched_tag}"
                ),
                "data": {
                    "version": data_version,
                    "split": split,
                    "obj_ids": sorted(obj_ids),
                    "scenes": scenes,
                },
                "model": {"cls": model_cls, "kwargs": model_kwargs},
                "train": {
                    "batch_size": batch_size,
                    "optimizer_cls": opt_cls,
                    "optimizer_kwargs": opt_kwargs,
                    "scheduler_cls": sched_cls,
                    "scheduler_kwargs": sched_kwargs,
                    "seed": seed,
                    **train_kwargs,
                },
            }
        )
    return configs


# --- grids
OPTIMIZER_SPECS = [
    (torch.optim.SGD, {"lr": 1e-2, "momentum": 0.9}, None, None),
    (
        torch.optim.SGD,
        {"lr": 1e-2, "momentum": 0.9},
        torch.optim.lr_scheduler.StepLR,
        {"step_size": 1000, "gamma": 0.5},
    ),
    (torch.optim.Adam, {"lr": 1e-3}, None, None),
    (
        torch.optim.Adam,
        {"lr": 1e-3},
        torch.optim.lr_scheduler.CosineAnnealingLR,
        {"T_max": 5000},
    ),
    (torch.optim.AdamW, {"lr": 1e-3, "weight_decay": 1e-2}, None, None),
    (
        torch.optim.AdamW,
        {"lr": 1e-3, "weight_decay": 1e-2},
        torch.optim.lr_scheduler.CosineAnnealingLR,
        {"T_max": 5000},
    ),
]
# Adam and AdamW only (as it have the best performance)
OPTIMIZER_SPECS_ADAM = [
    spec for spec in OPTIMIZER_SPECS if spec[0] in (torch.optim.Adam, torch.optim.AdamW)
]
MODEL_SPECS = [
    (SymmetryFieldMLP1, None),
    (SymmetryFieldMLP2, None),
    (SymmetryFieldMLP3, None),
    (SymmetryFieldMLP4, None),
    (SymmetryFieldMLP5, None),
]
MODEL_SPECS_CROSS = [  # MLP5 is the best performance, MLP1 4 for ablation
    (SymmetryFieldMLP5, {"dropout": 0.0}),
    (SymmetryFieldMLP5, {"dropout": 0.2}),
    (SymmetryFieldMLP5, {"dropout": 0.3}),
    (SymmetryFieldMLP4, None),
    (SymmetryFieldMLP1, None),
]
BATCH_SIZES_LMO = [32, 64, 128]
BATCH_SIZES_PBR = [1024, 2048]

# early stopping, in epochs: patience only counts after min_epochs
TRAIN_KWARGS = {
    "max_epochs": 3000,
    "min_epochs": 600,
    "patience": 500,
    "min_delta": 0.0005,
    "eval_every": 10,
}
# pbr epochs are much bigger (more instances): val loss plateaus by ~1000 epochs and
# later gains are ~1%, so shorter budget and a larger min_delta to ignore noise
TRAIN_KWARGS_PBR = {
    **TRAIN_KWARGS,
    "max_epochs": 1500,
    "min_epochs": 300,
    "patience": 200,
    "min_delta": 0.002,
}
# cosine T_max matched to the pbr budget so the lr actually anneals
OPTIMIZER_SPECS_ADAM_PBR = [
    (opt, opt_kw, sched, {"T_max": TRAIN_KWARGS_PBR["max_epochs"]} if sched else None)
    for opt, opt_kw, sched, _ in OPTIMIZER_SPECS_ADAM
]


def lmo_experiment(obj_ids: set[int]) -> Experiment:
    name = f"lmo-strat_obj{_tag(obj_ids)}"
    return Experiment(
        name=name,
        configs=build_configs(
            experiment_name=name,
            split="lmo-strat",
            obj_ids=obj_ids,
            model_specs=MODEL_SPECS_CROSS,
            batch_sizes=BATCH_SIZES_LMO,
            optimizer_specs=OPTIMIZER_SPECS_ADAM,
            **TRAIN_KWARGS,
        ),
    )


def pbr_lmo_experiment(obj_ids: set[int]) -> Experiment:
    name = f"pbr-train_lmo-test_obj{_tag(obj_ids)}"
    return Experiment(
        name=name,
        configs=build_configs(
            experiment_name=name,
            split="pbr-train_lmo-test",
            obj_ids=obj_ids,
            model_specs=MODEL_SPECS_CROSS,
            batch_sizes=BATCH_SIZES_PBR,
            optimizer_specs=OPTIMIZER_SPECS_ADAM_PBR,
            **TRAIN_KWARGS_PBR,
        ),
    )


_EXPERIMENTS = [
    lmo_experiment({10}),
    lmo_experiment({11}),
    lmo_experiment({10, 11}),
    # Testing server experiment
    Experiment(
        name="pbr-train_lmo-test_server-check",
        configs=build_configs(
            experiment_name="pbr-train_lmo-test_server-check",
            split="pbr-train_lmo-test",
            obj_ids={10, 11},
            scenes={"pbr": {3}, "lmo": {2}},
            model_specs=[(SymmetryFieldMLP5, {"dropout": 0.1})],
            batch_sizes=[128],
            optimizer_specs=[OPTIMIZER_SPECS[2]],
            max_epochs=50,
            min_epochs=10,
            patience=5,
            min_delta=0.0005,
            eval_every=1,
        ),
    ),
    pbr_lmo_experiment({10, 11}),
    pbr_lmo_experiment({10}),
    pbr_lmo_experiment({11}),
]
REGISTRY: dict[str, Experiment] = {e.name: e for e in _EXPERIMENTS}
