import math
from dataclasses import dataclass
from itertools import product

import torch

from experiments.analysis import uid_info
from pose6d.dataset import Splits, SymmetryFieldDataset
from pose6d.loader import BOPLoader
from pose6d.layout import CONFIGS_DIR, DataLayout, uid_dataset
from pose6d.model import SymmetryFieldMLP1, SymmetryFieldMLP4, SymmetryFieldMLP5

"""
Experiments for the CLI (scripts/run_experiments.py). An Experiment is a list of run
configs, each one a plain dict:
    data:  preprocessing version ({root}/versions/{version}), split name, obj_ids, scenes
    model: model class + kwargs
    train: batch size, optimizer, scheduler, early stopping, seed

"""

DATA_VERSION = "scalarfield_exp3"
DATA_VERSION_NORM = "scalarfield_exp3_norm"  # target / pT diameter
DATA_VERSION_NORMDIAM = "scalarfield_exp3_normdiam"  # target / object (mesh) diameter
# experiment name suffix of each version
VERSION_SUFFIX = {DATA_VERSION_NORM: "_norm", DATA_VERSION_NORMDIAM: "_normdiam"}

# test of lmo-strat in scalarfield_exp3: the 206 lmo test uids (205 of them, the ones with a
# sphere pT, are in lmo-strat_test_pT_sphere12_cadnorm.zip, used to compare other methods)
LMO_STRAT_TEST = CONFIGS_DIR / "splits" / "scalarfield_exp3" / "lmo-strat.json"

# base split: one scene per object (master chef can its most occluded one), visib >= 0.6
BASE_SCENES = {("lmo", 10): 2, ("lmo", 11): 2, ("ycbv", 1): 48, ("ycbv", 16): 55}
BASE_MIN_VISIB = 0.6


# built on the uids of PER_OBJECT_VERSION and reduced to those of the version: the sizes are fixed
# by glue, also in versions without it (wood block diagonal)
def per_object_base_split(uids: list[str], extended: bool = False) -> Splits:
    pool = []
    for u in SymmetryFieldDataset(DataLayout.from_env(), PER_OBJECT_VERSION).uids:
        dataset, scene_id, _, obj_id, _ = BOPLoader.parse_instance_uid(u)
        if BASE_SCENES.get((dataset, obj_id)) == scene_id:
            pool.append(u)
    visib = dict(zip(pool, uid_info(pool)["visib_fract"]))
    usable = [u for u in pool if visib[u] >= BASE_MIN_VISIB]
    return Splits.per_object_base(
        usable, visib, n_test=100, val_frac=0.15, gap=5, seed=0, extended=extended
    ).subset(uids)


# Splits are static to compare them between runs and experiments
SPLITS = {
    "lmo-strat": lambda uids: Splits.stratified(
        [u for u in uids if uid_dataset(u) == "lmo"], test_frac=0.15, val_frac=0.15
    ),
    "pbr-train_lmo-test": lambda uids: Splits.by_dataset(uids, "lmo", val_frac=0.15),
    # per-object models: 100 test uids per object (lmo ones from LMO_STRAT_TEST), seed 0
    "per-object-test100": lambda uids: Splits.per_object(
        uids, n_test=100, val_frac=0.15, test_pool=set(Splits.load(LMO_STRAT_TEST).test)
    ),
    # same, but the ycbv test is 100 consecutive frames instead of 100 random ones
    "per-object-test100-block": lambda uids: Splits.per_object(
        uids, n_test=100, val_frac=0.15, test_pool=set(Splits.load(LMO_STRAT_TEST).test), block=True
    ),
    # "per-object-visib06" has no builder: written by notebooks/per_object_models.py (section 6.5),
    # visib >= 0.6, 100 test per object, random for lmo and a block of frames for ycbv
    # same sizes for the 4 objects, ycbv test and val as blocks (notebooks/per_object_models.py 6.6)
    "per-object-base": per_object_base_split,
    # same val/test, train = all the remaining uids of the scene (visib >= 0.6, outside the gaps)
    "per-object-base-extended": lambda uids: per_object_base_split(uids, extended=True),
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
    seeds: tuple[int, ...] = (0,),
    **train_kwargs,
) -> list[dict]:
    """One config per (batch size, optimizer, model, seed). The seed sets model init and shuffling;
    seeds other than 0 add `_seed{s}` to the run name (seed 0 keeps the old names)."""
    scene_tag = "".join(f"_{d}-sc{_tag(s)}" for d, s in sorted((scenes or {}).items()))
    configs = []
    for batch_size, (opt_cls, opt_kwargs, sched_cls, sched_kwargs), (
        model_cls,
        model_kwargs,
    ), seed in product(batch_sizes, optimizer_specs, model_specs, seeds):  # cartesian product
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
                    + (f"_seed{seed}" if seed else "")
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
# On pbr optimizer, scheduler and batch size move the rmse ~0.1-0.2 mm (noise), the
# model ~1.5 mm: one optimizer setting and the two best models of the full grid
MODEL_SPECS_SHORTLIST = [
    (SymmetryFieldMLP5, {"dropout": 0.3}),
    (SymmetryFieldMLP1, None),
]
OPTIMIZER_SPECS_SHORTLIST_PBR = [
    spec
    for spec in OPTIMIZER_SPECS_ADAM_PBR
    if spec[0] is torch.optim.AdamW and spec[2] is None
]


# same run names in both versions, so norm and raw runs can be paired by run_name
def lmo_experiment(obj_ids: set[int], data_version: str = DATA_VERSION) -> Experiment:
    name = f"lmo-strat_obj{_tag(obj_ids)}" + VERSION_SUFFIX.get(data_version, "")
    return Experiment(
        name=name,
        configs=build_configs(
            experiment_name=name,
            split="lmo-strat",
            obj_ids=obj_ids,
            model_specs=MODEL_SPECS_CROSS,
            batch_sizes=BATCH_SIZES_LMO,
            optimizer_specs=OPTIMIZER_SPECS_ADAM,
            data_version=data_version,
            **TRAIN_KWARGS,
        ),
    )


def pbr_lmo_experiment(
    obj_ids: set[int],
    data_version: str = DATA_VERSION,
    model_specs=MODEL_SPECS_CROSS,
    batch_sizes=BATCH_SIZES_PBR,
    optimizer_specs=OPTIMIZER_SPECS_ADAM_PBR,
) -> Experiment:
    name = f"pbr-train_lmo-test_obj{_tag(obj_ids)}" + VERSION_SUFFIX.get(data_version, "")
    return Experiment(
        name=name,
        configs=build_configs(
            experiment_name=name,
            split="pbr-train_lmo-test",
            obj_ids=obj_ids,
            model_specs=model_specs,
            batch_sizes=batch_sizes,
            optimizer_specs=optimizer_specs,
            data_version=data_version,
            **TRAIN_KWARGS_PBR,
        ),
    )


# Per-object models (notebooks/per_object_models.py): best symmetry plane of each object,
# MLP5 dropout 0.2 bs 32 with the best optimizers of lmo-strat (obj10 Adam, obj11 AdamW)
PER_OBJECT_VERSION = "scalarfield_radiusremoval_cadnormalization_bestsymmplanes"
PER_OBJECT_VERSION_WOODDIAG = PER_OBJECT_VERSION + "_wooddiag"  # wood block, diagonal plane
OPTIMIZER_SPECS_PER_OBJECT = [
    spec for spec in OPTIMIZER_SPECS if spec[0] in (torch.optim.Adam, torch.optim.AdamW) and spec[2] is None
]


# full grid for the final split: every optimizer/scheduler, MLP5 with and without dropout, MLP1.
# bs fixed to 32: in lmo-strat bs 32/64/128 moved the test rmse <= 0.05 mm (std within a bs ~0.2)
# Schedulers matched to the epoch budget (they step per epoch): the old grid (T_max 5000, step
# 1000) barely moved the lr before early stopping (per-object runs stopped at ~1100-2600 epochs).
# Cosine anneals over max_epochs (it is periodic, a shorter T_max would raise the lr again),
# StepLR halves the lr every 500 epochs.
OPTIMIZER_SPECS_PER_OBJECT_GRID = [
    (
        opt,
        opt_kw,
        sched,
        {"T_max": TRAIN_KWARGS["max_epochs"]} if sched is torch.optim.lr_scheduler.CosineAnnealingLR
        else {"step_size": 500, "gamma": 0.5} if sched is torch.optim.lr_scheduler.StepLR
        else None,
    )
    for opt, opt_kw, sched, _ in OPTIMIZER_SPECS
]
MODEL_SPECS_PER_OBJECT = [
    (SymmetryFieldMLP5, {"dropout": 0.2}),
    (SymmetryFieldMLP5, {"dropout": 0.0}),
    (SymmetryFieldMLP1, None),
]


def per_object_experiment(
    name: str,
    obj_id: int,
    split: str,
    data_version: str = PER_OBJECT_VERSION,
    model_specs=((SymmetryFieldMLP5, {"dropout": 0.2}),),
    optimizer_specs=OPTIMIZER_SPECS_PER_OBJECT,
    seeds: tuple[int, ...] = (0,),
) -> Experiment:
    return Experiment(
        name=name,
        configs=build_configs(
            experiment_name=name,
            split=split,
            obj_ids={obj_id},
            model_specs=list(model_specs),
            batch_sizes=[32],
            optimizer_specs=optimizer_specs,
            data_version=data_version,
            seeds=seeds,
            **TRAIN_KWARGS,
        ),
    )


# pbr -> real: the early stopping of the base experiments (TRAIN_KWARGS, 431 train uids, bs 32,
# 14 steps per epoch) kept in gradient steps, converted to epochs for a bigger train set / batch
# size. With 431 uids and bs 32 it is TRAIN_KWARGS itself
BASE_N_TRAIN, BASE_BATCH_SIZE = 431, 32


def train_kwargs_by_steps(n_train: int, batch_size: int) -> dict:
    ratio = math.ceil(BASE_N_TRAIN / BASE_BATCH_SIZE) / math.ceil(n_train / batch_size)
    epochs = {
        key: max(1, round(TRAIN_KWARGS[key] * ratio))
        for key in ("max_epochs", "min_epochs", "patience", "eval_every")
    }
    return {**TRAIN_KWARGS, **epochs}


# grid of the base experiments reduced to what mattered there (architecture): MLP5 with and
# without dropout, Adam with constant lr
MODEL_SPECS_PBR_REAL = MODEL_SPECS_PER_OBJECT[:2]
OPTIMIZER_SPECS_PBR_REAL = [
    spec for spec in OPTIMIZER_SPECS_PER_OBJECT_GRID if spec[0] is torch.optim.Adam and spec[2] is None
]


def pbr_real_experiment(
    name: str,
    obj_id: int,
    split: str,
    n_train: int,
    data_version: str = PER_OBJECT_VERSION,
    batch_size: int = 256,
    seeds: tuple[int, ...] = (0, 1, 2),
) -> Experiment:
    """n_train: train uids of the split, sets the early stopping in epochs (train_kwargs_by_steps)."""
    return Experiment(
        name=name,
        configs=build_configs(
            experiment_name=name,
            split=split,
            obj_ids={obj_id},
            model_specs=MODEL_SPECS_PBR_REAL,
            batch_sizes=[batch_size],
            optimizer_specs=OPTIMIZER_SPECS_PBR_REAL,
            data_version=data_version,
            seeds=seeds,
            **train_kwargs_by_steps(n_train, batch_size),
        ),
    )


_EXPERIMENTS = [
    lmo_experiment({10}),
    lmo_experiment({11}),
    lmo_experiment({10, 11}),
    lmo_experiment({10}, DATA_VERSION_NORM),
    lmo_experiment({11}, DATA_VERSION_NORM),
    lmo_experiment({10, 11}, DATA_VERSION_NORM),
    # object diameter: a single object is a constant rescale that the target
    # standardization (Normalizer) undoes, run anyway to report it. Only obj10-11 can change
    lmo_experiment({10}, DATA_VERSION_NORMDIAM),
    lmo_experiment({11}, DATA_VERSION_NORMDIAM),
    lmo_experiment({10, 11}, DATA_VERSION_NORMDIAM),
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
    # norm on pbr, shortlist only (2 runs each); obj10-11 first
    *(
        pbr_lmo_experiment(
            objs,
            DATA_VERSION_NORM,
            model_specs=MODEL_SPECS_SHORTLIST,
            batch_sizes=[1024],
            optimizer_specs=OPTIMIZER_SPECS_SHORTLIST_PBR,
        )
        for objs in ({10, 11}, {10}, {11})
    ),
    per_object_experiment("per-object_eggbox", 10, "per-object-test100"),
    per_object_experiment("per-object_glue", 11, "per-object-test100"),
    per_object_experiment("per-object_masterchefcan", 1, "per-object-test100"),
    per_object_experiment("per-object_masterchefcan_block", 1, "per-object-test100-block"),
    per_object_experiment("per-object_woodblock-z", 16, "per-object-test100"),
    per_object_experiment("per-object_woodblock-z_block", 16, "per-object-test100-block"),
    per_object_experiment("per-object_woodblock-diag", 16, "per-object-test100", PER_OBJECT_VERSION_WOODDIAG),
    per_object_experiment("per-object_woodblock-diag_block", 16, "per-object-test100-block", PER_OBJECT_VERSION_WOODDIAG),
    # final split (visib >= 0.6): lmo random test, ycbv block test; full grid (18 runs each)
    *(
        per_object_experiment(
            f"per-object-visib06_{name}", obj_id, "per-object-visib06", version,
            model_specs=MODEL_SPECS_PER_OBJECT, optimizer_specs=OPTIMIZER_SPECS_PER_OBJECT_GRID,
        )
        for name, obj_id, version in [
            ("eggbox", 10, PER_OBJECT_VERSION),
            ("glue", 11, PER_OBJECT_VERSION),
            ("masterchefcan", 1, PER_OBJECT_VERSION),
            ("woodblock-z", 16, PER_OBJECT_VERSION),
            ("woodblock-diag", 16, PER_OBJECT_VERSION_WOODDIAG),
        ]
    ),
    # base split (notebooks/per_object_models.py 6.6): same train/val/test sizes for the 4 objects
    # (glue is the floor), ycbv test and val as blocks of frames; full grid x 3 seeds (54 runs each).
    # _extended: same configs and val/test, train with all the uids of the scene (no glue)
    *(
        per_object_experiment(
            f"per-object-base_{name}{suffix}", obj_id, split, version,
            model_specs=MODEL_SPECS_PER_OBJECT, optimizer_specs=OPTIMIZER_SPECS_PER_OBJECT_GRID,
            seeds=(0, 1, 2),
        )
        for split, suffix in [("per-object-base", ""), ("per-object-base-extended", "_extended")]
        for name, obj_id, version in [
            ("eggbox", 10, PER_OBJECT_VERSION),
            ("glue", 11, PER_OBJECT_VERSION),
            ("masterchefcan", 1, PER_OBJECT_VERSION),
            ("woodblock-z", 16, PER_OBJECT_VERSION),
            ("woodblock-diag", 16, PER_OBJECT_VERSION_WOODDIAG),
        ]
        if not (suffix and name == "glue")  # glue is the floor: its extended train is the base one
    ),
]
REGISTRY: dict[str, Experiment] = {e.name: e for e in _EXPERIMENTS}
