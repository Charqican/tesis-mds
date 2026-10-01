from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable
import tomllib

import torch

from pose6d.config import LMOConfig
from pose6d.loader import LMOLoader
from pose6d.model import (
    SymmetryFieldMLP1,
    SymmetryFieldMLP2,
    SymmetryFieldMLP3,
    SymmetryFieldMLP4,
    SymmetryFieldMLP5,
)
from experiments.experiment_setup import setup_exp1, setup_exp_cross
from experiments.experiment_training import training_function
from pose6d.dataset import DatasetSource

#  INFO: this configurations does not distinguish between using optimizer or not


# This dataclass is used for the experiments CLI
@dataclass
class Experiment:
    name: str
    setup_func: Callable[..., Any]
    train_eng: Callable[..., Any]
    configs: list[dict]


# automate the training configurations iterating over batch sizes, optimizers and models
def build_configs_exp1(
    experiment_name: str,
    sel_obj_ids: frozenset[int],
    points_pt_dir: Path,
    input_dir: Path,
    target_dir: Path,
    loader: LMOLoader,
    model_specs: list[tuple[type, dict | None]],
    batch_sizes: list[int],
    optimizer_specs: list[tuple[type, dict | None, type | None, dict | None]],
    patience: int | None = None,
    min_delta: float | None = None,
    min_epochs: int | None = None,
) -> list[dict]:
    obj_tag = "-".join(str(i) for i in sorted(sel_obj_ids))
    configs = []
    for batch_size, (
        optimizer_cls,
        optimizer_kwargs,
        scheduler_cls,
        scheduler_kwargs,
    ), (model_cls, model_kwargs) in product(batch_sizes, optimizer_specs, model_specs):
        run_name = (
            f"obj{obj_tag}_{model_cls.__name__}_bs{batch_size}_"
            f"{optimizer_cls.__name__}_{scheduler_cls.__name__ if scheduler_cls else 'noshd'}"
        )
        configs.append(
            {
                "experiment_name": experiment_name,
                "run_name": run_name,
                "setup_conf": {
                    "loader": loader,
                    "points_pt_dir": points_pt_dir,
                    "input_dir": input_dir,
                    "target_dir": target_dir,
                    "model_cls": model_cls,
                    "model_kwargs": model_kwargs,
                    "sel_obj_ids": sel_obj_ids,
                    "batch_size": batch_size,
                },
                "train_conf": {
                    "optimizer_cls": optimizer_cls,
                    "optimizer_kwargs": optimizer_kwargs,
                    "scheduler_cls": scheduler_cls,
                    "scheduler_kwargs": scheduler_kwargs,
                    "patience": patience,
                    "min_delta": min_delta,
                    "min_epochs": min_epochs,
                },
            }
        )
    return configs


# adds test - train  configurations + validation
def build_configs_exp_synthetic(
    experiment_name: str,
    sel_obj_ids: frozenset[int],
    train_source: DatasetSource,
    test_source: DatasetSource,
    model_specs: list[tuple[type, dict | None]],
    batch_sizes: list[int],
    optimizer_specs: list[tuple[type, dict | None, type | None, dict | None]],
    train_scene_ids: frozenset[int] | None = None,
    patience: int | None = None,
    min_delta: float | None = None,
    min_epochs: int | None = None,
) -> list[dict]:
    obj_tag = "-".join(str(i) for i in sorted(sel_obj_ids))
    scene_tag = (
        "sc" + "-".join(str(i) for i in sorted(train_scene_ids))
        if train_scene_ids
        else "scall"
    )
    configs = []
    for batch_size, (
        optimizer_cls,
        optimizer_kwargs,
        scheduler_cls,
        scheduler_kwargs,
    ), (model_cls, model_kwargs) in product(batch_sizes, optimizer_specs, model_specs):
        model_tag = model_cls.__name__
        if model_kwargs:
            kwargs_tag = "-".join(f"{k}{v}" for k, v in sorted(model_kwargs.items()))
            model_tag = f"{model_tag}_{kwargs_tag}"

        run_name = (
            f"obj{obj_tag}_{scene_tag}_bpr-train_lmo-test_{model_tag}_bs{batch_size}_"
            f"{optimizer_cls.__name__}_{scheduler_cls.__name__ if scheduler_cls else 'noshd'}"
        )
        configs.append(
            {
                "experiment_name": experiment_name,
                "run_name": run_name,
                "setup_conf": {
                    "sel_obj_ids": sel_obj_ids,
                    "train_scene_ids": train_scene_ids,
                    "train_source": train_source,
                    "test_source": test_source,
                    "model_cls": model_cls,
                    "model_kwargs": model_kwargs,
                    "batch_size": batch_size,
                },
                "train_conf": {
                    "optimizer_cls": optimizer_cls,
                    "optimizer_kwargs": optimizer_kwargs,
                    "scheduler_cls": scheduler_cls,
                    "scheduler_kwargs": scheduler_kwargs,
                    "patience": patience,
                    "min_delta": min_delta,
                    "min_epochs": min_epochs,
                },
            }
        )
    return configs


# Load toml paths
config_path = Path(__file__).parent / "paths.toml"
with config_path.open("rb") as f:
    data = tomllib.load(f)

# root files
ROOT = Path(data["training"]["root"])
LMO_ROOT = Path(data["lmo"]["root"])
PBR_ROOT = Path(data["pbr"]["root"])

#
lmo_config = LMOConfig.from_root(LMO_ROOT)
LOADER = LMOLoader(lmo_config)
POINTS_PT_DIR = ROOT / data["training"]["points_pt_dir"].lstrip("/")
INPUT_DIR = ROOT / data["training"]["input_dir"].lstrip("/")
TARGET_DIR = ROOT / data["training"]["target_dir"].lstrip("/")

TRAIN_SOURCE_BPR = DatasetSource(
    points_dir=ROOT / "pbr/cache/rm_outliers_20_2_visib_10/points_pT",
    input_dir=ROOT / "pbr/scalarfield_exp3/training/input",
    target_dir=ROOT / "pbr/scalarfield_exp3/training/target",
)
TEST_SOURCE_LMO = DatasetSource(
    points_dir=ROOT / "lmo/cache/rm_outliers_20_2_visib_10/points_pT",
    input_dir=ROOT / "lmo/scalarfield_exp3/training/input",
    target_dir=ROOT / "lmo/scalarfield_exp3/training/target",
)

BATCH_SIZES = [16, 32, 64, 128]
OPTIMIZER_SPECS = [
    (
        torch.optim.SGD,
        {"lr": 1e-2, "momentum": 0.9},
        None,
        None,
    ),
    (
        torch.optim.SGD,
        {"lr": 1e-2, "momentum": 0.9},
        torch.optim.lr_scheduler.StepLR,
        {"step_size": 1000, "gamma": 0.5},
    ),
    (
        torch.optim.Adam,
        {"lr": 1e-3},
        None,
        None,
    ),
    (
        torch.optim.Adam,
        {"lr": 1e-3},
        torch.optim.lr_scheduler.CosineAnnealingLR,
        {"T_max": 5000},
    ),
    (
        torch.optim.AdamW,
        {"lr": 1e-3, "weight_decay": 1e-2},
        None,
        None,
    ),
    (
        torch.optim.AdamW,
        {"lr": 1e-3, "weight_decay": 1e-2},
        torch.optim.lr_scheduler.CosineAnnealingLR,
        {"T_max": 5000},
    ),
]
MODEL_SPECS = [
    (SymmetryFieldMLP1, None),
    (SymmetryFieldMLP2, None),
    (SymmetryFieldMLP3, None),
    (SymmetryFieldMLP4, None),
    (SymmetryFieldMLP5, None),
]
PATIENCE = 50
MIN_DELTA = 0.0005

BATCH_SIZES_CROSS = [128, 256]

# Adam and AdamW only (with and without scheduler)
OPTIMIZER_SPECS_CROSS = [
    spec for spec in OPTIMIZER_SPECS if spec[0] in (torch.optim.Adam, torch.optim.AdamW)
]

MODEL_SPECS_CROSS = [
    (SymmetryFieldMLP5, {"dropout": 0.0}),
    (SymmetryFieldMLP5, {"dropout": 0.2}),
    (SymmetryFieldMLP5, {"dropout": 0.3}),
    (SymmetryFieldMLP4, None),
    (SymmetryFieldMLP1, None),
]

REGISTRY: dict[str, Experiment] = {
    # "exp2_11_10": Experiment(
    #     name="exp2",
    #     setup_func=setup_exp1,
    #     train_eng=training_function,
    #     configs=build_configs_exp1(
    #         experiment_name="experiment_2",
    #         sel_obj_ids=frozenset({10, 11}),
    #         points_pt_dir=POINTS_PT_DIR,
    #         input_dir=INPUT_DIR,
    #         target_dir=TARGET_DIR,
    #         loader=LOADER,
    #         model_specs=MODEL_SPECS,
    #         batch_sizes=BATCH_SIZES,
    #         optimizer_specs=OPTIMIZER_SPECS,
    #         patience=PATIENCE,
    #         min_delta=MIN_DELTA,
    #         min_epochs=2000,
    #     ),
    # ),
    # "exp1_10": Experiment(
    #     name="exp1_10",
    #     setup_func=setup_exp1,
    #     train_eng=training_function,
    #     configs=build_configs_exp1(
    #         experiment_name="experiment_1_10",
    #         sel_obj_ids=frozenset({10}),
    #         points_pt_dir=POINTS_PT_DIR,
    #         input_dir=INPUT_DIR,
    #         target_dir=TARGET_DIR,
    #         loader=LOADER,
    #         model_specs=MODEL_SPECS,
    #         batch_sizes=BATCH_SIZES,
    #         optimizer_specs=OPTIMIZER_SPECS,
    #         patience=PATIENCE,
    #         min_delta=MIN_DELTA,
    #         min_epochs=2000,
    #     ),
    # ),
    # "exp1_11": Experiment(
    #     name="exp1_11",
    #     setup_func=setup_exp1,
    #     train_eng=training_function,
    #     configs=build_configs_exp1(
    #         experiment_name="experiment_1_11",
    #         sel_obj_ids=frozenset({11}),
    #         points_pt_dir=POINTS_PT_DIR,
    #         input_dir=INPUT_DIR,
    #         target_dir=TARGET_DIR,
    #         loader=LOADER,
    #         model_specs=MODEL_SPECS,
    #         batch_sizes=BATCH_SIZES,
    #         optimizer_specs=OPTIMIZER_SPECS,
    #         patience=PATIENCE,
    #         min_delta=MIN_DELTA,
    #         min_epochs=2000,
    #     ),
    # ),
    # "exp3_bpr_train_lmo_test": Experiment(
    #     name="exp3_bpr_train_lmo_test",
    #     setup_func=setup_exp_cross,
    #     train_eng=training_function,
    #     configs=build_configs_exp_synthetic(
    #         experiment_name="experiment_3_cross",
    #         sel_obj_ids=frozenset({10, 11}),
    #         train_source=TRAIN_SOURCE_BPR,
    #         test_source=TEST_SOURCE_LMO,
    #         model_specs=MODEL_SPECS_CROSS,
    #         batch_sizes=BATCH_SIZES_CROSS,
    #         optimizer_specs=OPTIMIZER_SPECS,
    #         patience=PATIENCE,
    #         min_delta=MIN_DELTA,
    #         min_epochs=2000,
    #     ),
    # ),
    "exp3_test_single_scene": Experiment(
        name="exp3_test_single_scene",
        setup_func=setup_exp_cross,
        train_eng=training_function,
        configs=build_configs_exp_synthetic(
            experiment_name="experiment_3_cross_server_test",
            sel_obj_ids=frozenset({10, 11}),
            train_scene_ids=frozenset({3}),
            train_source=TRAIN_SOURCE_BPR,
            test_source=TEST_SOURCE_LMO,
            model_specs=[(SymmetryFieldMLP5, {"dropout": 0.1})],
            batch_sizes=[128],
            optimizer_specs=[OPTIMIZER_SPECS[2]],
            patience=5,
            min_delta=MIN_DELTA,
            min_epochs=10,
        ),
    ),
    "exp3_cross_10_11": Experiment(
        name="exp3_cross_10_11",
        setup_func=setup_exp_cross,
        train_eng=training_function,
        configs=build_configs_exp_synthetic(
            experiment_name="experiment_3_cross_server",
            sel_obj_ids=frozenset({10, 11}),
            train_source=TRAIN_SOURCE_BPR,
            test_source=TEST_SOURCE_LMO,
            model_specs=MODEL_SPECS_CROSS,
            batch_sizes=BATCH_SIZES_CROSS,
            optimizer_specs=OPTIMIZER_SPECS_CROSS,
            patience=PATIENCE,
            min_delta=MIN_DELTA,
            min_epochs=2000,
        ),
    ),
    "exp3_cross_10": Experiment(
        name="exp3_cross_10",
        setup_func=setup_exp_cross,
        train_eng=training_function,
        configs=build_configs_exp_synthetic(
            experiment_name="experiment_3_cross_server_10",
            sel_obj_ids=frozenset({10}),
            train_source=TRAIN_SOURCE_BPR,
            test_source=TEST_SOURCE_LMO,
            model_specs=MODEL_SPECS_CROSS,
            batch_sizes=BATCH_SIZES_CROSS,
            optimizer_specs=OPTIMIZER_SPECS_CROSS,
            patience=PATIENCE,
            min_delta=MIN_DELTA,
            min_epochs=2000,
        ),
    ),
    "exp3_cross_11": Experiment(
        name="exp3_cross_11",
        setup_func=setup_exp_cross,
        train_eng=training_function,
        configs=build_configs_exp_synthetic(
            experiment_name="experiment_3_cross_server_11",
            sel_obj_ids=frozenset({11}),
            train_source=TRAIN_SOURCE_BPR,
            test_source=TEST_SOURCE_LMO,
            model_specs=MODEL_SPECS_CROSS,
            batch_sizes=BATCH_SIZES_CROSS,
            optimizer_specs=OPTIMIZER_SPECS_CROSS,
            patience=PATIENCE,
            min_delta=MIN_DELTA,
            min_epochs=2000,
        ),
    ),
}
