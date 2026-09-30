from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable
import tomllib

import torch

from pose6d.config import LMOConfig
from pose6d.loader import LMOLoader
from pose6d.model import SymmetryFieldMLP
from experiments.experiment_setup import setup_exp1
from experiments.experiment_training import training_function

#  INFO: this configurations does not distinguish between using optimizer or not


# This dataclass is used for the experiments CLI
@dataclass
class Experiment:
    name: str
    setup_func: Callable[..., Any]
    train_eng: Callable[..., Any]
    configs: list[dict]


# automate the training configurations iterating over batch sizes and optimizers
def build_configs_exp1(
    experiment_name: str,
    sel_obj_ids: frozenset[int],
    points_pt_dir: Path,
    input_dir: Path,
    target_dir: Path,
    loader: LMOLoader,
    model_cls: type,
    batch_sizes: list[int],
    optimizer_specs: list[tuple[type, dict | None, type | None, dict | None]],
    model_kwargs: dict | None = None,
) -> list[dict]:
    obj_tag = "-".join(str(i) for i in sorted(sel_obj_ids))
    configs = []
    for batch_size, (
        optimizer_cls,
        optimizer_kwargs,
        scheduler_cls,
        scheduler_kwargs,
    ) in product(batch_sizes, optimizer_specs):
        run_name = f"obj{obj_tag}_bs{batch_size}_{optimizer_cls.__name__}_{scheduler_cls.__name__ if scheduler_cls else 'noshd'}"
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
                },
            }
        )
    return configs


# Load toml paths
config_path = Path(__file__).parent / "paths.toml"
with config_path.open("rb") as f:
    data = tomllib.load(f)

# root files
ROOT = Path(data["training_lmo"]["root"])
LMO_ROOT = Path(data["lmo"]["root"])
#
lmo_config = LMOConfig.from_root(LMO_ROOT)
LOADER = LMOLoader(lmo_config)
POINTS_PT_DIR = Path(ROOT / data["training_lmo"]["points_pt_dir"])
INPUT_DIR = Path(ROOT / data["training_lmo"]["input_dir"])
TARGET_DIR = Path(ROOT / data["training_lmo"]["target_dir"])

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
REGISTRY: dict[str, Experiment] = {
    "exp1_10": Experiment(
        name="exp1_10",
        setup_func=setup_exp1,
        train_eng=training_function,
        configs=build_configs_exp1(
            experiment_name="experiment_1_10",
            sel_obj_ids=frozenset({10}),
            points_pt_dir=POINTS_PT_DIR,
            input_dir=INPUT_DIR,
            target_dir=TARGET_DIR,
            loader=LOADER,
            model_cls=SymmetryFieldMLP,
            batch_sizes=BATCH_SIZES,
            optimizer_specs=OPTIMIZER_SPECS,
        ),
    ),
    "exp1_11": Experiment(
        name="exp1_11",
        setup_func=setup_exp1,
        train_eng=training_function,
        configs=build_configs_exp1(
            experiment_name="experiment_1_11",
            sel_obj_ids=frozenset({11}),
            points_pt_dir=POINTS_PT_DIR,
            input_dir=INPUT_DIR,
            target_dir=TARGET_DIR,
            loader=LOADER,
            model_cls=SymmetryFieldMLP,
            batch_sizes=BATCH_SIZES,
            optimizer_specs=OPTIMIZER_SPECS,
        ),
    ),
}
