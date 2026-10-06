import tempfile
import time
from pathlib import Path

import mlflow
import pandas as pd
import torch

from experiments.experiment_setup import setup
from experiments.experiment_training import instance_losses, predict, train
from pose6d.dataset import SPLIT_NAMES, SymmetryFieldDataset
from pose6d.layout import DataLayout
from pose6d.loader import BOPLoader, loader_from_env
from logger import experiments_logger as log

"""
One mlflow run per config. Logged:
    params      the flattened config (classes by name)
    tags        obj_ids, split, model, optimizer (to group runs)
    metrics     train/val loss curves; final_{train,val,test}_loss and test_rmse_{mean,median}
                of the best model; best_epoch, n_epochs, n_{train,val,test}, train_time_s
    artifacts   splits.json, normalizer.json, instances.parquet (per uid metrics),
                model.pt (state_dict) + model.json (class, kwargs, in_dim)
"""


def run_experiment(config: dict, layout: DataLayout) -> None:
    data, model_conf, train_conf = config["data"], config["model"], config["train"]
    train_conf = dict(train_conf)
    seed = train_conf.pop("seed")

    mlflow.set_experiment(config["experiment_name"])
    with mlflow.start_run(run_name=config["run_name"]):
        mlflow.log_params(flatten({k: config[k] for k in ("data", "model", "train")}))
        mlflow.set_tags(
            {
                "obj_ids": "-".join(map(str, data["obj_ids"])),
                "split": data["split"],
                "model": model_conf["cls"].__name__,
                "optimizer": train_conf["optimizer_cls"].__name__,
            }
        )

        ds, model = setup(layout, data, model_conf, seed)
        mlflow.log_metrics({f"n_{n}": len(getattr(ds.splits, n)) for n in SPLIT_NAMES})

        start = time.monotonic()
        result = train(
            model,
            ds.train,
            ds.val,
            on_eval=lambda m, step: mlflow.log_metrics(m, step=step),
            **train_conf,
        )
        train_time = time.monotonic() - start

        instances = instance_metrics(result.model, ds)
        test = instances[instances["split"] == "test"]
        mlflow.log_metrics(
            {
                **{
                    f"final_{n}_loss": instances.loc[
                        instances["split"] == n, "loss"
                    ].mean()
                    for n in SPLIT_NAMES
                },
                "test_rmse_mean": test["rmse"].mean(),
                "test_rmse_median": test["rmse"].median(),
                "best_epoch": result.best_epoch,
                "n_epochs": result.n_epochs,
                "train_time_s": train_time,
            }
        )

        mlflow.log_dict(ds.splits.to_dict(), "splits.json")
        mlflow.log_dict(ds.normalizer.to_dict(), "normalizer.json")
        mlflow.log_dict(
            {
                "cls": model_conf["cls"].__name__,
                "kwargs": model_conf["kwargs"] or {},
                "in_dim": ds.train.inputs.shape[-1],
            },
            "model.json",
        )
        # INFO: Mlflow limitation: needs things to exist in disk
        with tempfile.TemporaryDirectory() as tmp:
            instances.to_parquet(Path(tmp) / "instances.parquet", index=False)
            torch.save(result.model.cpu().state_dict(), Path(tmp) / "model.pt")
            mlflow.log_artifacts(tmp)
        log.info(
            f"Finished {config['run_name']}: test loss {test['loss'].mean():.4f}, "
            f"test rmse {test['rmse'].mean():.4f}"
        )


def instance_metrics(model: torch.nn.Module, ds: SymmetryFieldDataset) -> pd.DataFrame:
    """One row per uid of every split. loss is normalized, the rest in target units."""
    device = next(model.parameters()).device
    loaders: dict[str, BOPLoader] = {}
    rows = []
    for split_name in SPLIT_NAMES:
        if not getattr(ds.splits, split_name):
            continue
        split = getattr(ds, split_name)
        pred_norm = predict(model, split, device)
        losses = instance_losses(pred_norm, split.targets)
        pred = ds.normalizer.denormalize_targets(pred_norm)
        target = ds.normalizer.denormalize_targets(split.targets)
        error = pred - target

        for i, uid in enumerate(split.uids):
            dataset, scene_id, img_id, obj_id, inst_idx = BOPLoader.parse_instance_uid(
                uid
            )
            if dataset not in loaders:
                loaders[dataset] = loader_from_env(dataset)
            instance = loaders[dataset].load_instances(scene_id, img_id)[inst_idx]
            rows.append(
                {
                    "uid": uid,
                    "split": split_name,
                    "dataset": dataset,
                    "scene_id": scene_id,
                    "img_id": img_id,
                    "obj_id": obj_id,
                    "inst_idx": inst_idx,
                    "visib_fract": instance.visible_fract,
                    "loss": losses[i].item(),
                    "rmse": error[i].pow(2).mean().sqrt().item(),
                    "mae": error[i].abs().mean().item(),
                    "bias": error[i].mean().item(),
                    "target_mean": target[i].mean().item(),
                    "pred_mean": pred[i].mean().item(),
                }
            )
    return pd.DataFrame(rows)


def flatten(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten(v, new_key, sep=sep))
        elif isinstance(v, type):
            items[new_key] = v.__name__
        else:
            items[new_key] = v
    return items
