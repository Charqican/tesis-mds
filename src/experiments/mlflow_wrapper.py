from typing import cast

import torch
import numpy as np
import mlflow
import mlflow.pytorch as mlflowpy
from mlflow.models import infer_signature
from logger import notebook_logger as log
from experiments.experiment_setup import TrainData
from pose6d.dataset import SymmetryFieldInstanceDataset
from pose6d.loader import BOPLoader


def run_experiment(config: dict, setup_func, train_eng):
    mlflow.set_experiment(config["experiment_name"])
    with mlflow.start_run(run_name=config.get("run_name")):
        mlflow.log_params(flatten(config))

        train_data: TrainData = setup_func(**config["setup_conf"])
        model, loss_hist, val_hist = train_eng(
            train_data,
            on_epoch=lambda m, step: mlflow.log_metrics(m, step=step),
            **config["train_conf"],
        )
        model.eval()

        dataset: SymmetryFieldInstanceDataset = train_data.dataset

        split_uids = {
            "train": dataset.uids_for("train"),
            "val": dataset.uids_for("val"),
            "test": dataset.uids_for("test"),
        }
        mlflow.log_dict(split_uids, "split.json")

        device = next(model.parameters()).device
        per_instance_errors: dict[str, float] = {}
        errors_by_obj: dict[int, list[float]] = {}

        with torch.no_grad():
            for uid in split_uids["test"]:
                _, inp, target_raw = dataset.get_instance(uid)
                pred = dataset.denormalize(model(inp.to(device)))
                rmse = torch.sqrt(torch.mean((pred.cpu() - target_raw) ** 2)).item()
                per_instance_errors[uid] = rmse

                _, _, _, obj_id, _ = BOPLoader.parse_instance_uid(uid)
                mlflow.log_metric(f"test_rmse/obj{obj_id:06d}/{uid}", rmse)
                errors_by_obj.setdefault(obj_id, []).append(rmse)

        mlflow.log_dict(per_instance_errors, "per_instance_errors.json")

        # -- global, mean rmse
        rmse_values = list(per_instance_errors.values())
        mlflow.log_metric("test_rmse_mean", float(np.mean(rmse_values)))
        mlflow.log_metric("test_rmse_median", float(np.median(rmse_values)))

        worst_uid = max(per_instance_errors, key=per_instance_errors.get)
        mlflow.log_metric("test_rmse_worst", per_instance_errors[worst_uid])
        mlflow.set_tag("test_rmse_worst_uid", worst_uid)

        # -- per object
        obj_mean = {
            obj_id: float(np.mean(errors)) for obj_id, errors in errors_by_obj.items()
        }
        obj_median = {
            obj_id: float(np.median(errors)) for obj_id, errors in errors_by_obj.items()
        }
        for obj_id in errors_by_obj:
            mlflow.log_metric(f"test_rmse_mean/obj{obj_id:06d}", obj_mean[obj_id])
            mlflow.log_metric(f"test_rmse_median/obj{obj_id:06d}", obj_median[obj_id])

        worst_obj_id = max(obj_mean, key=obj_mean.get)
        mlflow.log_metric("test_rmse_worst_obj_mean", obj_mean[worst_obj_id])
        mlflow.set_tag("test_rmse_worst_obj_id", str(worst_obj_id))

        # -- loss final (normalizado, comparable con las curvas de entrenamiento)
        test_idx = dataset.indices_for("test")
        test_input = dataset.input[test_idx].to(device)
        test_targets = dataset.targets[test_idx].to(device)
        with torch.no_grad():
            test_pred = model(test_input.reshape(-1, train_data.input_dim)).reshape(
                test_targets.shape
            )
            final_test_loss = (
                ((test_pred - test_targets) ** 2).mean(dim=1).mean().item()
            )

        mlflow.log_metric("final_test_loss", final_test_loss)
        mlflow.log_metric("final_val_loss", val_hist[-1])
        mlflow.log_metric("final_train_loss", loss_hist[-1])

        # log model
        input_sample, _ = dataset[0]
        input_sample_np = input_sample.to("cpu").numpy()
        signature = infer_signature(input_sample_np)
        mlflowpy.log_model(
            model.to("cpu"),
            "model",
            input_example=cast(np.ndarray, input_sample_np),
            signature=signature,
            serialization_format="pickle",
        )

    return loss_hist, val_hist, model, dataset


def flatten(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items
