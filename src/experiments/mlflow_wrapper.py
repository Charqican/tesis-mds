from typing import cast

import torch
import numpy as np
import mlflow
import mlflow.pytorch as mlflowpy
from mlflow.models import infer_signature
from logger import notebook_logger as log
from experiments.experiment_setup import TrainData
from pose6d.dataset import SymmetryFieldInstanceDataset


def run_experiment(config: dict, setup_func, train_eng):
    mlflow.set_experiment(config["experiment_name"])
    with mlflow.start_run(run_name=config.get("run_name")):
        # save parameters from setup
        mlflow.log_params(flatten(config))

        train_data: TrainData = setup_func(**config["setup_conf"])
        model, loss_hist, test_hist = train_eng(
            train_data,
            on_epoch=lambda m, step: mlflow.log_metrics(m, step=step),
            **config["train_conf"],
        )

        dataset: SymmetryFieldInstanceDataset = train_data.dataset

        # uids used for train/test
        split_uids = {
            "train": dataset.uids_for("train"),
            "test": dataset.uids_for("test"),
        }
        mlflow.log_dict(split_uids, "split.json")

        # per-instance test error
        device = next(model.parameters()).device
        per_instance_errors = {}
        with torch.no_grad():
            for uid in split_uids["test"]:
                _, inp, target_raw = dataset.get_instance(uid)
                pred = dataset.denormalize(model(inp.to(device)))
                rmse = torch.sqrt(torch.mean((pred.cpu() - target_raw) ** 2)).item()
                per_instance_errors[uid] = rmse
        mlflow.log_dict(per_instance_errors, "per_instance_errors.json")

        input_sample, _ = dataset[0]
        input_sample_np = input_sample.to("cpu").numpy()
        signature = infer_signature(input_sample_np)
        # save model
        mlflowpy.log_model(
            model.eval().to("cpu"),
            "model",
            input_example=cast(np.ndarray, input_sample_np),
            signature=signature,
            serialization_format="pickle",
        )
        # final test loss
        mlflow.log_metric("final_test_loss", test_hist[-1])

    return loss_hist, test_hist, model, dataset


# logger function for formatting mlflow
def flatten(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items
