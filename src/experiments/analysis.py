import json
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch

import pose6d.model
from pose6d.dataset import Normalizer, Splits
from pose6d.layout import DataLayout, load_symm_configs
from pose6d.loader import BOPLoader, loader_from_env

"""
Analysis of the runs logged by experiments/mlflow_wrapper.py (notebooks/pose6d_analysis.py).

    runs = runs_table(["lmo-strat_obj10", "lmo-strat_obj11"])   # one row per run
    instances = runs_instances(runs["run_id"])                 # per uid metrics of those runs
    run = load_run(run_id)                                     # model, splits, normalizer, instances
    points, target, pred = predict_instance(run, layout, uid)  # uses the run's normalizer
"""

RUN_COLUMNS = {
    "tags.mlflow.runName": "run_name",
    "params.data.split": "split",
    "params.data.obj_ids": "obj_ids",
    "params.model.cls": "model",
    "params.train.batch_size": "batch_size",
    "params.train.optimizer_cls": "optimizer",
    "params.train.scheduler_cls": "scheduler",
    "metrics.final_test_loss": "final_test_loss",
    "metrics.final_val_loss": "final_val_loss",
    "metrics.final_train_loss": "final_train_loss",
    "metrics.test_rmse_mean": "test_rmse_mean",
    "metrics.test_rmse_median": "test_rmse_median",
    "metrics.best_epoch": "best_epoch",
    "run_id": "run_id",
}


@dataclass
class RunArtifacts:
    run_id: str
    run_name: str
    params: dict
    model: torch.nn.Module
    splits: Splits
    normalizer: Normalizer
    instances: pd.DataFrame


# --- many runs
# model kwargs are flattened (params.model.kwargs.dropout): SymmetryFieldMLP5-dropout0.2
def _model_label(run: pd.Series) -> str:
    kwargs = [
        f"-{key.rsplit('.', 1)[-1]}{value}"
        for key, value in run.items()
        if key.startswith("params.model.kwargs.") and isinstance(value, str)
    ]
    return run["params.model.cls"] + "".join(kwargs)


def runs_table(experiment_names: list[str]) -> pd.DataFrame:
    """Finished runs of these mlflow experiments, sorted by final_test_loss."""
    runs = mlflow.search_runs(
        experiment_names=experiment_names,
        filter_string="status = 'FINISHED'",
        order_by=["metrics.final_test_loss ASC"],
    )
    names = {
        e.experiment_id: e.name
        for e in mlflow.search_experiments()
        if e.name in experiment_names
    }
    # columns missing when no run logged them (e.g. no scheduler) are left empty
    table = runs.reindex(columns=list(RUN_COLUMNS)).rename(columns=RUN_COLUMNS)
    table.insert(0, "experiment", runs["experiment_id"].map(names))
    table["model"] = runs.apply(_model_label, axis=1)
    return table


def runs_instances(run_ids) -> pd.DataFrame:
    """instances.parquet of every run, with a run_id column."""
    return pd.concat(
        [
            pd.read_parquet(
                mlflow.artifacts.download_artifacts(
                    run_id=run_id, artifact_path="instances.parquet"
                )
            ).assign(run_id=run_id)
            for run_id in run_ids
        ],
        ignore_index=True,
    )


# --- one run
def load_run(run_id: str) -> RunArtifacts:
    run = mlflow.get_run(run_id)
    artifacts_dir = Path(mlflow.artifacts.download_artifacts(run_id=run_id))

    def read_json(name: str) -> dict:
        return json.loads((artifacts_dir / name).read_text())

    model_info = read_json("model.json")
    model_cls = getattr(pose6d.model, model_info["cls"])
    model = model_cls(model_info["in_dim"], **model_info["kwargs"])
    model.load_state_dict(torch.load(artifacts_dir / "model.pt", map_location="cpu"))

    return RunArtifacts(
        run_id=run_id,
        run_name=run.data.tags.get("mlflow.runName", ""),
        params=dict(run.data.params),
        model=model.eval(),
        splits=Splits.from_dict(read_json("splits.json")),
        normalizer=Normalizer.from_dict(read_json("normalizer.json")),
        instances=pd.read_parquet(artifacts_dir / "instances.parquet"),
    )


def predict_instance(
    run: RunArtifacts, layout: DataLayout, uid: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(points, target, pred) of one local uid, target and pred in target units."""
    points, features, target = load_instance(layout, run.params["data.version"], uid)
    with torch.no_grad():
        inputs = run.normalizer.inputs(torch.from_numpy(features).float())
        pred = run.normalizer.denormalize_targets(run.model(inputs))
    return points, target, pred.numpy()


# --- data of one version
def load_instance(
    layout: DataLayout, version: str, uid: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw (points, dgedi features, target) of one uid."""
    pT = load_symm_configs()[version].pT
    return (
        np.load(layout.points_path(pT, uid))["points"],
        np.load(layout.input_dir(version) / f"{uid}.npz")["features"],
        np.load(layout.target_dir(version) / f"{uid}.npz")["target"],
    )


def uid_info(uids: list[str]) -> pd.DataFrame:
    """dataset, scene_id, img_id, obj_id, inst_idx and visib_fract of each uid."""
    loaders: dict[str, BOPLoader] = {}
    rows = []
    for uid in uids:
        dataset, scene_id, img_id, obj_id, inst_idx = BOPLoader.parse_instance_uid(uid)
        if dataset not in loaders:
            loaders[dataset] = loader_from_env(dataset)
        instance = loaders[dataset].load_instances(scene_id, img_id)[inst_idx]
        rows.append(
            {
                "uid": uid,
                "dataset": dataset,
                "scene_id": scene_id,
                "img_id": img_id,
                "obj_id": obj_id,
                "inst_idx": inst_idx,
                "visib_fract": instance.visible_fract,
            }
        )
    return pd.DataFrame(rows)
