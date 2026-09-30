import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.export import ExportedProgram
import mlflow
import mlflow.pytorch as mlflowpy

from pose6d.dataset import SymmetryFieldInstanceDataset
from pose6d.loader import BOPLoader


@dataclass
class RunArtifacts:
    run_id: str
    run_name: str
    config: dict
    model: torch.nn.Module
    dataset: SymmetryFieldInstanceDataset
    split_uids: dict[str, list[str]]


def _unwrap(model):
    # duck-typed: some mlflow/cloudpickle round-trips deserialize ExportedProgram
    # under a class object that fails isinstance() against this module's import
    if isinstance(model, torch.nn.Module):
        return model
    if hasattr(model, "module"):
        return model.module()
    return model


def rank_runs(
    experiment_name: str, metric: str = "final_test_loss", ascending: bool = True
) -> pd.DataFrame:
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"no mlflow experiment named {experiment_name!r}")
    order = "ASC" if ascending else "DESC"
    return mlflow.search_runs(
        experiment_ids=[exp.experiment_id], order_by=[f"metrics.{metric} {order}"]
    )


def load_run(run_id: str) -> RunArtifacts:
    run = mlflow.get_run(run_id)
    params = run.data.params

    points_pt_dir = Path(params["setup_conf.points_pt_dir"])
    input_dir = Path(params["setup_conf.input_dir"])
    target_dir = Path(params["setup_conf.target_dir"])

    split_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="split.json"
    )
    split_uids = json.loads(Path(split_path).read_text())

    dataset = SymmetryFieldInstanceDataset(
        points_pt_dir,
        input_dir,
        target_dir,
        uids=split_uids["train"] + split_uids["test"],
    )
    dataset.assign_splits_explicit(split_uids["test"])
    dataset.load()

    model = _unwrap(mlflowpy.load_model(f"runs:/{run_id}/model"))

    return RunArtifacts(
        run_id=run_id,
        run_name=run.data.tags.get("mlflow.runName", ""),
        config=dict(params),
        model=model,
        dataset=dataset,
        split_uids=split_uids,
    )


def best_run(
    experiment_name: str, metric: str = "final_test_loss", ascending: bool = True
) -> RunArtifacts:
    ranked = rank_runs(experiment_name, metric, ascending)
    return load_run(ranked.iloc[0]["run_id"])


def instance_errors(
    run: RunArtifacts,
    uids: list[str],
    device=None,
    loader: BOPLoader | None = None,
) -> pd.DataFrame:
    device = device or next(run.model.parameters()).device
    model = _unwrap(run.model).to(device).eval()

    frame_cache: dict[tuple[int, int], list] = {}
    rows = []
    with torch.no_grad():
        for uid in uids:
            _, inp, target_raw = run.dataset.get_instance(uid)
            pred = run.dataset.denormalize(model(inp.to(device)).cpu()).numpy()
            target = target_raw.numpy()
            error = pred - target

            row = {
                "uid": uid,
                "rmse": float(np.sqrt((error**2).mean())),
                "mae": float(np.abs(error).mean()),
                "bias": float(error.mean()),
                "target_mean": float(target.mean()),
                "pred_mean": float(pred.mean()),
            }

            if loader is not None:
                dataset_name, scene_id, img_id, obj_id, inst_idx = (
                    loader.parse_instance_uid(uid)
                )
                key = (scene_id, img_id)
                if key not in frame_cache:
                    frame_cache[key] = loader.load_instances(scene_id, img_id)
                row["visib_fract"] = frame_cache[key][inst_idx].visible_fract

            rows.append(row)

    return pd.DataFrame(rows)


def point_errors(
    run: RunArtifacts, uid: str, device=None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = device or next(run.model.parameters()).device
    model = _unwrap(run.model).to(device).eval()

    points, inp, target_raw = run.dataset.get_instance(uid)
    with torch.no_grad():
        pred = run.dataset.denormalize(model(inp.to(device)).cpu()).numpy()

    return points.detach().cpu().numpy(), target_raw.numpy(), pred
