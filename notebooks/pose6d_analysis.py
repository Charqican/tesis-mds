import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Análisis de Modelos y EDA
    """)
    return


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from pose6d.loader import LMOLoader, PBRLoader
    from pose6d.selection import uids_by_visib_percentile
    from pose6d.notebook_plots import (
        plot_mesh_with_scalar_field,
        plot_mesh_instance_with_dgedi_features,
    )

    SCENE_ID = 2

    lmo_root = Path("/mnt/data/dev/dataset/tesis/BOP/lmo/lmo")
    pbr_root = Path("/mnt/data/dev/dataset/tesis/BOP/pbr/lm_train_pbr/train_pbr/")
    lmo_loader = LMOLoader.from_root(lmo_root)
    pbr_loader = PBRLoader.from_roots(pbr_root, lmo_root)
    loader = pbr_loader

    ROOT = Path("/mnt/data/dev/dataset/tesis/6dpose")
    # outlier-removed data
    POINTS_PT_DIR = ROOT / f"{loader.dataset_name}/cache/rm_outliers_20_2/points_pT/"
    FEATURES_INPUT_DIR = ROOT / f"{loader.dataset_name}/scalarfield_rm/training/input/"
    TARGET_DIR = ROOT / f"{loader.dataset_name}/scalarfield_rm/training/target/"

    extracted_uids = {p.stem for p in POINTS_PT_DIR.glob("*.npz")}
    return (
        FEATURES_INPUT_DIR,
        POINTS_PT_DIR,
        SCENE_ID,
        TARGET_DIR,
        extracted_uids,
        loader,
        mo,
        np,
        pd,
        plot_mesh_instance_with_dgedi_features,
        plot_mesh_with_scalar_field,
        plt,
        sns,
        uids_by_visib_percentile,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Functions

    Plot helpers live in `pose6d/notebook_plots.py`. Below: small loaders specific to this notebook.
    """)
    return


@app.cell
def _():
    from pose6d.dataset import load_instance_npz

    return (load_instance_npz,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## EDA — LMO dataset

    Quick look at the data before touching any trained model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Dataset info
    """)
    return


@app.cell
def _(loader):
    from pose6d.dataset_stats import print_summary_table, _dataset_summary

    print_summary_table(_dataset_summary(loader, 2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visibility distribution — objects 10 and 11

    `visib_fract` per instance, across all frames of the scene.
    """)
    return


@app.cell
def _(SCENE_ID, loader, pd):
    instances_10 = loader.list_instances(SCENE_ID, obj_id=10)
    instances_11 = loader.list_instances(SCENE_ID, obj_id=11)

    visib_10 = [
        inst.visible_fract
        for _, _, inst in instances_10
        if inst.visible_fract is not None
    ]
    visib_11 = [
        inst.visible_fract
        for _, _, inst in instances_11
        if inst.visible_fract is not None
    ]

    df_visib = pd.DataFrame(
        {
            "visib_fract": visib_10 + visib_11,
            "obj_id": [10] * len(visib_10) + [11] * len(visib_11),
        }
    )
    return df_visib, instances_10, instances_11, visib_10, visib_11


@app.cell
def _(plt, sns, visib_10, visib_11):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    sns.histplot(visib_10, bins=30, ax=_ax1, color="steelblue")
    _ax1.set(title="obj 10 - visib_fract", xlabel="visib_fract")
    sns.histplot(visib_11, bins=30, ax=_ax2, color="darkorange")
    _ax2.set(title="obj 11 - visib_fract", xlabel="visib_fract")
    _fig
    return


@app.cell
def _(df_visib, plt, sns):
    _fig, _ax = plt.subplots(figsize=(7, 4))
    sns.histplot(
        data=df_visib.astype({"obj_id": str}),
        x="visib_fract",
        hue="obj_id",
        multiple="layer",
        bins=30,
        ax=_ax,
    )
    _ax.set(title="obj 10 vs 11 - visib_fract")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Instances at different visibility levels

    Pick an object and a visibility percentile — shows the symmetry-field target and the dGeDi
    features (PCA-to-RGB) for one representative instance. Heavy plots, so it only renders one
    instance at a time.
    """)
    return


@app.cell
def _(
    SCENE_ID,
    extracted_uids,
    instances_10,
    instances_11,
    loader,
    mo,
    uids_by_visib_percentile,
):
    percentiles = [10.0, 50.0, 90.0]
    uids_by_obj = {
        10: uids_by_visib_percentile(
            loader, instances_10, SCENE_ID, percentiles, valid_uids=extracted_uids
        ),
        11: uids_by_visib_percentile(
            loader, instances_11, SCENE_ID, percentiles, valid_uids=extracted_uids
        ),
    }

    obj_dropdown = mo.ui.dropdown(options=["10", "11"], value="10", label="object")
    percentile_dropdown = mo.ui.dropdown(
        options=[str(p) for p in percentiles],
        value=str(percentiles[0]),
        label="visib percentile",
    )
    mo.hstack([obj_dropdown, percentile_dropdown])
    return obj_dropdown, percentile_dropdown, uids_by_obj


@app.cell
def _(
    FEATURES_INPUT_DIR,
    POINTS_PT_DIR,
    TARGET_DIR,
    load_instance_npz,
    obj_dropdown,
    percentile_dropdown,
    uids_by_obj,
):
    selected_obj = int(obj_dropdown.value)
    selected_p = float(percentile_dropdown.value)
    print(uids_by_obj)
    selected_uid = uids_by_obj[selected_obj][selected_p]

    points, features, target, _ = load_instance_npz(
        selected_uid, POINTS_PT_DIR, FEATURES_INPUT_DIR, TARGET_DIR
    )
    return features, points, selected_uid, target


@app.cell
def _(loader, plot_mesh_with_scalar_field, points, selected_uid, target):
    _dataset_name, _scene_id, _img_id, _obj_id, _inst_idx = loader.parse_instance_uid(
        selected_uid
    )
    plot_mesh_with_scalar_field(
        loader,
        _scene_id,
        _img_id,
        _inst_idx,
        points,
        target,
        colorbar_title="symmetry field",
    )
    return


@app.cell
def _(
    features,
    loader,
    plot_mesh_instance_with_dgedi_features,
    points,
    selected_uid,
):
    _dataset_name, _scene_id, _img_id, _obj_id, _inst_idx = loader.parse_instance_uid(
        selected_uid
    )
    plot_mesh_instance_with_dgedi_features(
        loader,
        _scene_id,
        _img_id,
        _inst_idx,
        points,
        features,
        percentiles=(1.0, 100.0),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiments analysis

    Pick a run from an MLflow experiment, then inspect it: training curves, error
    distributions on train/test, error vs visibility, and error fields on the mesh.
    """)
    return


@app.cell
def _():
    from dotenv import load_dotenv

    load_dotenv()
    import mlflow
    from mlflow.tracking import MlflowClient

    from experiments.analysis import (
        rank_runs,
        load_run,
        instance_errors,
        point_errors,
    )
    from pose6d.notebook_plots import (
        plot_gt_vs_pred_comparison,
        plot_gt_vs_pred_comparison_robust,
        plot_error_boxplot_by_group,
        combine_splits,
    )

    return (
        MlflowClient,
        combine_splits,
        instance_errors,
        load_run,
        plot_error_boxplot_by_group,
        plot_gt_vs_pred_comparison,
        plot_gt_vs_pred_comparison_robust,
        point_errors,
        rank_runs,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.1 — Pick a run

    Runs ranked by final test loss. Pick one config to analyze in the rest of this section.
    """)
    return


@app.cell
def _(rank_runs):
    mlflow_experiment_name = "experiment_3_cross"

    ranked_runs = rank_runs(mlflow_experiment_name)
    ranked_runs[["run_id", "tags.mlflow.runName", "metrics.final_test_loss"]].head(6)
    return (ranked_runs,)


@app.cell
def _(mo, ranked_runs):
    run_options = {
        f"{row['tags.mlflow.runName']} ({row['run_id'][:8]})": row["run_id"]
        for _, row in ranked_runs.iterrows()
    }
    run_dropdown = mo.ui.dropdown(
        options=run_options, value=list(run_options)[0], label="run"
    )
    run_dropdown
    return (run_dropdown,)


@app.cell(hide_code=True)
def _(load_run, run_dropdown):
    run = load_run(run_dropdown.value)
    run.run_name
    return (run,)


@app.cell
def _(MlflowClient, pd, plt, run, sns):
    _client = MlflowClient()
    _train_hist = _client.get_metric_history(run.run_id, "loss")
    _test_hist = _client.get_metric_history(run.run_id, "test_loss")

    df_loss = pd.DataFrame(
        [{"epoch": m.step, "loss": m.value, "split": "train"} for m in _train_hist]
        + [{"epoch": m.step, "loss": m.value, "split": "test"} for m in _test_hist]
    )

    _fig, _ax = plt.subplots(figsize=(7, 4))
    sns.lineplot(data=df_loss, x="epoch", y="loss", hue="split", ax=_ax)
    _ax.set_yscale("log")
    _ax.set(title=run.run_name, xlabel="epoch", ylabel="loss")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.2 / 5.3 — Error distribution, test vs train

    Per-instance RMSE. Train errors show fit quality; test errors show generalization.
    """)
    return


@app.cell
def _(instance_errors, loader, run):
    errors_test = instance_errors(
        run, run.split_uids["test"], device="cuda", loader=loader
    )
    errors_train = instance_errors(
        run, run.split_uids["train"], device="cuda", loader=loader
    )
    return errors_test, errors_train


@app.cell
def _(errors_test, errors_train, plt, sns):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    sns.histplot(errors_test["rmse"].values, bins=20, ax=_ax1)
    _ax1.set(title="test - per-instance RMSE", xlabel="rmse")
    sns.histplot(errors_train["rmse"].values, bins=20, ax=_ax2)
    _ax2.set(title="train - per-instance RMSE", xlabel="rmse")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.4 — Error vs visibility

    Does occlusion predict error?
    """)
    return


@app.cell
def _(combine_splits, errors_test, errors_train, plt, sns):
    errors_by_split = combine_splits({"test": errors_test, "train": errors_train})

    _fig, _ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(
        data=errors_by_split, x="visib_fract", y="rmse", hue="split", ax=_ax
    )
    _ax.set(title="RMSE vs visib_fract", xlabel="visib_fract", ylabel="rmse")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.5 — GT vs predicted, one instance

    Pick a split and a rank (best / median / worst by RMSE) to inspect one representative
    instance: per-point error histogram for it.
    """)
    return


@app.cell
def _(mo):
    split_dropdown = mo.ui.dropdown(
        options=["test", "train"], value="test", label="split"
    )
    error_percentile_dropdown = mo.ui.dropdown(
        options=["best", "median", "worst"], value="best", label="rank"
    )
    mo.hstack([split_dropdown, error_percentile_dropdown])
    return error_percentile_dropdown, split_dropdown


@app.cell
def _(error_percentile_dropdown, errors_test, errors_train, split_dropdown):
    _errors = errors_test if split_dropdown.value == "test" else errors_train
    _sorted = _errors.sort_values("rmse").reset_index(drop=True)
    _rank = {"best": 0, "median": len(_sorted) // 2, "worst": len(_sorted) - 1}[
        error_percentile_dropdown.value
    ]
    error_uid = _sorted.loc[_rank, "uid"]
    error_uid
    return (error_uid,)


@app.cell
def _(error_uid, point_errors, run):
    error_points, error_target, error_pred = point_errors(run, error_uid)
    return error_points, error_pred, error_target


@app.cell
def _(error_pred, error_target, error_uid, plt, sns):
    _fig, _ax = plt.subplots(figsize=(7, 4))
    sns.histplot(error_pred - error_target, bins=20, ax=_ax)
    _ax.set(title=f"{error_uid} - per-point error", xlabel="pred - target")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.6 — Per-point error by group

    Pooled per-point errors for low/high visibility and best-3/worst-3 instances by RMSE
    (from train — most useful on a larger split).
    """)
    return


@app.cell
def _(errors_train, np, point_errors, run):
    _sorted_by_visib = errors_train.sort_values("visib_fract")
    _low_visib_uids = _sorted_by_visib["uid"].head(3).tolist()
    _high_visib_uids = _sorted_by_visib["uid"].tail(3).tolist()

    _sorted_by_rmse = errors_train.sort_values("rmse")
    _best_uids = _sorted_by_rmse["uid"].head(3).tolist()
    _worst_uids = _sorted_by_rmse["uid"].tail(3).tolist()

    def _pooled_error(uids):
        return np.concatenate(
            [
                (point_errors(run, uid)[2] - point_errors(run, uid)[1]).ravel()
                for uid in uids
            ]
        )

    error_groups = {
        "low visib": _pooled_error(_low_visib_uids),
        "high visib": _pooled_error(_high_visib_uids),
        "best-3 rmse": _pooled_error(_best_uids),
        "worst-3 rmse": _pooled_error(_worst_uids),
    }
    return (error_groups,)


@app.cell
def _(error_groups, plot_error_boxplot_by_group):
    plot_error_boxplot_by_group(error_groups, title="train - per-point error by group")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.7 — Error field on the mesh

    Same 3D viz as the symmetry field, but colored by `pred - target` for the selected instance.
    """)
    return


@app.cell
def _(
    error_points,
    error_pred,
    error_target,
    error_uid,
    loader,
    plot_mesh_with_scalar_field,
):
    _dataset_name, _scene_id, _img_id, _obj_id, _inst_idx = loader.parse_instance_uid(
        error_uid
    )
    plot_mesh_with_scalar_field(
        loader,
        _scene_id,
        _img_id,
        _inst_idx,
        error_points,
        error_pred - error_target,
        colorscale="RdBu",
        colorbar_title="pred - target",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.8 — GT vs predicted, shared colorbar

    Same colorscale and range for GT and predicted fields, so they're actually comparable
    side by side rather than each auto-scaled on its own.
    """)
    return


@app.cell
def _(
    error_points,
    error_pred,
    error_target,
    error_uid,
    loader,
    plot_gt_vs_pred_comparison,
):
    _dataset_name, _scene_id, _img_id, _obj_id, _inst_idx = loader.parse_instance_uid(
        error_uid
    )
    plot_gt_vs_pred_comparison(
        loader,
        _scene_id,
        _img_id,
        _inst_idx,
        error_points,
        error_target,
        error_pred,
        title=str(error_uid),
    )
    return


@app.cell
def _(
    FEATURES_INPUT_DIR,
    POINTS_PT_DIR,
    TARGET_DIR,
    error_uid,
    load_instance_npz,
    loader,
    plot_mesh_instance_with_dgedi_features,
):
    _dataset_name, _scene_id, _img_id, _obj_id, _inst_idx = loader.parse_instance_uid(
        error_uid
    )
    _points, _features, _, _ = load_instance_npz(
        error_uid, POINTS_PT_DIR, FEATURES_INPUT_DIR, TARGET_DIR
    )
    plot_mesh_instance_with_dgedi_features(
        loader,
        _scene_id,
        _img_id,
        _inst_idx,
        _points,
        _features,
        percentiles=(0.0, 100.0),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.9 — Same, outlier-robust

    Colors only the non-outlier points (± 2 std by default) — checks whether the error
    pattern is driven by a few extreme points or is general across the instance.
    """)
    return


@app.cell
def _(
    error_points,
    error_pred,
    error_target,
    error_uid,
    loader,
    plot_gt_vs_pred_comparison_robust,
):
    _dataset_name, _scene_id, _img_id, _obj_id, _inst_idx = loader.parse_instance_uid(
        error_uid
    )
    plot_gt_vs_pred_comparison_robust(
        loader,
        _scene_id,
        _img_id,
        _inst_idx,
        error_points,
        error_target,
        error_pred,
        title=str(error_uid),
    )
    return


if __name__ == "__main__":
    app.run()
