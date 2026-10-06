import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data and experiments analysis

    1. Data of one preprocessing version: counts, visibility, one instance.
    2. Experiments: every config of the selected mlflow experiments (min / max).
    3. One run: loss curves and per instance errors (`instances.parquet`).
    4. One instance of that run: per point errors on the mesh.

    Paths come from `.env` (`POSE6D_ROOT`, `LMO_ROOT`, `PBR_ROOT`, `MLFLOW_TRACKING_URI`).
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from dotenv import load_dotenv

    load_dotenv()
    import mlflow
    from mlflow.tracking import MlflowClient

    from experiments.analysis import (
        load_instance,
        load_run,
        predict_instance,
        runs_instances,
        runs_table,
        uid_info,
    )
    from pose6d.dataset import SymmetryFieldDataset
    from pose6d.layout import DataLayout
    from pose6d.loader import BOPLoader, loader_from_env
    from pose6d.notebook_plots import (
        plot_gt_vs_pred_comparison,
        plot_gt_vs_pred_comparison_robust,
        plot_mesh_instance_with_dgedi_features,
        plot_mesh_with_scalar_field,
    )

    VERSION = "scalarfield_exp3"
    layout = DataLayout.from_env()
    loaders = {d: loader_from_env(d) for d in ("lmo", "pbr")}

    # (loader, scene_id, img_id, inst_idx) of a uid, what the mesh plots take
    def frame_args(uid: str):
        dataset, scene_id, img_id, _, inst_idx = BOPLoader.parse_instance_uid(uid)
        return loaders[dataset], scene_id, img_id, inst_idx

    return (
        MlflowClient,
        SymmetryFieldDataset,
        VERSION,
        frame_args,
        layout,
        load_instance,
        load_run,
        mlflow,
        mo,
        pd,
        plot_gt_vs_pred_comparison,
        plot_mesh_instance_with_dgedi_features,
        plot_mesh_with_scalar_field,
        plt,
        predict_instance,
        runs_instances,
        runs_table,
        sns,
        uid_info,
    )


@app.cell(hide_code=True)
def _(VERSION, mo):
    mo.md(f"""
    ## 1. Data — `{VERSION}`

    Every local instance of the version, by dataset and object.
    """)
    return


@app.cell
def _(SymmetryFieldDataset, VERSION, layout, uid_info):
    info = uid_info(SymmetryFieldDataset(layout, VERSION).uids)
    info.groupby(["dataset", "obj_id"]).size().unstack(fill_value=0).add_prefix("obj")
    return (info,)


@app.cell
def _(info, sns):
    _g = sns.displot(
        info.astype({"obj_id": str}),
        x="visib_fract",
        hue="obj_id",
        col="dataset",
        bins=30,
        stat="density",
        common_norm=False,
        height=3.5,
        aspect=1.4,
    )
    _g.figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### One instance by visibility

    Instance at the selected `visib_fract` percentile of a dataset and object: target
    symmetry field and dGeDi features (PCA to RGB).
    """)
    return


@app.cell
def _(info, mo):
    data_dataset = mo.ui.dropdown(
        sorted(info["dataset"].unique()), value="lmo", label="dataset"
    )
    data_obj = mo.ui.dropdown(
        [str(o) for o in sorted(info["obj_id"].unique())],
        value=str(info["obj_id"].min()),
        label="object",
    )
    data_visib = mo.ui.slider(0, 100, step=5, value=50, label="visib percentile")
    mo.hstack([data_dataset, data_obj, data_visib])
    return data_dataset, data_obj, data_visib


@app.cell
def _(
    VERSION,
    data_dataset,
    data_obj,
    data_visib,
    info,
    layout,
    load_instance,
    mo,
):
    _df = (
        info[
            (info["dataset"] == data_dataset.value)
            & (info["obj_id"] == int(data_obj.value))
        ]
        .sort_values("visib_fract")
        .reset_index(drop=True)
    )
    _row = _df.loc[round(data_visib.value / 100 * (len(_df) - 1))]
    data_uid = _row["uid"]
    data_points, data_features, data_target = load_instance(layout, VERSION, data_uid)
    mo.md(f"`{data_uid}` — visib_fract {_row['visib_fract']:.2f}")
    return data_features, data_points, data_target, data_uid


@app.cell
def _(
    data_points,
    data_target,
    data_uid,
    frame_args,
    plot_mesh_with_scalar_field,
):
    plot_mesh_with_scalar_field(
        *frame_args(data_uid),
        data_points,
        data_target,
        colorbar_title="symmetry field",
    )
    return


@app.cell
def _(
    data_features,
    data_points,
    data_uid,
    frame_args,
    plot_mesh_instance_with_dgedi_features,
):
    plot_mesh_instance_with_dgedi_features(
        *frame_args(data_uid), data_points, data_features, percentiles=(1.0, 100.0)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Experiments

    Finished runs of the selected mlflow experiments, best `final_test_loss` first.
    `final_test_loss` is normalized (comparable with the curves), `test_rmse_mean` is in
    target units.
    """)
    return


@app.cell
def _(mlflow, mo):
    _names = [
        e.name for e in mlflow.search_experiments() if e.name != "Default"
    ]  # default comes with mlflow.
    experiments_select = mo.ui.multiselect(_names, value=_names, label="experiments")
    experiments_select
    return (experiments_select,)


@app.cell
def _(experiments_select, runs_table):
    runs = runs_table(list(experiments_select.value))
    runs
    return (runs,)


@app.cell
def _(runs):
    runs.groupby("experiment").agg(
        n_runs=("run_id", "size"),
        loss_min=("final_test_loss", "min"),
        loss_max=("final_test_loss", "max"),
        rmse_mean_min=("test_rmse_mean", "min"),
        rmse_mean_max=("test_rmse_mean", "max"),
        best_run=("run_name", "first"),
        worst_run=("run_name", "last"),
    )
    return


@app.cell
def _(plt, runs, sns):
    _fig, _ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(runs, x="experiment", y="final_test_loss", hue="model", ax=_ax)
    _ax.set(title="final test loss by experiment and model")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Per instance test RMSE of every run

    Downloads `instances.parquet` of each run. Best / worst instance and mean RMSE per
    object (e.g. to see if obj 10 changes between the obj 10 and obj 10+11 experiments).
    """)
    return


@app.cell
def _(mo):
    load_instances_button = mo.ui.run_button(label="load per instance metrics")
    load_instances_button
    return (load_instances_button,)


@app.cell
def _(load_instances_button, mo, runs, runs_instances):
    mo.stop(not load_instances_button.value)
    test_instances = runs_instances(runs["run_id"]).query("split == 'test'")

    _per_run = test_instances.groupby("run_id")["rmse"].agg(
        rmse_min="min", rmse_max="max", rmse_std="std"
    )
    _per_obj = (
        test_instances.groupby(["run_id", "obj_id"])["rmse"]
        .mean()
        .unstack()
        .add_prefix("rmse_obj")
    )
    runs[
        ["experiment", "run_name", "final_test_loss", "test_rmse_mean", "run_id"]
    ].merge(_per_run, on="run_id").merge(_per_obj, on="run_id")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. One run
    """)
    return


@app.cell
def _(mo, runs):
    _options = {
        f"{r.experiment} / {r.run_name} ({r.run_id[:8]})": r.run_id
        for r in runs.itertuples()
    }
    run_dropdown = mo.ui.dropdown(_options, value=next(iter(_options)), label="run")
    run_dropdown
    return (run_dropdown,)


@app.cell
def _(load_run, run_dropdown):
    run = load_run(run_dropdown.value)
    run.run_name
    return (run,)


@app.cell
def _(MlflowClient, pd, plt, run, sns):
    _client = MlflowClient()
    _df = pd.DataFrame(
        [
            {"epoch": m.step, "loss": m.value, "split": name}
            for name in ("train", "val")
            for m in _client.get_metric_history(run.run_id, f"{name}_loss")
        ]
    )
    _fig, _ax = plt.subplots(figsize=(7, 4))
    sns.lineplot(_df, x="epoch", y="loss", hue="split", ax=_ax)
    _ax.axvline(float(run.params.get("train.min_epochs", 0)), color="gray", ls="--")
    _ax.set(title=run.run_name, yscale="log")
    _fig
    return


@app.cell
def _(plt, run, sns):
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    for _ax, _split in zip(_axes, ("train", "val", "test")):
        _rmse = run.instances.loc[run.instances["split"] == _split, "rmse"]
        sns.histplot(_rmse, bins=30, ax=_ax)
        _ax.set(
            title=f"{_split} - per instance RMSE (mean {_rmse.mean():.2f}, std {_rmse.std():.2f})"
        )
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(plt, run, sns):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(
        run.instances.astype({"obj_id": str}),
        x="split",
        y="rmse",
        hue="obj_id",
        ax=_ax,
    )
    _ax.set(title="per instance RMSE by object")
    _fig
    return


@app.cell
def _(run, sns):
    _g = sns.relplot(
        run.instances,
        x="visib_fract",
        y="rmse",
        hue="split",
        col="obj_id",
        alpha=0.6,
        height=3.5,
        aspect=1.3,
    )
    _g.figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. One instance of the run

    Best / median / worst instance of a split by RMSE (it has to be on local disk).
    """)
    return


@app.cell
def _(mo):
    split_dropdown = mo.ui.dropdown(
        ["test", "val", "train"], value="test", label="split"
    )
    rank_dropdown = mo.ui.dropdown(
        ["best", "median", "worst"], value="worst", label="rank"
    )
    mo.hstack([split_dropdown, rank_dropdown])
    return rank_dropdown, split_dropdown


@app.cell
def _(layout, mo, predict_instance, rank_dropdown, run, split_dropdown):
    _df = (
        run.instances[run.instances["split"] == split_dropdown.value]
        .sort_values("rmse")
        .reset_index(drop=True)
    )
    _row = _df.loc[
        {"best": 0, "median": len(_df) // 2, "worst": len(_df) - 1}[rank_dropdown.value]
    ]
    error_uid = _row["uid"]
    error_points, error_target, error_pred = predict_instance(run, layout, error_uid)
    mo.md(
        f"`{error_uid}` — rmse {_row['rmse']:.3f}, visib_fract {_row['visib_fract']:.2f}"
    )
    return error_points, error_pred, error_target, error_uid


@app.cell
def _(error_pred, error_target, error_uid, plt, sns):
    _fig, _ax = plt.subplots(figsize=(7, 4))
    sns.histplot(error_pred - error_target, bins=30, ax=_ax)
    _ax.set(title=f"{error_uid} - per point error", xlabel="pred - target")
    _fig
    return


@app.cell
def _(
    error_points,
    error_pred,
    error_target,
    error_uid,
    frame_args,
    plot_mesh_with_scalar_field,
):
    plot_mesh_with_scalar_field(
        *frame_args(error_uid),
        error_points,
        error_pred - error_target,
        colorscale="RdBu",
        colorbar_title="pred - target",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Target and prediction with a shared colorbar, then the same without the error
    outliers (± 2 std, grayed out) to see if a few points drive the error.
    """)
    return


@app.cell
def _(
    error_points,
    error_pred,
    error_target,
    error_uid,
    frame_args,
    plot_gt_vs_pred_comparison,
):
    plot_gt_vs_pred_comparison(
        *frame_args(error_uid), error_points, error_target, error_pred, title=error_uid
    )
    return


@app.cell
def _(
    error_uid,
    frame_args,
    layout,
    load_instance,
    plot_mesh_instance_with_dgedi_features,
    run,
):
    _points, _features, _ = load_instance(layout, run.params["data.version"], error_uid)
    plot_mesh_instance_with_dgedi_features(
        *frame_args(error_uid), _points, _features, percentiles=(0.0, 100.0)
    )
    return


if __name__ == "__main__":
    app.run()
