import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Inference on unseen objects (YCB-V)

    The models were trained on LMO / PBR (eggbox and glue). This notebook tests them on
    symmetric objects they never saw, from the YCB-V test set (BOP19 subset, scenes 48-59).

    1. The new objects: symmetry metadata, instances by visibility, canonical symmetry field.
    2. Data of `scalarfield_inference`: one instance (target and dGeDi features) and how
       targets / features compare with the training data (`scalarfield_exp3`).
    3. Trained models: runs, per instance metrics and model artifacts from mlflow.

    Paths come from `.env` (`POSE6D_ROOT`, `LMO_ROOT`, `YCBV_ROOT`, `MLFLOW_TRACKING_URI`).

    ### Data processing

    YCB-V is a source of the pT version `rm_outliers_20_2_visib_10` and has its own
    preprocessing version `scalarfield_inference` (same `dgedi_dim` as the training data):

    ``````bash
    uv run scripts/pose6d_prepare_data.py --version-name scalarfield_inference --scenes ycbv:48-59
    # then run the dgedi command printed by the script
    ``````
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import seaborn as sns
    from dotenv import load_dotenv

    load_dotenv()
    import mlflow

    from experiments.analysis import (
        load_instance,
        load_run,
        runs_instances,
        runs_table,
        uid_info,
    )
    from pose6d.dataset import SymmetryFieldDataset
    from pose6d.features import compute_canonical_symmetry_field
    from pose6d.layout import DataLayout
    from pose6d.loader import BOPLoader, loader_from_env
    from pose6d.notebook_plots import (
        plot_frame_meshes_and_sensor,
        plot_mesh_instance_with_dgedi_features,
        plot_mesh_with_scalar_field,
    )

    VERSION = "scalarfield_inference"
    TRAIN_VERSION = "scalarfield_exp3"
    YCBV_SCENES = range(48, 60)
    layout = DataLayout.from_env()
    ycbv = loader_from_env("ycbv")
    loaders = {d: loader_from_env(d) for d in ("lmo", "pbr", "ycbv")}

    # (loader, scene_id, img_id, inst_idx) of a uid, what the mesh plots take
    def frame_args(uid: str):
        dataset, scene_id, img_id, _, inst_idx = BOPLoader.parse_instance_uid(uid)
        return loaders[dataset], scene_id, img_id, inst_idx

    return (
        SymmetryFieldDataset,
        TRAIN_VERSION,
        VERSION,
        YCBV_SCENES,
        compute_canonical_symmetry_field,
        frame_args,
        go,
        layout,
        load_instance,
        load_run,
        mlflow,
        mo,
        np,
        pd,
        plt,
        plot_frame_meshes_and_sensor,
        plot_mesh_instance_with_dgedi_features,
        plot_mesh_with_scalar_field,
        runs_instances,
        runs_table,
        sns,
        uid_info,
        ycbv,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The new symmetric objects

    YCB-V only declares symmetries that hold with texture, so boxes and bottles have no
    symmetry in `models_info.json`. For now we use the objects BOP marks with
    `symmetries_discrete` (bowl 13 and marker 18 only have continuous symmetries, so they
    are left out). As in LMO, the plane is taken from the first discrete symmetry
    (`BOPLoader.load_symmetry_plane`).
    """)
    return


@app.cell
def _(pd, ycbv):
    YCBV_NAMES = {
        1: "002_master_chef_can",
        16: "036_wood_block",
        19: "051_large_clamp",
        20: "052_extra_large_clamp",
        21: "061_foam_brick",
    }
    _info = ycbv.load_models_info()
    obj_ids = sorted(ycbv.symmetric_obj_ids())
    pd.DataFrame(
        [
            {
                "obj_id": o,
                "name": YCBV_NAMES.get(o, "?"),
                "diameter_mm": round(_info[o].diameter, 1),
                "size_xyz_mm": _info[o].size_xyz.round(1),
                "n_discrete_symmetries": len(_info[o].symmetries_discrete),
                "plane_normal": ycbv.load_symmetry_plane(o).normal.numpy().round(3),
            }
            for o in obj_ids
        ]
    )
    return YCBV_NAMES, obj_ids


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Instances by visibility

    Every instance of the symmetric objects in the test scenes (before the pT filters).
    `visib_fract > 0.9` works as the "complete" reference, the rest are occluded clouds.
    """)
    return


@app.cell
def _(YCBV_NAMES, YCBV_SCENES, obj_ids, pd, ycbv):
    instances = pd.DataFrame(
        [
            {
                "scene_id": scene_id,
                "img_id": img_id,
                "inst_idx": inst_idx,
                "obj_id": inst.obj_id,
                "name": YCBV_NAMES.get(inst.obj_id, "?"),
                "visib_fract": inst.visible_fract,
            }
            for scene_id in YCBV_SCENES
            for img_id, inst_idx, inst in ycbv.list_instances(scene_id, set(obj_ids))
        ]
    )
    instances["visib_bin"] = pd.cut(
        instances["visib_fract"], [0, 0.1, 0.3, 0.9, 1.0], include_lowest=True
    )
    instances.groupby(["name", "visib_bin"], observed=False).size().unstack(
        fill_value=0
    )
    return (instances,)


@app.cell
def _(instances, sns):
    _g = sns.displot(
        instances,
        x="visib_fract",
        col="name",
        col_wrap=3,
        bins=20,
        height=2.8,
        aspect=1.3,
    )
    _g.figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Canonical symmetry field

    Distance to the symmetry plane over points sampled on the mesh (model frame). This is
    the field propagated to each pT as target.
    """)
    return


@app.cell
def _(YCBV_NAMES, mo, obj_ids):
    canonical_obj = mo.ui.dropdown(
        {f"{o} {YCBV_NAMES.get(o, '?')}": o for o in obj_ids},
        value=f"{obj_ids[0]} {YCBV_NAMES.get(obj_ids[0], '?')}",
        label="object",
    )
    canonical_obj
    return (canonical_obj,)


@app.cell
def _(canonical_obj, compute_canonical_symmetry_field, go, ycbv):
    _points, _field = compute_canonical_symmetry_field(ycbv, canonical_obj.value, 5000)
    _fig = go.Figure(
        go.Scatter3d(
            x=_points[:, 0],
            y=_points[:, 1],
            z=_points[:, 2],
            mode="markers",
            marker=dict(
                size=2,
                color=_field,
                colorscale="Viridis",
                colorbar=dict(title="symmetry field"),
            ),
        )
    )
    _fig.update_layout(
        title=f"obj {canonical_obj.value}", height=600, scene=dict(aspectmode="data")
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### One frame

    Posed meshes over the sensor point cloud, for the instance at the selected
    `visib_fract` percentile of the object.
    """)
    return


@app.cell
def _(mo):
    frame_visib = mo.ui.slider(0, 100, step=5, value=50, label="visib percentile")
    frame_visib
    return (frame_visib,)


@app.cell
def _(canonical_obj, frame_visib, instances, mo):
    _df = (
        instances[instances["obj_id"] == canonical_obj.value]
        .sort_values("visib_fract")
        .reset_index(drop=True)
    )
    frame_row = _df.loc[round(frame_visib.value / 100 * (len(_df) - 1))]
    mo.md(
        f"scene {frame_row['scene_id']}, img {frame_row['img_id']}, "
        f"inst {frame_row['inst_idx']} — visib_fract {frame_row['visib_fract']:.2f}"
    )
    return (frame_row,)


@app.cell
def _(frame_row, plot_frame_meshes_and_sensor, ycbv):
    plot_frame_meshes_and_sensor(
        ycbv, int(frame_row["scene_id"]), int(frame_row["img_id"])
    )
    return


@app.cell(hide_code=True)
def _(VERSION, mo):
    mo.md(f"""
    ## 2. Data — `{VERSION}`

    Instances that passed the pT filters (`min_visib`, outliers, enough points), by object.
    """)
    return


@app.cell
def _(SymmetryFieldDataset, VERSION, layout, uid_info):
    info = uid_info(SymmetryFieldDataset(layout, VERSION).uids)
    info.groupby(["dataset", "obj_id"]).size().unstack(fill_value=0).add_prefix("obj")
    return (info,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### One instance by visibility

    Instance at the selected `visib_fract` percentile of an object: target symmetry field
    and dGeDi features (PCA to RGB, fitted on the instance itself).
    """)
    return


@app.cell
def _(YCBV_NAMES, info, mo):
    _objs = sorted(info["obj_id"].unique())
    data_obj = mo.ui.dropdown(
        {f"{o} {YCBV_NAMES.get(o, '?')}": int(o) for o in _objs},
        value=f"{_objs[0]} {YCBV_NAMES.get(_objs[0], '?')}",
        label="object",
    )
    data_visib = mo.ui.slider(0, 100, step=5, value=50, label="visib percentile")
    mo.hstack([data_obj, data_visib])
    return data_obj, data_visib


@app.cell
def _(VERSION, data_obj, data_visib, info, layout, load_instance, mo):
    _df = (
        info[info["obj_id"] == data_obj.value]
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
def _(TRAIN_VERSION, VERSION, mo):
    mo.md(f"""
    ### Compared with the training data

    A few instances per object of `{VERSION}` (ycbv) and `{TRAIN_VERSION}` (lmo / pbr,
    eggbox 10 and glue 11), a random subset of points of each one. The models normalize
    inputs and targets with the training statistics, so values far from the training
    range are extrapolation.
    """)
    return


@app.cell
def _(
    SymmetryFieldDataset,
    TRAIN_VERSION,
    VERSION,
    YCBV_NAMES,
    layout,
    load_instance,
    np,
    pd,
):
    _rng = np.random.default_rng(0)
    _n_instances, _n_points = 20, 200

    _frames, _features = [], []
    for _version in (TRAIN_VERSION, VERSION):
        _uids = pd.Series(SymmetryFieldDataset(layout, _version).uids)
        _groups = _uids.groupby(_uids.str.split("_").str[0] + _uids.str.split("_").str[3])
        for _, _group in _groups:
            for _uid in _group.sample(min(_n_instances, len(_group)), random_state=0):
                _points, _feat, _target = load_instance(layout, _version, _uid)
                _idx = _rng.choice(len(_target), _n_points, replace=False)
                _dataset, _obj = _uid.split("_")[0], int(_uid.split("_")[3][3:])
                _name = YCBV_NAMES.get(_obj, "?") if _dataset == "ycbv" else f"lmo_{_obj}"
                _frames.append(
                    pd.DataFrame(
                        {
                            "version": _version,
                            "dataset": _dataset,
                            "object": f"{_dataset} {_obj} {_name}",
                            "uid": _uid,
                            "target": _target[_idx],
                        }
                    )
                )
                _features.append(_feat[_idx])

    compare = pd.concat(_frames, ignore_index=True)
    compare_features = np.concatenate(_features)
    compare.groupby("object")["target"].describe()
    return compare, compare_features


@app.cell
def _(compare, plt, sns):
    _fig, _ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(compare, x="object", y="target", hue="version", ax=_ax)
    _ax.set(title="symmetry field (target) per object", ylabel="distance to plane [mm]")
    _ax.tick_params(axis="x", labelrotation=-30)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    dGeDi features projected on the first two PCA components of the training points.
    If ycbv falls outside the training cloud, the models see inputs they never trained on.
    """)
    return


@app.cell
def _(TRAIN_VERSION, compare, compare_features, np, plt, sns):
    from sklearn.decomposition import PCA

    _train = (compare["version"] == TRAIN_VERSION).to_numpy()
    _pca = PCA(n_components=2).fit(compare_features[_train])
    _proj = _pca.transform(compare_features)
    _df = compare.assign(pc1=_proj[:, 0], pc2=_proj[:, 1])

    _fig, _axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    sns.scatterplot(
        _df[_train], x="pc1", y="pc2", hue="object", s=4, alpha=0.4, ax=_axes[0]
    )
    sns.scatterplot(
        _df[~_train], x="pc1", y="pc2", hue="object", s=4, alpha=0.4, ax=_axes[1]
    )
    _evr = np.round(_pca.explained_variance_ratio_ * 100, 1)
    _axes[0].set(title=f"training (PC1 {_evr[0]}%, PC2 {_evr[1]}%)")
    _axes[1].set(title="ycbv, same projection")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Trained models

    Finished runs of the selected mlflow experiments (best `final_test_loss` first), the
    per instance metrics of their own test sets and the model artifacts. Which runs are
    used for inference is decided later.
    """)
    return


@app.cell
def _(mlflow, mo):
    _names = [e.name for e in mlflow.search_experiments() if e.name != "Default"]
    experiments_select = mo.ui.multiselect(_names, value=_names, label="experiments")
    experiments_select
    return (experiments_select,)


@app.cell
def _(experiments_select, runs_table):
    runs = runs_table(list(experiments_select.value))
    runs
    return (runs,)


@app.cell
def _(mo):
    load_artifacts_button = mo.ui.run_button(label="load metrics and models")
    load_artifacts_button
    return (load_artifacts_button,)


@app.cell
def _(load_artifacts_button, mo, runs, runs_instances):
    mo.stop(not load_artifacts_button.value)
    # per uid metrics (train / val / test) of every run
    run_metrics = runs_instances(runs["run_id"])
    run_metrics.groupby(["run_id", "split"])["rmse"].mean().unstack()
    return


@app.cell
def _(load_artifacts_button, load_run, mo, runs):
    mo.stop(not load_artifacts_button.value)
    # run_id -> RunArtifacts (model, splits, normalizer, instances)
    trained = {run_id: load_run(run_id) for run_id in runs["run_id"]}
    mo.md(f"{len(trained)} models loaded")
    return


if __name__ == "__main__":
    app.run()
