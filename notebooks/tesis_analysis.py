import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import json

    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import seaborn as sns
    from dotenv import load_dotenv

    load_dotenv()
    import mlflow

    from experiments.analysis import load_instance, load_run, predict_instance, runs_instances, runs_table, uid_info
    from pose6d.layout import DataLayout, load_symm_configs
    from pose6d.loader import BOPLoader, loader_from_env
    from pose6d import notebook_plots as nplots
    from pose6d.features import compute_canonical_symmetry_field

    return (
        BOPLoader,
        DataLayout,
        compute_canonical_symmetry_field,
        go,
        json,
        load_instance,
        load_run,
        load_symm_configs,
        loader_from_env,
        mlflow,
        mo,
        np,
        nplots,
        pd,
        plt,
        predict_instance,
        runs_instances,
        runs_table,
        sns,
        uid_info,
    )


@app.cell
def _(mo):
    # outputs read by the typst report (doc/informes/inf1/main.typ)
    REPORT_DIR = mo.notebook_dir().parent / "doc" / "informes" / "inf1"
    CSV_DIR = REPORT_DIR / "data"
    IMG_DIR = REPORT_DIR / "img"
    return CSV_DIR, IMG_DIR


@app.cell
def _():
    return


@app.cell
def _():
    # experiment names are built in src/experiments/registry.py as "<split>_obj<ids>[_norm|_normdiam]",
    # target: raw (mm), norm (divided by the pT diameter) or normdiam (divided by the object diameter)
    PARSE = {
        "split": lambda name: name.split("_obj")[0],
        "objs": lambda name: name.split("_obj")[1].split("_")[0],
        "target": lambda name: name.split("_obj")[1].partition("_")[2] or "raw",
        "normalized": lambda name: name.endswith(("_norm", "_normdiam")),
    }
    SPLITS = {
        "lmo-strat": {"data": "lmo-real", "train": "lmo", "test": "lmo"},
        "pbr-train_lmo-test": {"data": "lmo-pbr", "train": "pbr", "test": "lmo"},
    }
    EXCLUDED = ["Default", "pbr-train_lmo-test_server-check"]  # server tests
    return EXCLUDED, PARSE, SPLITS


@app.cell
def _(DataLayout, load_symm_configs, loader_from_env):
    layout = DataLayout.from_env()
    versions = load_symm_configs()
    loaders = {dataset: loader_from_env(dataset) for dataset in ("lmo", "pbr")}
    return (layout,)


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Dataframes

    - `experiments` : this dataframe summarizes every experiment runned.
    - `runs` : this dataframe expands every configuration runned.
    - `best` : show the best configurations in therms of validation loss.
    - 'instances' : show metrics for every instance uid and every run.
    - `poc` : proof of concept of `available_symm`, per test instance metrics of eggbox models trained on sets filtered by it.
    """)
    return


@app.cell
def _(EXCLUDED, PARSE, SPLITS, mlflow, pd):
    experiments = pd.DataFrame(
        {"experiment": [e.name for e in mlflow.search_experiments() if e.name not in EXCLUDED]}
    )

    for column, parse in PARSE.items():
        experiments[column] = experiments["experiment"].map(parse)
    experiments = experiments.join(pd.DataFrame(SPLITS).T, on="split")
    experiments["label"] = (
        experiments["data"] + " obj" + experiments["objs"] + experiments["target"].map({"raw": "", "norm": " norm", "normdiam": " normdiam"})
    )
    experiments = experiments.sort_values(["split", "normalized", "target", "objs"], ignore_index=True)
    experiments
    return (experiments,)


@app.cell
def _(experiments, runs_table):
    # Some of the experiments were run multiple time, those are excluded.
    runs = (
        runs_table(experiments["experiment"].tolist())
        .sort_values("start_time")
        .drop_duplicates(["experiment", "run_name"], keep="last")
        .merge(experiments, on="experiment")
    )
    # raw runs log the test rmse in mm, normalized runs log it in diameters and denormalized as test_rmse_mm_mean
    runs["test_rmse_mm"] = runs["test_rmse_mm_mean"].fillna(runs["test_rmse_mean"].where(~runs["normalized"]))
    runs
    return (runs,)


@app.cell
def _(experiments, runs):
    # model selection on validation, test is only reported
    best = runs.sort_values("final_val_loss").drop_duplicates("experiment").set_index("experiment").loc[experiments["experiment"]]
    best
    return (best,)


@app.cell
def _(experiments, runs):
    # best run on test, to see how much the val selection differs
    best_by_test = runs.sort_values("final_test_loss").drop_duplicates("experiment").set_index("experiment").loc[experiments["experiment"]]
    best_by_test
    return (best_by_test,)


@app.cell
def _(best, runs_instances):
    # per instance metrics (instances.parquet) of the best runs
    instances = runs_instances(best["run_id"]).merge(
        best.reset_index()[["run_id", "experiment", "label", "normalized"]], on="run_id"
    )
    # older raw runs did not log rmse_mm, their rmse is already in mm
    instances["rmse_mm"] = instances["rmse_mm"].fillna(instances["rmse"].where(~instances["normalized"]))
    instances
    return (instances,)


@app.cell
def _(json, mo):
    # lmo-strat split. Every experiment tests on lmo, its test uids are the subset all of them share
    LMO_SPLIT = json.loads((mo.notebook_dir().parent / "configs/splits/scalarfield_exp3/lmo-strat.json").read_text())
    LMO_TEST_UIDS = LMO_SPLIT["test"]
    return LMO_SPLIT, LMO_TEST_UIDS


@app.cell
def _(LMO_SPLIT, layout, np):
    # raw targets (in mm) of the lmo-strat train and test uids, used by the constant baselines
    lmo_targets = {
        uid: np.load(layout.target_dir("scalarfield_exp3") / f"{uid}.npz")["target"]
        for uid in LMO_SPLIT["train"] + LMO_SPLIT["test"]
    }
    return (lmo_targets,)


@app.cell
def _(BOPLoader, LMO_SPLIT, LMO_TEST_UIDS, instances, lmo_targets, np, pd):
    # Is the model better than predicting a constant? Evaluating vs:
    #   - train mean: mean of every train point of the object
    #   - instance mean: mean of the instance itsel (rmse = std))
    #   - the best (val) run of every experiment
    obj_of = lambda uid: BOPLoader.parse_instance_uid(uid)[3]
    train_mean = {
        obj_id: np.concatenate([lmo_targets[uid] for uid in LMO_SPLIT["train"] if obj_of(uid) == obj_id]).mean()
        for obj_id in (10, 11)
    }
    constants = pd.DataFrame(
        [
            {"uid": uid, "obj_id": obj_of(uid), "predictor": name, "rmse_mm": rmse}
            for uid in LMO_TEST_UIDS
            for name, rmse in [
                ("train mean", np.sqrt(np.mean((lmo_targets[uid] - train_mean[obj_of(uid)]) ** 2))),
                ("instance mean", lmo_targets[uid].std()),
            ]
        ]
    )
    # extract the shared instances metrics
    models = instances[(instances["split"] == "test") & instances["uid"].isin(LMO_TEST_UIDS)] 
    baselines = pd.concat(
        [constants, models[["uid", "obj_id", "label", "rmse_mm"]].rename(columns={"label": "predictor"})],
        ignore_index=True,
    )
    baselines.groupby(["obj_id", "predictor"])["rmse_mm"].mean().unstack(0).round(2)
    return (baselines,)


@app.cell
def _(LMO_TEST_UIDS, baselines, pd, uid_info):
    # baselines (constants and best runs) with the visibility of the instance (visib_fract of BOP gt_info)
    visibility = baselines.merge(uid_info(LMO_TEST_UIDS)[["uid", "visib_fract"]], on="uid")
    visibility["visib_bin"] = pd.cut(
        visibility["visib_fract"], [0.25, 0.5, 0.7, 0.9, 1.0], labels=["0.25-0.5", "0.5-0.7", "0.7-0.9", "0.9-1"]
    )
    visibility
    return (visibility,)


@app.cell
def _(layout, lmo_targets, pd):
    # Proof of concept (trained in notebooks/symmetry_plane_analysis.py, not in mlflow): eggbox models, one
    # configuration (best val of lmo-strat_obj10), trained on lmo-strat train sets of the same size sampled
    # above an available_symm threshold. Val and test are the lmo-strat ones. One row per (train set, test uid).
    # available_symm: fraction of the pT points (outside a band around the plane) whose reflection is in the pT
    poc = pd.read_parquet(layout.version_dir("scalarfield_exp3") / "poc_available_symm_obj10.parquet")
    poc["symm_bin"] = pd.cut(
        poc["available_symm"], [0, 0.2, 0.4, 0.6, 1.0], include_lowest=True, labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-1"]
    )
    # constant oracle: the instance mean, its rmse is the std of the target
    poc["oracle_rmse_mm"] = poc["uid"].map(lambda uid: lmo_targets[uid].std())
    poc
    return (poc,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Report tables
    """)
    return


@app.cell
def _(LMO_TEST_UIDS, pd, runs, runs_instances):
    def results_table(selected: pd.DataFrame) -> pd.DataFrame:
        """One row per experiment with the per instance metrics of its selected run (indexed by experiment)."""

        inst = runs_instances(selected["run_id"]).merge(
                   runs[["run_id", "experiment", "normalized"]], on="run_id"
                )
        # older raw runs did not log rmse_mm, their rmse is already in mm
        inst["rmse_mm"] = inst["rmse_mm"].fillna(inst["rmse"].where(~inst["normalized"]))
        own_test = inst[inst["split"] == "test"].groupby("experiment")
        same_test = inst[(inst["split"] == "test") & inst["uid"].isin(LMO_TEST_UIDS)].groupby("experiment")
        table = pd.DataFrame(
            {
                "Experiment": selected["label"],
                "Own test set": own_test.size(),
                "Configs": runs.groupby("experiment").size(),
                "RMSE own test [mm]": own_test["rmse_mm"].mean(),
                "Median RMSE [mm]": own_test["rmse_mm"].median(),
                "Pearson same test": same_test["pearson"].mean(),
                "RMSE same test [mm]": same_test["rmse_mm"].mean(),
            }
        )
        return table.loc[selected.index]

    return (results_table,)


@app.cell
def _(best, results_table):
    results_best_val = results_table(best)
    results_best_val
    return (results_best_val,)


@app.cell
def _(best_by_test, results_table):
    results_best_test = results_table(best_by_test)
    results_best_test
    return (results_best_test,)


@app.cell
def _(CSV_DIR, results_best_test, results_best_val):
    # float_format keeps the trailing zeros (10.10, not 10.1) in the typst tables
    results_best_val.to_csv(CSV_DIR / "results_best_val.csv", index=False, float_format="%.2f")
    results_best_test.to_csv(CSV_DIR / "results_best_test.csv", index=False, float_format="%.2f")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Report figures
    """)
    return


@app.cell
def _(sns):
    # shared look of the report figures: seaborn theme with recessive grid, colors by role
    sns.set_theme(
        context="paper",
        style="whitegrid",
        rc={"grid.color": "#ebeae6", "axes.edgecolor": "#c9c8c1", "hatch.color": "white", "savefig.bbox": "tight"},
    )
    COLORS = {"constant": "#c9c8c1", "oracle": "#9b9a94", "lmo-pbr": "#eb6834", "lmo-real": "#2a78d6", "normdiam": "#1baf7a", "ink": "#0b0b0b"}
    OBJECT_NAMES = {10: "obj 10 (eggbox)", 11: "obj 11 (glue)"}
    return COLORS, OBJECT_NAMES


@app.cell
def _(COLORS, IMG_DIR, OBJECT_NAMES, baselines, pd, sns):
    # per object: constant baselines vs the best (val) run of every experiment that tests on it.
    # One run per experiment, seaborn averages its rmse over the lmo-strat test instances of the object.
    # color = training data, hatch = normalized target (/// pT diameter, ... object diameter)
    _bars = {  # bar name: (predictor in baselines, color, hatch)
        "train mean": ("train mean", COLORS["constant"], ""),
        "instance mean": ("instance mean", COLORS["oracle"], ""),
        "lmo-real, this obj": ("lmo-real obj{}", COLORS["lmo-real"], ""),
        "lmo-real, this obj, norm": ("lmo-real obj{} norm", COLORS["lmo-real"], "///"),
        "lmo-real, this obj, normdiam": ("lmo-real obj{} normdiam", COLORS["lmo-real"], "..."),
        "lmo-real, obj10-11": ("lmo-real obj10-11", COLORS["lmo-real"], ""),
        "lmo-real, obj10-11, norm": ("lmo-real obj10-11 norm", COLORS["lmo-real"], "///"),
        "lmo-real, obj10-11, normdiam": ("lmo-real obj10-11 normdiam", COLORS["lmo-real"], "..."),
        "lmo-pbr, this obj": ("lmo-pbr obj{}", COLORS["lmo-pbr"], ""),
        "lmo-pbr, this obj, norm": ("lmo-pbr obj{} norm", COLORS["lmo-pbr"], "///"),
        "lmo-pbr, obj10-11": ("lmo-pbr obj10-11", COLORS["lmo-pbr"], ""),
        "lmo-pbr, obj10-11, norm": ("lmo-pbr obj10-11 norm", COLORS["lmo-pbr"], "///"),
    }
    _plot = pd.concat(
        [
            baselines[(baselines["obj_id"] == _obj_id) & (baselines["predictor"] == predictor.format(_obj_id))].assign(
                bar=name, obj=OBJECT_NAMES[_obj_id]
            )
            for _obj_id in OBJECT_NAMES
            for name, (predictor, _, _) in _bars.items()
        ]
    )
    _grid = sns.catplot(
        data=_plot,
        kind="bar",
        x="rmse_mm",
        y="bar",
        col="obj",
        hue="bar",
        order=list(_bars),
        hue_order=list(_bars),
        palette={name: color for name, (_, color, _) in _bars.items()},
        errorbar=None,
        saturation=1,  # seaborn desaturates bars by default
        sharex=False,
        height=3.6,
        aspect=1.0,
    )
    for _ax in _grid.axes.flat:
        for _container, (_, _, _hatch) in zip(_ax.containers, _bars.values()):
            for _bar in _container:
                _bar.set_hatch(_hatch)
            _ax.bar_label(_container, fmt="%.1f", padding=2, fontsize=7)
        _ax.grid(axis="y", visible=False)
    _grid.set_titles("{col_name}")
    _grid.set_axis_labels("mean RMSE per instance [mm]", "")
    fig_baselines = _grid.figure
    fig_baselines.savefig(IMG_DIR / "baselines.svg")
    fig_baselines
    return


@app.cell
def _(
    COLORS,
    IMG_DIR,
    experiments,
    pd,
    plt,
    results_best_test,
    results_best_val,
    sns,
):
    # How much does selecting on validation cost? Same test rmse of the best val run vs the best test run.
    # seaborn has no dumbbell plot, so this one is plain matplotlib with the seaborn theme
    _selection = pd.DataFrame(
        {
            "Experiment": results_best_val["Experiment"],
            "data": experiments.set_index("experiment")["data"],
            "objs": experiments.set_index("experiment")["objs"],
            "val": results_best_val["RMSE same test [mm]"],
            "test": results_best_test["RMSE same test [mm]"],
        }
    )
    _markers = {"10": "o", "11": "^", "10-11": "s"}  # objects the experiment trains and tests on
    fig_selection, _ax = plt.subplots(figsize=(6.0, 4.0))
    for _yi, _row in enumerate(_selection.itertuples()):
        _color = COLORS[_row.data]
        _ax.hlines(_yi, _row.test, _row.val, color=_color, linewidth=2, alpha=0.5)
        _marker = _markers[_row.objs]
        _ax.scatter(_row.val, _yi, s=36, marker=_marker, color=_color, zorder=3)
        _ax.scatter(_row.test, _yi, s=36, marker=_marker, facecolor="white", edgecolor=_color, linewidth=1.5, zorder=3)
        if _row.val - _row.test > 0.5:
            _ax.text(_row.val + 0.3, _yi, f"-{_row.val - _row.test:.1f}", va="center", fontsize=7, color="#52514e")
    _ax.set_yticks(range(len(_selection)), _selection["Experiment"])
    _ax.invert_yaxis()
    _ax.grid(axis="y", linestyle="--")
    _ax.set_xlabel("mean RMSE per instance on the lmo-strat test instances [mm]")
    _ax.legend(
        handles=[
            plt.Line2D([], [], marker="o", ls="", color=COLORS["lmo-real"], label="lmo-real"),
            plt.Line2D([], [], marker="o", ls="", color=COLORS["lmo-pbr"], label="lmo-pbr"),
            *[plt.Line2D([], [], marker=_m, ls="", color="#52514e", label=f"obj {_objs}") for _objs, _m in _markers.items()],
            plt.Line2D([], [], marker="o", ls="", color="#52514e", label="best validation run"),
            plt.Line2D([], [], marker="o", ls="", markerfacecolor="white", color="#52514e", label="best test run"),
        ],
        frameon=False,
        fontsize=7,
        loc="upper right",
    )
    sns.despine(ax=_ax)
    fig_selection.savefig(IMG_DIR / "selection.svg")
    fig_selection
    return


@app.cell
def _(COLORS, IMG_DIR, OBJECT_NAMES, experiments, sns, visibility):
    # Does the error grow when less of the object is visible? Mean RMSE per visibility bin, one axis
    # per object with every model tested on it (own and obj10-11).
    # color = training data, line = objects the model was trained on, gray = constant oracle
    _plot = visibility.merge(experiments, left_on="predictor", right_on="label").query("target != 'norm'")
    _plot["training data"] = _plot["data"] + _plot["target"].map({"raw": "", "normdiam": " normdiam"})
    _plot["trained on"] = "obj " + _plot["objs"]
    _plot["obj"] = _plot["obj_id"].map(OBJECT_NAMES)

    _grid = sns.relplot(
        data=_plot,
        kind="line",
        x="visib_bin",
        y="rmse_mm",
        col="obj",
        hue="training data",
        style="trained on",
        palette={"lmo-real": COLORS["lmo-real"], "lmo-pbr": COLORS["lmo-pbr"], "lmo-real normdiam": COLORS["normdiam"]},
        hue_order=["lmo-real", "lmo-pbr", "lmo-real normdiam"],
        style_order=["obj 10", "obj 11", "obj 10-11"],
        dashes={"obj 10": "", "obj 11": (1, 2), "obj 10-11": (4, 2)},
        markers={"obj 10": "o", "obj 11": "^", "obj 10-11": "s"},
        errorbar=None,
        height=2.7,  # ~ the page width, so typst does not scale the text down
        aspect=1.15,
        facet_kws={"sharex": False, "sharey": False},  # each axis has its own counts per bin
    )

    # oracle line and instances per bin, per object
    _oracle = visibility[visibility["predictor"] == "instance mean"].groupby(["obj_id", "visib_bin"], observed=True)["rmse_mm"]
    for _obj_id, _ax in zip(OBJECT_NAMES, _grid.axes.flat):
        _ax.plot(_oracle.mean()[_obj_id].index, _oracle.mean()[_obj_id], color=COLORS["oracle"])
        _ax.text(3, _oracle.mean()[_obj_id].iloc[-1], " oracle", va="center", fontsize=7, color="#52514e")
        _ax.set_xticks(range(4), [f"{_bin}\nn={_n}" for _bin, _n in _oracle.size()[_obj_id].items()])

    # legend below the axes, one row per group
    _handles, _labels = _grid.legend.legend_handles, [_t.get_text() for _t in _grid.legend.texts]
    _grid.legend.remove()
    _order = [0, 4, 1, 5, 2, 6, 3, 7]  # [training data, 3 colors, trained on, 3 lines] -> two rows
    _grid.figure.legend(
        [_handles[i] for i in _order], [_labels[i] for i in _order],
        loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=4, frameon=False, fontsize=7,
    )
    _grid.set_titles("{col_name}")
    _grid.set_axis_labels("visible fraction", "mean RMSE per instance [mm]")
    fig_visibility = _grid.figure
    fig_visibility.savefig(IMG_DIR / "visibility.svg")
    fig_visibility
    return


@app.cell
def _(COLORS, IMG_DIR, plt, poc, sns):
    # Does training on instances with more available symmetry help? Mean RMSE per available_symm bin of the
    # test, one line per train set (same size, darker = higher threshold), gray = constant oracle
    _order = ["all", ">10%", ">20%", ">30%"]
    fig_available_symm, _ax = plt.subplots(figsize=(3.6, 2.7))
    sns.lineplot(
        data=poc,
        x="symm_bin",
        y="rmse_mm",
        hue="train_set",
        hue_order=_order,
        palette=sns.light_palette(COLORS["lmo-real"], n_colors=len(_order) + 2)[2:],
        marker="o",
        errorbar=None,
        ax=_ax,
    )

    # oracle line and test instances per bin (every train set has the same test)
    _oracle = poc[poc["train_set"] == "all"].groupby("symm_bin", observed=True)["oracle_rmse_mm"]
    _ax.plot(range(len(_oracle.mean())), _oracle.mean(), color=COLORS["oracle"])
    _ax.text(len(_oracle.mean()) - 1, _oracle.mean().iloc[-1], " oracle", va="center", fontsize=7, color="#52514e")
    _ax.set_xticks(range(len(_oracle.size())), [f"{_bin}\nn={_n}" for _bin, _n in _oracle.size().items()])

    _ax.set(xlabel="available symmetry of the test instance", ylabel="mean RMSE per instance [mm]", title="obj 10 (eggbox)")
    sns.move_legend(_ax, "center right", title="train set", frameon=False, fontsize=7, title_fontsize=7)
    sns.despine(ax=_ax)
    fig_available_symm.savefig(IMG_DIR / "available_symm.svg")
    fig_available_symm
    return


@app.cell
def _(IMG_DIR, experiments, plt, runs, sns):
    # Does the configuration matter? Test RMSE (own test set) of every run, one row per experiment.
    # The vertical position inside a row is random jitter, color = model family (MLP5 joins its 3 dropouts)
    _plot = runs.assign(family=runs["model"].str.removeprefix("SymmetryField").str.split("-").str[0])
    _ranges = _plot.groupby("label")["test_rmse_mm"].agg(["size", "min", "max"]).loc[experiments["label"]]

    fig_hparams, _ax = plt.subplots(figsize=(5.6, 4.4))
    sns.stripplot(
        data=_plot,
        x="test_rmse_mm",
        y="label",
        hue="family",
        order=experiments["label"],
        hue_order=["MLP1", "MLP4", "MLP5"],
        palette=["#2a78d6", "#eb6834", "#1baf7a"],
        jitter=0.25,
        size=3,
        alpha=0.75,
        ax=_ax,
    )
    for _y, _row in enumerate(_ranges.itertuples()):
        _ax.text(_row.max + 0.2, _y, f"n={_row.size}, range {_row.max - _row.min:.2f}", va="center", fontsize=6.5, color="#52514e")
    _ax.set_xlim(right=_plot["test_rmse_mm"].max() + 3.5)  # room for the labels
    _ax.set(xlabel="test RMSE of the run, own test set [mm]", ylabel="")
    _ax.grid(axis="y", visible=False)
    sns.move_legend(_ax, "upper right", title="model", frameon=False, fontsize=7, title_fontsize=7)
    sns.despine(ax=_ax)
    fig_hparams.savefig(IMG_DIR / "hparams.svg")
    fig_hparams
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Unseen objects (YCB-V)
    """)
    return


@app.cell
def _(loader_from_env):
    # YCB-V test (BOP19, scenes 48-59): the objects BOP marks with discrete symmetries
    YCBV_NAMES = {1: "master chef can", 16: "wood block", 19: "large clamp", 20: "extra large clamp", 21: "foam brick"}
    ycbv_loader = loader_from_env("ycbv")
    # inference version of each target recipe (same pT and target as the training version, ycbv instances)
    YCBV_VERSIONS = {"raw": "scalarfield_inference", "norm": "scalarfield_inference_norm", "normdiam": "scalarfield_inference_normdiam"}
    return YCBV_NAMES, YCBV_VERSIONS, ycbv_loader


@app.cell
def _(mo):
    # camera of every 3D figure of this section. The download icon of plotly exports the camera of
    # the layout, not the one rotated with the mouse, so the view is set here
    ycbv_azimuth = mo.ui.slider(0, 345, step=15, value=45, label="azimuth [deg]")
    ycbv_elevation = mo.ui.slider(-75, 75, step=15, value=30, label="elevation [deg]")
    ycbv_distance = mo.ui.slider(0.5, 3.0, step=0.25, value=2.0, label="distance")
    mo.hstack([ycbv_azimuth, ycbv_elevation, ycbv_distance], justify="start")
    return ycbv_azimuth, ycbv_distance, ycbv_elevation


@app.cell
def _(go, mo, np, ycbv_azimuth, ycbv_distance, ycbv_elevation):
    def ycbv_view(fig: go.Figure, width: int = 550, height: int = 500) -> mo.ui.plotly:
        """Camera of the sliders for a 3D figure. The figure fits the notebook, the download icon
        exports it at width x height (x2), the same size for every image."""
        az, el = np.radians(ycbv_azimuth.value), np.radians(ycbv_elevation.value)
        r = ycbv_distance.value
        eye = dict(x=r * np.cos(el) * np.cos(az), y=r * np.cos(el) * np.sin(az), z=r * np.sin(el))
        fig.update_scenes(camera=dict(eye=eye))
        # a new uirevision makes plotly drop the camera rotated with the mouse and take this one
        fig.update_layout(height=height, uirevision=f"{az}-{el}-{r}")
        return mo.ui.plotly(fig, config={"toImageButtonOptions": {"width": width, "height": height, "scale": 2}})

    return (ycbv_view,)


@app.cell
def _(YCBV_NAMES, np, pd, ycbv_loader):
    # BOP symmetries are rotations, the target plane is taken from the first one
    # (BOPLoader.load_symmetry_plane). rotations: angle of every discrete symmetry
    _angle = lambda M: round(float(np.degrees(np.arccos(np.clip((np.trace(M[:3, :3]) - 1) / 2, -1, 1)))))
    _info = ycbv_loader.load_models_info()
    ycbv_objects = pd.DataFrame(
        [
            {
                "obj_id": obj_id,
                "object": name,
                "diameter [mm]": round(_info[obj_id].diameter, 1),
                "size xyz [mm]": _info[obj_id].size_xyz.round(1),
                "rotations [deg]": [_angle(M) for M in _info[obj_id].symmetries_discrete],
                "plane normal": ycbv_loader.load_symmetry_plane(obj_id).normal.numpy().round(3),
            }
            for obj_id, name in YCBV_NAMES.items()
        ]
    )
    ycbv_objects
    return


@app.cell
def _(YCBV_NAMES, mo):
    ycbv_obj = mo.ui.dropdown({f"{o} {n}": o for o, n in YCBV_NAMES.items()}, value="1 master chef can", label="object")
    ycbv_obj
    return (ycbv_obj,)


@app.cell
def _(nplots, ycbv_loader, ycbv_obj, ycbv_view):
    # mesh in its model frame with the x/y/z axes (red/green/blue)
    ycbv_view(nplots.plot_model_mesh(ycbv_loader, ycbv_obj.value))
    return


@app.cell
def _(compute_canonical_symmetry_field, go, ycbv_loader, ycbv_obj, ycbv_view):
    # the target: distance to the symmetry plane over points sampled on the mesh (0 = on the plane)
    _points, _field = compute_canonical_symmetry_field(ycbv_loader, ycbv_obj.value, 5000)
    _field = _field.ravel()  # (N, 1) -> (N,), plotly needs a flat color array
    _canonical = go.Figure(
        go.Scatter3d(
            x=_points[:, 0], y=_points[:, 1], z=_points[:, 2],
            mode="markers",
            marker=dict(size=2, color=_field, colorscale="Viridis", colorbar=dict(title="distance [mm]")),
        )
    )
    _canonical.update_layout(title="target (canonical field)", scene=dict(aspectmode="data"))
    ycbv_view(_canonical)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### One instance: target and dGeDi features
    """)
    return


@app.cell
def _(YCBV_VERSIONS, layout, uid_info):
    # ycbv instances that passed the pT filters, with their visibility
    ycbv_instances = uid_info([p.stem for p in sorted(layout.input_dir(YCBV_VERSIONS["raw"]).glob("ycbv_*.npz"))])
    ycbv_instances.groupby("obj_id").size()
    return (ycbv_instances,)


@app.cell
def _(mo, ycbv_obj):
    ycbv_visib = mo.ui.slider(0, 100, step=5, value=50, label="visib percentile")
    mo.hstack([ycbv_obj, ycbv_visib], justify="start")
    return (ycbv_visib,)


@app.cell
def _(
    YCBV_VERSIONS,
    layout,
    load_instance,
    mo,
    ycbv_instances,
    ycbv_loader,
    ycbv_obj,
    ycbv_visib,
):
    # instance of the object at the selected visib_fract percentile
    _df = ycbv_instances[ycbv_instances["obj_id"] == ycbv_obj.value].sort_values("visib_fract", ignore_index=True)
    _row = _df.loc[round(ycbv_visib.value / 100 * (len(_df) - 1))]
    ycbv_uid = _row["uid"]
    ycbv_frame = (ycbv_loader, _row["scene_id"], _row["img_id"], _row["inst_idx"])  # what the mesh plots take
    ycbv_points, ycbv_features, ycbv_target = load_instance(layout, YCBV_VERSIONS["raw"], ycbv_uid)
    mo.md(f"`{ycbv_uid}`, visib_fract {_row['visib_fract']:.2f}")
    return ycbv_features, ycbv_frame, ycbv_points, ycbv_target, ycbv_uid


@app.cell
def _(nplots, ycbv_frame, ycbv_points, ycbv_target, ycbv_view):
    # target on the visible points (posed mesh in orange)
    ycbv_view(nplots.plot_mesh_with_scalar_field(*ycbv_frame, ycbv_points, ycbv_target, colorbar_title="target [mm]"))
    return


@app.cell
def _(nplots, ycbv_features, ycbv_frame, ycbv_points, ycbv_view):
    # dGeDi features of the same points (PCA to RGB)
    ycbv_view(nplots.plot_mesh_instance_with_dgedi_features(*ycbv_frame, ycbv_points, ycbv_features, percentiles=(1.0, 100.0)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Inference

    The best (val) run of every experiment predicts on the selected instance, with the
    normalizer of its training data. Each model uses the inference version with its own
    target recipe (raw, norm or normdiam), everything is shown in mm.
    """)
    return


@app.cell
def _(mo):
    ycbv_button = mo.ui.run_button(label="load the best runs")
    ycbv_button
    return (ycbv_button,)


@app.cell
def _(best, load_run, mo, ycbv_button):
    mo.stop(not ycbv_button.value)
    # label -> RunArtifacts (model, normalizer, ...) of the best (val) run of every experiment
    trained = {label: load_run(run_id) for label, run_id in best.set_index("label")["run_id"].items()}
    mo.md(f"{len(trained)} models loaded")
    return (trained,)


@app.cell
def _(best, mo, ycbv_obj, ycbv_visib):
    _labels = best.groupby("data")["label"].apply(list)
    ycbv_real = mo.ui.dropdown(_labels["lmo-real"], value="lmo-real obj10-11", label="lmo-real model")
    ycbv_pbr = mo.ui.dropdown(_labels["lmo-pbr"], value="lmo-pbr obj10-11", label="lmo-pbr model")
    mo.hstack([ycbv_obj, ycbv_visib, ycbv_real, ycbv_pbr], justify="start")
    return ycbv_pbr, ycbv_real


@app.cell
def _(
    YCBV_VERSIONS,
    best,
    layout,
    np,
    nplots,
    predict_instance,
    trained,
    ycbv_frame,
    ycbv_uid,
    ycbv_view,
):
    def ycbv_compare(label: str):
        """target vs prediction of one model on the selected instance, in mm."""
        version = YCBV_VERSIONS[best.set_index("label").loc[label, "target"]]
        points, target, pred = predict_instance(trained[label], layout, ycbv_uid, version)
        # target * scale is in mm, targets saved before the normalization have no scale (mm already)
        npz = np.load(layout.target_dir(version) / f"{ycbv_uid}.npz")
        scale = float(npz["scale"]) if "scale" in npz else 1.0
        target, pred = target.ravel() * scale, pred.ravel() * scale
        pearson = np.corrcoef(target, pred)[0, 1]
        rmse = np.sqrt(np.mean((target - pred) ** 2))
        title = f"{label}: pearson {pearson:.2f}, rmse {rmse:.1f} mm (oracle {target.std():.1f} mm)"
        fig = nplots.plot_gt_vs_pred_comparison(*ycbv_frame, points, target, pred, title=title)
        return ycbv_view(fig, width=1100)  # two panels


    return (ycbv_compare,)


@app.cell
def _(ycbv_compare, ycbv_real):
    ycbv_compare(ycbv_real.value)
    return


@app.cell
def _(ycbv_compare, ycbv_pbr):
    ycbv_compare(ycbv_pbr.value)
    return


if __name__ == "__main__":
    app.run()
