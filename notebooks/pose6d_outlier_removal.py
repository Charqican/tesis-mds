import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Outlier Removal (LMO)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Statistical outlier removal on the partial point clouds (pT), tuned interactively
    with sliders and compared against the cached `rm_outliers_*` version.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Config
    """)
    return


@app.cell
def _():
    import marimo as mo
    from pose6d.loader import LMOLoader, PBRLoader
    from pose6d.notebook_plots import (
        plot_mesh_with_scalar_field,
        plot_mesh_instance_with_dgedi_features,
        plot_pcd_outlier_mask,
    )
    import numpy as np
    import open3d as o3d
    from pathlib import Path

    lmo_root = Path("/mnt/data/dev/dataset/tesis/BOP/lmo/lmo")
    pbr_root = Path("/mnt/data/dev/dataset/tesis/BOP/pbr/lm_train_pbr/train_pbr/")

    lmo_loader = LMOLoader.from_root(lmo_root)
    pbr_loader = PBRLoader.from_roots(pbr_root, lmo_root)  # reusa los mismos modelos de LMO

    lmo_config = lmo_loader.cfg
    pbr_config = pbr_loader.cfg

    ROOT = Path("/mnt/data/dev/dataset/tesis/6dpose")
    POINTS_PT_DIR = lambda dataset: ROOT / f"{dataset}/cache/rm_outliers_20_2/points_pT/"
    FEATURES_INPUT_DIR = lambda dataset: ROOT / f"{dataset}/scalarfield_rm/training/input/"
    TARGET_DIR = lambda dataset: ROOT / f"{dataset}/scalarfield_rm/training/target/"
    return (
        FEATURES_INPUT_DIR,
        POINTS_PT_DIR,
        ROOT,
        TARGET_DIR,
        mo,
        np,
        o3d,
        pbr_loader,
        plot_mesh_instance_with_dgedi_features,
        plot_mesh_with_scalar_field,
        plot_pcd_outlier_mask,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load an example instance (before removal)
    """)
    return


@app.cell
def _(FEATURES_INPUT_DIR, POINTS_PT_DIR, TARGET_DIR, np, pbr_loader):
    loader = pbr_loader
    example_uid = "pbr_scene000002_img000003_obj000010_inst01"
    example_uid_file = example_uid + ".npz"
    _, scene_id, img_id, obj_id, inst_idx = loader.parse_instance_uid(example_uid)
    test = np.load(POINTS_PT_DIR(loader.dataset_name) / example_uid_file)["points"]
    symmetry_field_1 = np.load(TARGET_DIR(loader.dataset_name) / example_uid_file)["target"]
    features_dgedi = np.load(FEATURES_INPUT_DIR(loader.dataset_name) / example_uid_file)["features"]
    return (
        example_uid,
        example_uid_file,
        features_dgedi,
        img_id,
        inst_idx,
        loader,
        scene_id,
        symmetry_field_1,
        test,
    )


@app.cell
def _(
    img_id,
    inst_idx,
    loader,
    plot_mesh_with_scalar_field,
    scene_id,
    symmetry_field_1,
    test,
):
    plot_mesh_with_scalar_field(
        loader,
        scene_id,
        img_id,
        inst_idx,
        test,
        symmetry_field_1,
        colorbar_title="Symmetry field",
    )
    return


@app.cell
def _(
    features_dgedi,
    img_id,
    inst_idx,
    loader,
    plot_mesh_instance_with_dgedi_features,
    scene_id,
    test,
):
    plot_mesh_instance_with_dgedi_features(
        loader,
        scene_id,
        img_id,
        inst_idx,
        test,
        features_dgedi,
        percentiles=(1.0, 100.0),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Outlier Removal - Geometric statistics
    """)
    return


@app.cell
def _(mo):
    nb_neighbors_slider = mo.ui.slider(
        start=5, stop=50, step=1, value=20, label="nb_neighbors"
    )
    std_ratio_slider = mo.ui.slider(
        start=0.5, stop=5.0, step=0.1, value=2.0, label="std_ratio"
    )
    mo.hstack([nb_neighbors_slider, std_ratio_slider])
    return nb_neighbors_slider, std_ratio_slider


@app.cell
def _(
    example_uid,
    nb_neighbors_slider,
    np,
    o3d,
    plot_pcd_outlier_mask,
    std_ratio_slider,
    test,
):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(test)

    pcd_clean, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors_slider.value,
        std_ratio=std_ratio_slider.value,
    )
    ind_idx = np.asarray(ind)
    outlier_mask = np.ones(test.shape[0], dtype=bool)
    outlier_mask[ind_idx] = False
    print(
        f"Removing {outlier_mask.sum()} points. Ratio: {outlier_mask.sum() / test.shape[0]}"
    )
    plot_pcd_outlier_mask(test, outlier_mask, example_uid)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compare against the cached `rm_outliers_*` version
    """)
    return


@app.cell
def _(ROOT, example_uid_file, np):
    POINTS_PT_DIR_RM = ROOT / "lmo/cache/rm_outliers_20_2/points_pT/"
    FEATURES_INPUT_DIR_RM = ROOT / "lmo/scalarfield_rm/training/input/"
    TARGET_DIR_RM = ROOT / "lmo/scalarfield_rm/training/target/"
    test_rm = np.load(POINTS_PT_DIR_RM / example_uid_file)["points"]
    symmetry_field_rm = np.load(TARGET_DIR_RM / example_uid_file)["target"]
    features_dgedi_rm = np.load(FEATURES_INPUT_DIR_RM / example_uid_file)["features"]
    return features_dgedi_rm, symmetry_field_rm, test_rm


@app.cell
def _(
    img_id,
    inst_idx,
    loader,
    plot_mesh_with_scalar_field,
    scene_id,
    symmetry_field_rm,
    test_rm,
):
    plot_mesh_with_scalar_field(
        loader,
        scene_id,
        img_id,
        inst_idx,
        test_rm,
        symmetry_field_rm,
        colorbar_title="Symmetry field",
    )
    return


@app.cell
def _(
    features_dgedi_rm,
    img_id,
    inst_idx,
    loader,
    plot_mesh_instance_with_dgedi_features,
    scene_id,
    test_rm,
):
    plot_mesh_instance_with_dgedi_features(
        loader,
        scene_id,
        img_id,
        inst_idx,
        test_rm,
        features_dgedi_rm,
        percentiles=(1.0, 100.0),
    )
    return


if __name__ == "__main__":
    app.run()
