import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Notebook 1 (FM): feature compression, sobre .obj (Feature-Mesh Sampling)
    """)
    return


@app.cell(hide_code=True)
def _():
    from pipelines.backprojected_features import extract_features_fm
    from pipelines.eval_invariance import eval_invariance_features
    from feature_extractor.config import FeatureConfig
    from data_loaders.SimplePointCloudLoader import MeshLoader
    from data_loaders.SimplePointCloudLoader import PointCloudLoader
    from data_loaders.config import DataLoaderConfig
    from transformations.PCA_compresor import compress_features_pca, compress_batch_pca
    from model_wrappers import DINOWrapper
    from ProjPaths import ProjPath
    from pytorch3d.io import IO
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from dotenv import load_dotenv
    from pathlib import Path
    from loguru import logger
    import plotly.graph_objects as go
    import plotly.express as px
    import marimo as mo
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import torch
    import os
    import json
    import sys
    import gc

    #logger.remove()
    #logger.add(sys.stderr, level="WARNING")  # Just show errors :).
    return (
        DINOWrapper,
        DataLoaderConfig,
        FeatureConfig,
        IO,
        MeshLoader,
        PCA,
        Path,
        PointCloudLoader,
        ProjPath,
        StandardScaler,
        compress_batch_pca,
        eval_invariance_features,
        extract_features_fm,
        gc,
        json,
        mo,
        np,
        pd,
        px,
        sns,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Configurations
    """)
    return


@app.cell(hide_code=True)
def _(Path, ProjPath):
    # We use ProjPath as a path resolver. Changes to any path are advised to be made by modifying an .env file in the root of the local repository. See .env-example.
    paths = ProjPath()

    # mesh_gt_path == DATA_GT_PLANE_SYMM_ORI: la misma carpeta que ya se usaba
    # como gt_path en eval_invariance_features, ahora ademas como fuente de
    # los .obj para el pipeline FM.
    mesh_gt_path = paths.gt_plane_symm_ori
    features_gt_path = paths.get_path_feature(
        "features_fm_ori"
    )  # creates a new directoy for this experiments features
    features_gt_pca_path = paths.get_path_feature("features_fm_pca")
    points_gt_path = paths.get_path_feature(
        "points_fm_ori"
    )  # puntos muestreados por FM, necesarios para eval_invariance_features

    # subcarpeta separada para no pisar la cache de invariance del pipeline PC
    invariance_dataset_path = Path(paths.invariance) / "fm"
    invariance_dataset_path.mkdir(parents=True, exist_ok=True)
    return (
        features_gt_path,
        features_gt_pca_path,
        invariance_dataset_path,
        mesh_gt_path,
        paths,
        points_gt_path,
    )


@app.cell(hide_code=True)
def _(
    DINOWrapper,
    DataLoaderConfig,
    FeatureConfig,
    MeshLoader,
    mesh_gt_path,
    torch,
):
    # Initialization of classes.
    dataloader_settings = DataLoaderConfig(batch_size=16)
    feature_settings = FeatureConfig(
        batch_size=1
    )  # max_points no aplica a FM (no se subsamplean vertices)
    dataloader = MeshLoader(config=dataloader_settings, input_path=mesh_gt_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DINOWrapper(device)

    # mismo tamano que max_points=8192 del pipeline PC, para comparabilidad
    NUM_SAMPLES_FM = 8192

    feature_settings.print_summary()
    return (
        NUM_SAMPLES_FM,
        dataloader,
        dataloader_settings,
        device,
        feature_settings,
        model,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PCA Feature extractor Pipeline

    1. Check & load fetures.npy saved in the features directory, run feature extractor pipeline other wise.
    2. Apply PCA to evey feature.npy saved in features directory, then saved them with a "_p" suffix. With p the projected dimension
    3. Save features & a .json file with a dictionary of total explained variance. Both on the features_pca directory.
    """)
    return


@app.cell(hide_code=True)
def _(
    NUM_SAMPLES_FM,
    Path,
    PointCloudLoader,
    compress_batch_pca,
    dataloader,
    device,
    extract_features_fm,
    feature_settings,
    features_gt_path,
    features_gt_pca_path,
    gc,
    json,
    mo,
    model,
    np,
    pd,
    points_gt_path,
    torch,
):
    def extract_base_features_fm(num_samples: int = NUM_SAMPLES_FM) -> None:
        """
        Puebla features_gt_path y points_gt_path a partir de los .obj en
        mesh_gt_path, corriendo el pipeline FM (RM -> sampling baricentrico).
        Analogo al paso que en el pipeline PC ya estaba resuelto (point clouds
        + features pre-cacheadas); aca hay que generarlo primero porque no
        existe un dataset FM previo.

        Guarda points y features con el mismo nombre en carpetas separadas,
        ya que eval_invariance_features necesita ambos para poder emparejar
        puntos simetricos.
        """
        with mo.status.progress_bar(
            total=len(dataloader), title="Extracting FM base features"
        ) as bar:
            cache_flush = 1
            for name, mesh in dataloader:
                features_path = features_gt_path / f"{name}.npy"
                points_path = points_gt_path / f"{name}.npy"

                if features_path.is_file() and points_path.is_file():
                    bar.update(subtitle=f"Cache for {name}, skipping...")
                    continue

                bar.update(subtitle=f"No cache for {name}, extracting FM...")
                points, features = extract_features_fm(
                    mesh.to(device), model, num_samples, feature_settings
                )
                np.save(points_path, points.cpu().numpy())
                np.save(features_path, features.cpu().numpy())

                del mesh, points, features
                cache_flush += 1
                if cache_flush % 8 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

        torch.cuda.empty_cache()

    def extract_features_pca_p_batch(
        p: int,
        feature_loader,
        device=device,
    ):
        explained_variance = {}
        features_gt_pca_path.mkdir(parents=True, exist_ok=True)

        cache_path = features_gt_pca_path / f"exp_var_{p}.json"
        if cache_path.exists():
            return  # ya calculado, no recalcular

        with mo.status.progress_bar(
            total=len(feature_loader), title=f"Compressing batch of features (p={p})"
        ) as bar:
            cache_flush_counter = 0
            for names, batch_features in feature_loader.batches():
                batch_features = batch_features.to(device)  # (K, N, D)
                batch_features_pca, batch_var = compress_batch_pca(batch_features, p)

                for i, name in enumerate(names):
                    out_path = features_gt_pca_path / f"{name}_{p}.npy"
                    np.save(out_path, batch_features_pca[i].cpu().numpy())
                    explained_variance[name] = batch_var[i].item()
                    bar.update(subtitle=f"Saved {name} (p={p})")
                    cache_flush_counter += 1

                del batch_features, batch_features_pca
                if cache_flush_counter >= 8:
                    torch.cuda.empty_cache()
                    gc.collect()
                    cache_flush_counter = 0

        with open(cache_path, "w") as f:
            json.dump(explained_variance, f)
        torch.cuda.empty_cache()

    # Wrapper function to iterate over a list of expected dimensions
    def run_pca_compression_batches(
        ps: list[int], feature_loader: PointCloudLoader, model=model, device=device
    ):
        for p in ps:
            extract_features_pca_p_batch(p, feature_loader, device=device)

    # Create a Pandas DataFrame using all the available .json files.
    def build_explained_variance_dataset(
        ps: list[int], features_gt_pca_path: Path
    ) -> pd.DataFrame:
        explained_variances = []
        for p in ps:
            f = features_gt_pca_path / f"exp_var_{p}.json"
            data = json.loads(f.read_text())  # {hash: var, ...}

            for name, var in data.items():
                explained_variances.append(
                    {"name": name, "p": p, "explained_variance": var}
                )

        return pd.DataFrame(explained_variances)

    return (
        build_explained_variance_dataset,
        extract_base_features_fm,
        run_pca_compression_batches,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### Experiment 1: how do the features degrade as we decrese p?
    """)
    return


@app.cell
def _(
    PointCloudLoader,
    build_explained_variance_dataset,
    dataloader_settings,
    extract_base_features_fm,
    features_gt_path,
    features_gt_pca_path,
    run_pca_compression_batches,
):
    ps = [2, 4, 8, 16, 32, 64, 128, 256]
    extract_base_features_fm()  # puebla features_gt_path y points_gt_path si falta cache
    feature_loader = PointCloudLoader(
        config=dataloader_settings, input_path=features_gt_path
    )
    run_pca_compression_batches(ps, feature_loader)
    df = build_explained_variance_dataset(ps, features_gt_pca_path)
    return df, ps


@app.cell
def _(df):
    df_grouped = df.groupby("p")["explained_variance"].agg(["mean", "std"])
    return


@app.cell(hide_code=True)
def _(df, ps, sns):
    import matplotlib.pyplot as plt

    sns.set_theme(
        style="whitegrid",
        palette="deep",
        font="sans-serif",
        font_scale=1.1,
    )

    plt.rcParams.update(
        {
            "figure.figsize": (8, 5),
            "figure.dpi": 120,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )
    p_values = [2, 4, 8, 16, 32, 64, 128, 256]
    sns.lineplot(data=df, x="p", y="explained_variance", marker="o", errorbar="sd")
    plt.xscale("log")
    plt.xticks(ps, labels=[str(p) for p in ps])
    plt.title("Mean explained variance v.s PCA components")
    plt.xlabel("number of principal components")
    plt.ylabel("Explained Var")
    plt.show()
    return (plt,)


@app.cell(disabled=True, hide_code=True)
def _(PCA, StandardScaler, features_gt_path, np):

    # TODO: verificar que este hash exista en features_gt_path (features_fm_ori)
    # -- viene del pipeline PC y puede no corresponder a un .obj real en
    # curated_plane_sym_obj_ori. Reemplazar por un nombre valido del dataset FM.
    features_real = np.load(
        features_gt_path / "1298634053ad50d36d07c55cf995503e.npy"
    )  # (P, 384)
    P, D = features_real.shape

    # baseline: ruido gaussiano puro, mismas dimensiones
    np.random.seed(0)
    features_random = np.random.randn(P, D)

    features_real_scaled = StandardScaler().fit_transform(features_real)
    features_random_scaled = StandardScaler().fit_transform(features_random)

    for p in [8, 16, 32, 64, 128, 264]:
        pca_real = PCA(n_components=p).fit(features_real_scaled)
        pca_random = PCA(n_components=p).fit(features_random_scaled)

        print(
            f"p={p:>3} | "
            f"real: {pca_real.explained_variance_ratio_.sum():.4f}  |  "
            f"random: {pca_random.explained_variance_ratio_.sum():.4f}"
        )
    return


@app.cell(hide_code=True)
def _(Path, df, json, pd, plt, ps, sns):
    def build_n_planes_dataset(gt_path: Path) -> pd.DataFrame:
        gt = json.loads(gt_path.read_text())  # {hash: [[n,p], ...], ...}

        records = [
            {"hash": hash_, "n_planes": len(planes)} for hash_, planes in gt.items()
        ]

        return pd.DataFrame(records)

    def show_pca_statistics_by_planes(n_planes: int, gt_df):
        hashes = gt_df[gt_df["n_planes"] == n_planes]["hash"]
        p_values = [2, 4, 8, 16, 32, 64, 128, 256]
        sns.lineplot(
            data=df[df["name"].isin(hashes)],
            x="p",
            y="explained_variance",
            marker="o",
            errorbar="sd",
        )
        plt.xscale("log")
        plt.xticks(ps, labels=[str(p) for p in ps])
        plt.title(f"MEV v.s PCA components, {n_planes} planes of symmetry")
        plt.xlabel("number of principal components")
        plt.ylabel("Explained Var")
        plt.show()

    return build_n_planes_dataset, show_pca_statistics_by_planes


@app.cell
def _(gt_df, sns):
    sns.countplot(data=gt_df, x="n_planes")
    return


@app.cell(hide_code=True)
def _(df, gt_df, plt, ps, sns):
    df_merged = df.merge(gt_df[["hash", "n_planes"]], left_on="name", right_on="hash")
    df_filtered = df_merged[df_merged["n_planes"].isin([1, 2])]

    fig, ax = plt.subplots()
    sns.lineplot(
        ax=ax,
        data=df_filtered,
        x="p",
        y="explained_variance",
        hue="n_planes",
        style="n_planes",
        markers=True,
        palette="deep",
        # errorbar=None
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(ps)
    ax.set_xticklabels([str(p) for p in ps])
    ax.set_title("MEV vs PCA components by planes of symmetry")
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Explained variance")
    plt.show()
    return


@app.cell
def _(
    Path,
    build_n_planes_dataset,
    mesh_gt_path,
    show_pca_statistics_by_planes,
):
    # TODO: confirmar que mesh_gt_path (curated_plane_sym_obj_ori) contenga
    # un ground_truth.json con el mismo esquema {hash: [[n,p], ...], ...}
    # que se usaba en data_gt_path (paths.gt_plane_symm) del pipeline PC.
    gt_df = build_n_planes_dataset(Path(mesh_gt_path) / "ground_truth.json")
    show_pca_statistics_by_planes(1, gt_df)
    show_pca_statistics_by_planes(2, gt_df)
    # show_pca_statistics_by_planes(3, gt_df)
    return (gt_df,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Visualize PCA
    """)
    return


@app.cell(hide_code=True)
def _(
    DINOWrapper,
    FeatureConfig,
    IO,
    PCA,
    Path,
    StandardScaler,
    extract_features_fm,
    mesh_gt_path,
    pd,
    px,
    torch,
):
    def example_pipeline_features(
        obj_name, mesh_path: Path, num_samples: int = 8000
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mesh = IO().load_mesh(Path(mesh_gt_path) / obj_name, device=device)
        print(f"vertices: {mesh.verts_packed().shape[0]}")
        model = DINOWrapper(device=device, small=True, reg=True)
        config = FeatureConfig()
        points, features = extract_features_fm(mesh, model, num_samples, config)
        return features, points

    def plot_obj_with_PCA(obj_name, p: int = 32, show_comp: int = 1):
        # feature extractor
        features, pc = example_pipeline_features(
            obj_name, Path(mesh_gt_path) / obj_name, num_samples=8000
        )
        features = features.cpu()
        pc = pc.cpu()
        # PCA
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        pca = PCA(n_components=p)
        pca_features = pca.fit_transform(features_scaled)  # (P, n_comp)
        var_expl = pca.explained_variance_ratio_
        print(f"n comp: {p} explained_var: {var_expl[:8]}")
        print(f"explained: {sum(var_expl)}")
        df_dict = {
            "x": pc[:, 0],
            "y": pc[:, 1],
            "z": pc[:, 2],
        }
        for i in range(p):
            df_dict |= {f"PC{i + 1}": pca_features[:, i]}
        df = pd.DataFrame(df_dict)

        # plotly
        show_comp_ = f"PC{show_comp}"
        fig = px.scatter_3d(
            df,
            x="x",
            y="y",
            z="z",
            color=show_comp_,
            color_continuous_scale="Viridis",
            opacity=0.8,
            title=f"Heat Map: Component {show_comp_}<br>"
            f"(Explained var: {sum(var_expl):.4f})"
            f"(Component var: {var_expl[show_comp - 1]:.4f})",
        )

        fig.update_traces(marker=dict(size=2.5))
        fig.update_layout(
            scene=dict(
                xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"
            ),
            margin=dict(l=0, r=0, b=0, t=50),  #
        )
        return fig

    return (plot_obj_with_PCA,)


@app.cell(hide_code=True)
def _(plot_obj_with_PCA):
    # TODO: confirmar que este hash exista como .obj en curated_plane_sym_obj_ori
    # (viene del dataset PC, puede no coincidir 1:1 con los nombres del dataset mesh)
    obj_name = "15cb1696b45ef647dcad484e89744ca.obj"
    ex1_1 = plot_obj_with_PCA(obj_name, p=8)
    ex1_2 = plot_obj_with_PCA(obj_name, p=8, show_comp=2)
    ex1_16 = plot_obj_with_PCA(obj_name, p=3, show_comp=3)
    ex1_32 = plot_obj_with_PCA(obj_name, show_comp=32)
    return ex1_1, ex1_16, ex1_2, ex1_32


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Visualize components of an example object

    - components 1, 2, 16, 32
    """)
    return


@app.cell
def _(ex1_1):
    ex1_1.show()
    return


@app.cell
def _(ex1_2):
    ex1_2.show()
    return


@app.cell
def _(ex1_16):
    ex1_16.show()
    return


@app.cell
def _(ex1_32):
    ex1_32.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Visualizing a 3 plane symmetric object
    """)
    return


@app.cell(hide_code=True)
def _(plot_obj_with_PCA):
    # TODO: mismo caveat que obj_name -- verificar existencia en el dataset mesh
    obj_name_3 = "83412e29d5978b101f6dfedaba98d5f9.obj"
    ex3_1 = plot_obj_with_PCA(obj_name_3)
    ex3_2 = plot_obj_with_PCA(obj_name_3, show_comp=2)
    ex3_16 = plot_obj_with_PCA(obj_name_3, show_comp=16)
    ex3_32 = plot_obj_with_PCA(obj_name_3, show_comp=32)
    return ex3_1, ex3_16, ex3_2, ex3_32


@app.cell
def _(ex3_1):
    ex3_1.show()
    return


@app.cell
def _(ex3_2):
    ex3_2.show()
    return


@app.cell
def _(ex3_16):
    ex3_16.show()
    return


@app.cell
def _(ex3_32):
    ex3_32.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Experiment 2: how does symmetry invariance degrade as we decrease p?

    For each PCA dimension p, we compute the mean L1 distance between features
    of geometrically paired points (reflected across the ground truth symmetry plane).
    Lower is better — a perfect invariant extractor would score 0.
    We also include p=original (no PCA) as the baseline.
    """)
    return


@app.cell
def _(
    Path,
    eval_invariance_features,
    feature_settings,
    features_gt_path,
    features_gt_pca_path,
    invariance_dataset_path,
    json,
    mo,
    paths,
    pd,
    points_gt_path,
    ps,
):
    def run_invariance_eval(ps: list[int], filter_by=None) -> None:
        # features originales (sin PCA) como baseline
        original_cache = Path(invariance_dataset_path) / "inv_original.json"

        if not original_cache.exists():
            with mo.status.spinner(title="Computing invariance (original features)"):
                df_ori = eval_invariance_features(
                    config=feature_settings,
                    point_cloud_path=points_gt_path,
                    feature_path=features_gt_path,
                    gt_path=paths.gt_plane_symm_ori,
                    model=None,
                    filter_by=None,
                )
                result = {
                    row["name"]: row["invariance"] for _, row in df_ori.iterrows()
                }
                original_cache.write_text(json.dumps(result))

        # features PCA por dimension p
        for p in ps:
            cache = Path(invariance_dataset_path) / f"inv_{p}.json"
            if cache.exists():
                continue
            with mo.status.spinner(title=f"Computing invariance (p={p})"):
                df_p = eval_invariance_features(
                    config=feature_settings,
                    point_cloud_path=points_gt_path,
                    feature_path=features_gt_pca_path,
                    gt_path=paths.gt_plane_symm_ori,
                    model=None,
                    filter_by=f"_{p}",
                )
                result = {row["name"]: row["invariance"] for _, row in df_p.iterrows()}
                cache.write_text(json.dumps(result))

    def build_invariance_dataset(ps: list[int], invariance_path: Path) -> pd.DataFrame:
        """
        Lee los .json cacheados y construye un DataFrame con columnas:
            name, p, invariance
        Incluye p=original como referencia baseline.
        """
        records = []

        # baseline: features originales sin PCA
        original_cache = invariance_path / "inv_original.json"
        data = json.loads(original_cache.read_text())
        for name, inv in data.items():
            records.append(
                {"name": name, "p": "original", "p_num": 384, "invariance": inv}
            )

        # features PCA
        for p in ps:
            cache = invariance_path / f"inv_{p}.json"
            data = json.loads(cache.read_text())
            for name, inv in data.items():
                records.append(
                    {"name": name, "p": str(p), "p_num": p, "invariance": inv}
                )

        return pd.DataFrame(records)

    run_invariance_eval(ps)
    df_inv = build_invariance_dataset(ps, Path(invariance_dataset_path))
    return (df_inv,)


@app.cell(hide_code=True)
def _(df_inv, plt, ps, sns):
    df_inv["invariance"] = df_inv["invariance"] / df_inv["p_num"]
    # orden del eje x: original primero, luego p creciente
    p_order = ["ori"] + [str(p) for p in sorted(ps)]
    p_num_order = [384] + sorted(ps)  # para el eje x numerico

    sns.set_theme(
        style="whitegrid",
        palette="deep",
        font="sans-serif",
        font_scale=1.1,
    )
    plt.rcParams.update(
        {
            "figure.figsize": (8, 5),
            "figure.dpi": 120,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )

    sns.lineplot(
        data=df_inv,
        x="p_num",
        y="invariance",
        marker="o",
        errorbar="sd",
    )
    plt.xscale("log")
    plt.xticks(p_num_order, labels=p_order)
    plt.axvline(
        x=384, color="gray", linestyle="--", linewidth=0.8, label="original (no PCA)"
    )
    plt.legend()
    plt.title("Mean symmetry invariance vs PCA components")
    plt.xlabel("Number of principal components")
    plt.ylabel("L1 invariance distance (lower = better)")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
