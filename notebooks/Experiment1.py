import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Notebook 1 : feature compression
    """)
    return


@app.cell(hide_code=True)
def _():
    from pipelines.backprojected_features import extract_features
    from feature_extractor.config import FeatureConfig
    from feature_extractor.sampling import sample_fibonacci_views
    from feature_extractor.rendering import render_point_cloud
    from feature_extractor.backprojection import aggregate_features
    from data_loaders.SimplePointCloudLoader import PointCloudLoader
    from data_loaders.config import DataLoaderConfig
    from transformations.PCA_compresor import compress_features_pca
    from model_wrappers import DINOWrapper
    from ProjPaths import ProjPath
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

    logger.remove()
    logger.add(sys.stderr, level="WARNING")  # Just show errors :).
    return (
        DINOWrapper,
        DataLoaderConfig,
        FeatureConfig,
        PCA,
        Path,
        PointCloudLoader,
        ProjPath,
        StandardScaler,
        aggregate_features,
        compress_features_pca,
        extract_features,
        gc,
        json,
        mo,
        np,
        pd,
        px,
        render_point_cloud,
        sample_fibonacci_views,
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
def _(ProjPath):
    # We use ProjPath as a path resolver. Changes to any path are advised to be made by modifying an .env file in the root of the local repository. See .env-example.
    paths = ProjPath()
    data_gt_path = paths.gt_plane_symm
    features_gt_path = paths.get_path_feature(
        "plane_symm"
    )  # creates a new directoy for this experiments features
    features_gt_pca_path = paths.get_path_feature("plane_symm_pca")
    return data_gt_path, features_gt_path, features_gt_pca_path


@app.cell(hide_code=True)
def _(
    DINOWrapper,
    DataLoaderConfig,
    FeatureConfig,
    PointCloudLoader,
    data_gt_path,
    torch,
):
    # Initialization of classes.
    dataloader_settings = DataLoaderConfig(processed_dir=data_gt_path)
    feature_settings = FeatureConfig(batch_size=1)
    dataloader = PointCloudLoader(config=dataloader_settings)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DINOWrapper(device)

    feature_settings.print_summary()
    return dataloader, device, feature_settings, model


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
    Path,
    compress_features_pca,
    dataloader,
    device,
    extract_features,
    feature_settings,
    features_gt_path,
    features_gt_pca_path,
    gc,
    json,
    mo,
    model,
    np,
    pd,
    torch,
):
    def extract_features_pca_p(p: int, model=model, device=device):
        explained_variance = {}
        features_gt_pca_path.mkdir(parents=True, exist_ok=True)
        with mo.status.progress_bar(
            total=len(dataloader), title=f"Compressing features (p={p})"
        ) as bar:
            cache_flush = 1
            for name, pc in dataloader:
                pc = pc.to(device)
                # Load or extract features from the objec
                features_path = features_gt_path / f"{name}.npy"
                if features_path.is_file():
                    features: torch.Tensor = torch.from_numpy(np.load(features_path))
                    bar.update(subtitle=f"Cache for {name}, loading...")

                else:
                    bar.update(
                        subtitle=f"No cached features for {name}, extracting & saving..."
                    )
                    features: torch.Tensor = extract_features(
                        pc, model, feature_settings
                    )
                    np.save(features_path, features.cpu().numpy())
                features_pca_p_path = features_gt_pca_path / f"{name}_{p}.npy"
                features_pca_p, explained_var = compress_features_pca(features, p)
                explained_variance[name] = explained_var
                np.save(features_pca_p_path, features_pca_p.cpu().numpy())
                del pc
                del features
                del features_pca_p
                cache_flush += 1
                if cache_flush % 32 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

        with open(features_gt_pca_path / f"exp_var_{p}.json", "w") as f:
            json.dump(explained_variance, f)
        torch.cuda.empty_cache()

    # Wrapper function to iterate over a list of expected dimensions
    def run_pca_compression(ps: list[int], model=model, device=device) -> None:
        for p in ps:
            extract_features_pca_p(p, model=model, device=device)

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

    return build_explained_variance_dataset, run_pca_compression


@app.cell
def _(mo):
    mo.md(r"""
    ### Experiment 1: how do the features degrade as we decrese p?
    """)
    return


@app.cell
def _(
    build_explained_variance_dataset,
    features_gt_pca_path,
    run_pca_compression,
):
    ps = [2, 4, 8, 16, 32, 64, 128, 256]
    run_pca_compression(ps)
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

    # features reales de un objeto
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
def _(gt_df):
    gt_df["n_planes"].value_counts()
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
    data_gt_path,
    show_pca_statistics_by_planes,
):
    gt_df = build_n_planes_dataset(Path(data_gt_path) / "ground_truth.json")
    show_pca_statistics_by_planes(1, gt_df)
    show_pca_statistics_by_planes(2, gt_df)
    show_pca_statistics_by_planes(3, gt_df)
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
    PCA,
    Path,
    StandardScaler,
    aggregate_features,
    data_gt_path,
    np,
    pd,
    px,
    render_point_cloud,
    sample_fibonacci_views,
    torch,
):
    def _subsample(point_cloud: torch.Tensor, max_points: int) -> torch.Tensor:
        """
        Subsamplea aleatoriamente la nube a max_points puntos.
        Seed fija para reproducibilidad.
        """
        if len(point_cloud) <= max_points:
            return point_cloud

        np.random.seed(0)
        indices = torch.from_numpy(
            np.random.choice(len(point_cloud), max_points, replace=False)
        )
        return point_cloud[indices]

    def example_pipeline_features(
        obj_name, pc_path: Path, max_point: int = 1000
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pc = torch.from_numpy(np.load(Path(data_gt_path) / obj_name))
        pc_subsampled = _subsample(pc, max_point)
        print(pc_subsampled.shape)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pc_subsampled = pc_subsampled.to(device)
        model = DINOWrapper(device=device, small=True, reg=True)
        config = FeatureConfig(max_points=max_point)
        # Fibonacci sampling views & camera
        R, T = sample_fibonacci_views(pc_subsampled, config)
        rendered_images, mappings = render_point_cloud(pc_subsampled, R, T, config)
        _render = rendered_images
        model_outputs = model(rendered_images)
        features = aggregate_features(
            model_outputs, mappings, pc_subsampled, FeatureConfig()
        )
        return features, pc_subsampled

    def plot_obj_with_PCA(obj_name, p: int = 32, show_comp: int = 1):
        # feature extractor
        features, pc = example_pipeline_features(
            obj_name, Path(data_gt_path) / obj_name, max_point=4000
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

        fig.update_traces(marker=dict(size=2))
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
    obj_name = "895563d304772f50ad5067eac75a07f7.npy"
    ex1_1 = plot_obj_with_PCA(obj_name)
    ex1_2 = plot_obj_with_PCA(obj_name, show_comp=2)
    ex1_16 = plot_obj_with_PCA(obj_name, show_comp=16)
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
    obj_name_3 = "83412e29d5978b101f6dfedaba98d5f983412e29d5978b101f6dfedaba98d5f9.npy"
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


if __name__ == "__main__":
    app.run()
