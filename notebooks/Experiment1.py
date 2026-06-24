import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    from data_loaders.SimplePointCloudLoader import PointCloudLoader
    from data_loaders.config import DataLoaderConfig
    from feature_extractor.feature_pipeline import extract_features
    from feature_extractor.config import FeatureConfig
    from transformations.PCA_compresor import compress_features_pca
    from model_wrappers import DINOWrapper
    from dotenv import load_dotenv
    from pathlib import Path
    import numpy as np
    import torch
    import os
    import json
    import sys
    import gc
    from loguru import logger
    from rich.progress import Progress
    from rich.console import Console
    import marimo as mo
    console = Console()
    logger.remove()  # quita el handler default
    logger.add(sys.stderr, level="WARNING")  # solo INFO y superior (oculta DEBUG)
    return (
        DINOWrapper,
        DataLoaderConfig,
        FeatureConfig,
        Path,
        PointCloudLoader,
        compress_features_pca,
        extract_features,
        gc,
        json,
        load_dotenv,
        mo,
        np,
        os,
        torch,
    )


@app.cell
def _(Path, load_dotenv, os):
    # Load .env file with the paths
    load_dotenv()
    data_gt_path = os.environ["DATA_SYM_PLANE_PROCESSED"]
    features_gt_path = Path(os.environ["FEATURES_SYM"])
    features_gt_pca_path = Path(os.environ["FEATURES_SYM_PCA"])
    return data_gt_path, features_gt_path, features_gt_pca_path


@app.cell
def _(
    DINOWrapper,
    DataLoaderConfig,
    FeatureConfig,
    PointCloudLoader,
    data_gt_path,
    torch,
):
    # Initialization of classes.
    dataloader_settings=DataLoaderConfig(processed_dir=data_gt_path)
    feature_settings=FeatureConfig(batch_size=1)
    dataloader = PointCloudLoader(config=dataloader_settings)
    print(len(dataloader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DINOWrapper(device)
    return dataloader, device, feature_settings, model


@app.cell
def _(
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
    torch,
):
    # PCA feature_extractor : Takes pointclouds and returns the features proyected onto a lower dimension p. 
    # (hashName : str, point_cloud : torch.Tensor(N, 3) -> features_pca_p : torch.Tensor(K), var_exp : float)
    def extract_features_pca_p(p : int, model = model, device = device): 
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
                    features : torch.Tensor = torch.from_numpy(
                        np.load(features_path)
                    )
                    bar.update(
                    subtitle=f"Cache for {name}, loading..."
                    )
                
                else: 
                    bar.update(
                    subtitle=f"No cached features for {name}, extracting & saving..."
                    )
                    features : torch.Tensor = extract_features(
                        pc, model, feature_settings
                    )
                    np.save(
                        features_path, features.cpu().numpy()
                    )
                features_pca_p_path = features_gt_pca_path / f"{name}_{p}.npy"
                if features_pca_p_path.is_file():
                    continue
                features_pca_p, explained_var = compress_features_pca(
                    features, p
                )
                explained_variance[name] = explained_var
                np.save(features_pca_p_path, features_pca_p.cpu().numpy())
                del pc
                del features
                del features_pca_p
                cache_flush+=1
                if cache_flush%32 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

        with open(features_gt_pca_path / f"exp_var_{p}.json", "w") as f:
            json.dump(explained_variance, f)
        torch.cuda.empty_cache()


    return (extract_features_pca_p,)


@app.cell
def _(extract_features_pca_p):
    extract_features_pca_p(256)
    return


if __name__ == "__main__":
    app.run()
