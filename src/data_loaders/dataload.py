from pathlib import Path
from data_loaders.config import DataLoaderConfig
import numpy as np
import torch
# from logger import dataload_logger


def load_labels(processed_dir: str) -> torch.Tensor:
    labels = np.load(Path(processed_dir) / "labels.npy")
    return torch.from_numpy(labels)


def load_batch_point_clouds(
    processed_dir: str,
    k_batches: int | None = None,
    config: DataLoaderConfig = DataLoaderConfig(),
) -> list[torch.Tensor]:
    """
    Loads obj_batchsize_i.parquet from data/processed.
    y retorna una lista de point clouds individuales.

    Returns:
        list[Tensor(N, 3)]
    """
    batch_size = config.batch_size
    if k_batches:
        files = [
            Path(processed_dir) / f"obj{batch_size}_{i}.npy" for i in range(k_batches)
        ]
    else:
        files = sorted(Path(processed_dir).glob(f"obj{batch_size}_*.npy"))

    # dataload_logger.info(f"Found {len(files)} batch files in {processed_dir}")

    point_clouds = []
    for f in files:
        batch = np.load(f)  # (batch_size, P, 3)
        for point_cloud in batch:
            point_clouds.append(torch.from_numpy(point_cloud))

    # dataload_logger.info(f"Loaded {len(point_clouds)} point clouds")

    return point_clouds
