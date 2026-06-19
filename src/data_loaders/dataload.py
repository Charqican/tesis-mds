from pathlib import Path
from data_loaders.config import DataLoaderConfig
import numpy as np
import torch
# from logger import dataload_logger


def load_labels(processed_dir: str) -> torch.Tensor:
    labels = np.load(Path(processed_dir) / "labels.npy")
    return torch.from_numpy(labels)


def load_batch_point_clouds(
    k_batches: int | None = None,
    config: DataLoaderConfig = DataLoaderConfig(),
) -> torch.Tensor:
    """
    Loads obj_batchsize_i.parquet from data/processed.
    y retorna una lista de point clouds individuales.

    Returns:
        list[Tensor(N, 3)]
    """
    batch_size = config.batch_size
    processed_dir = config.processed_dir
    if k_batches:
        files = [
            Path(processed_dir) / f"obj{batch_size}_{i}.npy" for i in range(k_batches)
        ]
    else:
        files = sorted(Path(processed_dir).glob(f"obj{batch_size}_*.npy"))

    # dataload_logger.info(f"Found {len(files)} batch files in {processed_dir}")
    point_clouds_list = []

    for f in files:
        batch = np.load(f)  # (batch_size, P, 3)
        # Convertimos todo el batch de numpy a un tensor de una vez
        point_clouds_list.append(torch.from_numpy(batch))

    # Apilamos todas las listas de tensores en un solo tensor de gran tamaño
    # Esto resultará en un tensor de forma (Total_N, P, 3)
    point_clouds_tensor = torch.cat(point_clouds_list, dim=0)

    # dataload_logger.info(f"Loaded {len(point_clouds)} point clouds")
    return point_clouds_tensor
