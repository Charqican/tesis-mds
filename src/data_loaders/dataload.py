import json
from pathlib import Path

import numpy as np
import torch

from data_loaders.config import DataLoaderConfig
from logger import dataload_logger


# deprecated, remove
def load_labels(processed_dir: str) -> torch.Tensor:
    labels = np.load(Path(processed_dir) / "labels.npy")
    return torch.from_numpy(labels)


# TODO: deprecated, remove
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


# TODO: deprecated, remove
def load_batch_point_clouds_gt(
    k_batches: int | None = None,
    config: DataLoaderConfig = DataLoaderConfig(),
) -> tuple[torch.Tensor, list[dict]]:
    """
    Loads point clouds and their symmetry ground truth from processed_dir.
    Files are matched by name (object hash), not by batch index.

    Returns:
        point_clouds: Tensor (K, P, 3)
        gt_index: list where gt_index[i] = {"name": str, "planes": list[list[float]]}
                  corresponds to point_clouds[i]
    """
    processed_dir = Path(config.symmetry_processed_dir)

    files = sorted(processed_dir.glob("*.npy"))
    if k_batches:
        files = files[:k_batches]

    dataload_logger.info(f"Found {len(files)} files in {processed_dir}")

    with open(processed_dir / "ground_truth.json") as f:
        ground_truth = json.load(f)

    point_clouds_list = []
    gt_index: list[dict] = []

    for f in files:
        name = f.stem
        pc = np.load(f)
        point_clouds_list.append(torch.from_numpy(pc))
        gt_index.append({"name": name, "planes": ground_truth.get(name, [])})

    point_clouds_tensor = torch.stack(point_clouds_list, dim=0)
    dataload_logger.info(f"Loaded {len(point_clouds_list)} point clouds")

    return point_clouds_tensor, gt_index
