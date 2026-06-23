import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes
import torch

from data_loaders.config import DataLoaderConfig
from data_loaders.utils import find_files
from logger import ingestion_logger


def load_parquet(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    assume the following data model for parquet files:

            [label]         [inputs]
    [index]   str    np.array[list[...]]

    with each input encoding a 8196 point cloud in an array of lists with 3 floats.
    """
    df = pd.read_parquet(path)
    pc = np.stack([np.stack(row) for row in df["inputs"]]).astype(np.float32)
    labels = df["labels"].to_numpy()

    ingestion_logger.debug(f"Loaded {path} | point_clouds={pc.shape}")
    return pc, labels


def _save_separate(
    point_clouds: np.ndarray,  # (K, P, 3)
    labels: np.ndarray,  # (K,)
    output_dir: str,
    batch_size: int,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    num_batches = len(point_clouds) // batch_size
    for i in range(num_batches):
        start, end = i * batch_size, (i + 1) * batch_size
        np.save(
            out / f"obj{batch_size}_{i}.npy", point_clouds[start:end]
        )  # (batch_size, P, 3)

    np.save(out / "labels.npy", labels)
    ingestion_logger.info(
        f"Saved {num_batches} batches of {batch_size} objects to {output_dir}"
    )


def _save_stacked(
    point_clouds: np.ndarray, labels: np.ndarray, output_dir: str
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "point_clouds.npy", point_clouds)
    np.save(out / "labels.npy", labels)
    ingestion_logger.info(
        f"Saved {len(point_clouds)} objects (stacked mode) to {output_dir}"
    )


def ingest_parquet(config: DataLoaderConfig = DataLoaderConfig()) -> None:
    parquet_files = find_files(Path(config.raw_dir), ".parquet", sort=True)
    ingestion_logger.info(
        f"Found {len(parquet_files)} parquet files in {config.raw_dir}"
    )

    results = [load_parquet(p) for p in parquet_files]
    point_clouds = np.concatenate([r[0] for r in results], axis=0)
    labels = np.concatenate([r[1] for r in results], axis=0)

    if config.mode == "separate":
        _save_separate(point_clouds, labels, config.processed_dir, config.batch_size)
    elif config.mode == "stacked":
        _save_stacked(point_clouds, labels, config.processed_dir)
    else:
        raise ValueError(
            f"mode debe ser 'separate' o 'stacked', recibido: {config.mode}"
        )


def _parse_ground_truth(txt_path: Path) -> list[list[float]]:
    """
    Parsea un archivo de ground truth con formato:
        n_planos
        plane nx ny nz cx cy cz
        ...

    Returns:
        lista de planos, cada uno [nx, ny, nz, cx, cy, cz]
    """
    lines = txt_path.read_text().strip().splitlines()
    n_planes = int(lines[0])

    planes = []
    for line in lines[1 : 1 + n_planes]:
        parts = line.split()
        values = [float(v) for v in parts[1:]]
        planes.append(values)

    return planes


def _mesh_to_point_cloud(
    obj_path: Path, num_points: int, device: torch.device
) -> np.ndarray:
    """
    Carga un .obj como mesh y samplea num_points puntos sobre su superficie.
    """
    mesh = load_objs_as_meshes([str(obj_path)], device=device)
    point_cloud = sample_points_from_meshes(mesh, num_points).squeeze(0)
    return point_cloud.cpu().numpy()


def _normalize_point_cloud_and_gt(
    point_cloud: np.ndarray,  # (N, 3)
    planes: list[list[float]],  # [[nx,ny,nz,cx,cy,cz], ...]
) -> tuple[np.ndarray, list[list[float]]]:
    mean = point_cloud.mean(axis=0)
    centered = point_cloud - mean
    scale = np.linalg.norm(centered, axis=-1).max()

    point_cloud_norm = centered / scale

    planes_norm = []
    for nx, ny, nz, cx, cy, cz in planes:
        c_norm = (np.array([cx, cy, cz]) - mean) / scale
        planes_norm.append([nx, ny, nz, *c_norm.tolist()])

    return point_cloud_norm, planes_norm


def ingest_symmetry_dataset(
    config: DataLoaderConfig,
    num_points: int = 8192,
    flush_every: int = 128,
) -> None:
    """
    Export every .obj to a .npy file. Then, parse the gt planes files into a unified json dictionary
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # directories
    obj_dir = Path(config.symmetry_obj_dir)
    processed_dir = Path(config.symmetry_processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    # obj files + ground truth
    obj_files = _find_files(obj_dir, "obj")
    ground_truth: dict[str, list[list[float]]] = {}
    for i, obj_path in enumerate(obj_files):
        name = obj_path.stem
        txt_path = obj_path.with_suffix(".txt")
        if not txt_path.exists():
            ingestion_logger.warning(f"Missing ground truth for {name}, skipping")
            continue
        point_cloud = _mesh_to_point_cloud(obj_path, num_points, device)
        planes = _parse_ground_truth(txt_path)

        point_cloud, planes = _normalize_point_cloud_and_gt(point_cloud, planes)

        np.save(processed_dir / f"{name}.npy", point_cloud)
        ground_truth[name] = planes
        if (i + 1) % flush_every == 0:
            ingestion_logger.info(f"Processed {i + 1}/{len(obj_files)} objects")
    with open(processed_dir / "ground_truth.json", "w") as f:
        json.dump(ground_truth, f)
    ingestion_logger.info(
        f"Done | {len(ground_truth)} objects ingested to {processed_dir}"
    )


if __name__ == "__main__":
    config = DataLoaderConfig()
    ingest_symmetry_dataset(config=config)
