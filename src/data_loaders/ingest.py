import glob
import numpy as np
import pandas as pd
from pathlib import Path
from data_loaders.config import DataLoaderConfig
from logger import ingestion_logger


def find_parquet_files(raw_dir: Path) -> list[Path]:
    files = sorted(raw_dir.glob("*.parquet"))  # buscar arhivos .parquet
    ingestion_logger.info(f"Found {len(files)} parquet files in {raw_dir}")
    return files


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


def ingest(config: DataLoaderConfig = DataLoaderConfig()) -> None:
    parquet_files = find_parquet_files(Path(config.raw_dir))

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


if __name__ == "__main__":
    ingest()
