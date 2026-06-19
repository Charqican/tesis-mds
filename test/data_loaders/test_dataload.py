import numpy as np
import torch
from pathlib import Path
from data_loaders.dataload import load_batch_point_clouds, load_labels
from data_loaders.config import DataLoaderConfig


def write_fake_batches(
    processed_dir: Path, batch_size: int, num_batches: int, num_points: int = 8192
):
    for i in range(num_batches):
        fake_batch = np.random.randn(batch_size, num_points, 3).astype(np.float32)
        np.save(processed_dir / f"obj{batch_size}_{i}.npy", fake_batch)

    fake_labels = np.arange(batch_size * num_batches)
    np.save(processed_dir / "labels.npy", fake_labels)


def test_load_point_clouds_single_batch(tmp_path):
    write_fake_batches(tmp_path, batch_size=16, num_batches=1)

    point_clouds = load_batch_point_clouds(str(tmp_path))
    assert len(point_clouds) == 16
    assert isinstance(point_clouds[0], torch.Tensor)
    assert point_clouds[0].dtype == torch.float32
    assert point_clouds[0].shape == (8192, 3)


def test_load_point_clouds_multiple_batches(tmp_path):
    write_fake_batches(tmp_path, batch_size=16, num_batches=3)

    point_clouds = load_batch_point_clouds(str(tmp_path))

    # 3 batches de 16 -> 48 objetos totales
    assert len(point_clouds) == 48
    assert point_clouds[0].shape == (8192, 3)


def test_load_point_clouds_wrong_batch_size_returns_empty(tmp_path):
    write_fake_batches(tmp_path, batch_size=16, num_batches=1)

    # buscar con un batch_size que no existe en el directorio

    settings = DataLoaderConfig(batch_size=32)
    point_clouds = load_batch_point_clouds(str(tmp_path), config=settings)
    assert len(point_clouds) == 0


def test_load_labels(tmp_path):
    write_fake_batches(tmp_path, batch_size=16, num_batches=2)

    labels = load_labels(str(tmp_path))

    assert isinstance(labels, torch.Tensor)
    assert labels.shape == (32,)
