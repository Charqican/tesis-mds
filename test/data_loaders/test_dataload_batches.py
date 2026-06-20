import numpy as np
import torch
import pytest
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
    settings = DataLoaderConfig(processed_dir=str(tmp_path))
    point_clouds = load_batch_point_clouds(1, config=settings)

    assert isinstance(point_clouds, torch.Tensor)
    assert point_clouds.shape == (16, 8192, 3)
    assert point_clouds.dtype == torch.float32


def test_load_point_clouds_multiple_batches(tmp_path):
    write_fake_batches(tmp_path, batch_size=16, num_batches=3)

    settings = DataLoaderConfig(processed_dir=str(tmp_path))
    point_clouds = load_batch_point_clouds(3, config=settings)

    assert isinstance(point_clouds, torch.Tensor)
    assert point_clouds.shape == (48, 8192, 3)


def test_load_point_clouds_wrong_batch_size_returns_empty(tmp_path):
    write_fake_batches(tmp_path, batch_size=16, num_batches=1)

    settings = DataLoaderConfig(batch_size=32)
    with pytest.raises(FileNotFoundError):
        point_clouds = load_batch_point_clouds(1, config=settings)
