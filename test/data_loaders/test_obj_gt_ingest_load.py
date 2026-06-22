# test/data_loaders/test_symmetry_ingestion.py
import os
import json
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

from data_loaders.ingest import ingest_symmetry_dataset
from data_loaders.dataload import load_batch_point_clouds_gt
from data_loaders.config import DataLoaderConfig


FIXTURES_DIR = Path(os.environ["FIXTURES_DIR"])


def test_ingest_symmetry_dataset_output_structure(tmp_path):
    processed_dir = tmp_path / "processed"

    config = DataLoaderConfig(
        symmetry_obj_dir=str(FIXTURES_DIR),
        symmetry_processed_dir=str(processed_dir),
    )

    ingest_symmetry_dataset(config, num_points=8192, flush_every=1)

    # verificar que el .npy se genero con el nombre correcto
    npy_path = processed_dir / "sample_001.npy"
    assert npy_path.exists()

    point_cloud = np.load(npy_path)
    assert point_cloud.shape == (8192, 3)

    # verificar normalizacion: centrado y escala unitaria
    assert np.allclose(point_cloud.mean(axis=0), 0, atol=5e-3)
    assert np.isclose(np.linalg.norm(point_cloud, axis=-1).max(), 1.0, atol=1e-3)

    # verificar que el ground_truth.json se genero con la estructura esperada
    gt_path = processed_dir / "ground_truth.json"
    assert gt_path.exists()

    with open(gt_path) as f:
        ground_truth = json.load(f)

    assert "sample_001" in ground_truth
    planes = ground_truth["sample_001"]
    assert len(planes) == 1
    assert len(planes[0]) == 6  # nx, ny, nz, cx, cy, cz

    # la normal del plano del cubo no deberia cambiar con la normalizacion
    nx, ny, nz, cx, cy, cz = planes[0]
    assert np.isclose(nx, 1.0, atol=1e-6)
    assert np.isclose(ny, 0.0, atol=1e-6)
    assert np.isclose(nz, 0.0, atol=1e-6)
    # el punto del plano deberia seguir cerca del origen tras normalizar
    assert np.isclose(cx, 0.0, atol=1e-2)
    assert np.isclose(cy, 0.0, atol=1e-2)
    assert np.isclose(cz, 0.0, atol=1e-2)


def test_ingest_symmetry_dataset_skips_missing_gt(tmp_path):
    """
    Si un .obj no tiene su .txt correspondiente, debe omitirse sin fallar.
    """
    obj_only_dir = tmp_path / "obj_only"
    obj_only_dir.mkdir()

    # copiar solo el .obj, sin el .txt
    obj_content = (FIXTURES_DIR / "sample_001.obj").read_text()
    (obj_only_dir / "orphan.obj").write_text(obj_content)

    processed_dir = tmp_path / "processed"
    config = DataLoaderConfig(
        symmetry_obj_dir=str(obj_only_dir),
        symmetry_processed_dir=str(processed_dir),
    )

    ingest_symmetry_dataset(config, num_points=512, flush_every=1)

    gt_path = processed_dir / "ground_truth.json"
    with open(gt_path) as f:
        ground_truth = json.load(f)

    assert "orphan" not in ground_truth
    assert not (processed_dir / "orphan.npy").exists()


def test_load_batch_point_clouds_gt_missing_entry_returns_empty(tmp_path):
    """
    Si un .npy existe pero su nombre no esta en ground_truth.json,
    retorna lista vacia en vez de fallar (documentar este comportamiento).
    """
    config = DataLoaderConfig(
        symmetry_obj_dir=str(FIXTURES_DIR), symmetry_processed_dir=str(tmp_path)
    )
    ingest_symmetry_dataset(config, num_points=512, flush_every=1)

    # simular un npy huerfano sin entrada en el json
    np.save(tmp_path / "orphan.npy", np.random.randn(512, 3).astype(np.float32))

    point_clouds, gt_index = load_batch_point_clouds_gt(config=config)

    orphan_entry = next(g for g in gt_index if g["name"] == "orphan")
    assert orphan_entry["planes"] == []


def test_load_batch_point_clouds_gt_index_alignment(tmp_path):
    """
    Verifica que gt_index[i] corresponde exactamente a point_clouds[i].
    """
    config = DataLoaderConfig(
        symmetry_obj_dir=str(FIXTURES_DIR), symmetry_processed_dir=str(tmp_path)
    )
    ingest_symmetry_dataset(config, num_points=512, flush_every=1)

    point_clouds, gt_index = load_batch_point_clouds_gt(config=config)

    assert point_clouds.shape == (1, 512, 3)
    assert len(gt_index) == 1
    assert gt_index[0]["name"] == "sample_001"
    assert len(gt_index[0]["planes"]) == 1
