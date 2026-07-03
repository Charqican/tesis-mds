import numpy as np
import torch
import pandas as pd
from pathlib import Path
from metrics.invariance import (
    compute_symmetry_invariance_object,
)
from feature_extractor.config import FeatureConfig


def _load_symmetry_planes(
    txt_path: Path,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    lines = txt_path.read_text().strip().splitlines()
    try:
        n_planes = int(lines[0])
    except ValueError:
        print(txt_path)
        raise ValueError(f" en {txt_path}")
    planes = []
    for line in lines[1 : n_planes + 1]:
        parts = line.split()
        normal = torch.tensor([float(parts[1]), float(parts[2]), float(parts[3])])
        midpoint = torch.tensor([float(parts[4]), float(parts[5]), float(parts[6])])
        planes.append((normal, midpoint))
    return planes


def eval_invariance_features(
    config: FeatureConfig,
    point_cloud_path: str | Path,
    feature_path: str | Path,
    gt_path: str | Path,
    model: torch.nn.Module,
    filter_by: str | None = None,
) -> pd.DataFrame:
    """
    Calcula la invarianza de simetria por objeto.

    Returns:
        DataFrame con columnas:
            name:        hash del objeto (sin sufijo filter_by)
            invariance:  distancia L1 promedio sobre planos (float)
            n_planes:    numero de planos de simetria del objeto
    """
    point_cloud_path = Path(point_cloud_path)
    feature_path = Path(feature_path)
    gt_path = Path(gt_path)

    npy_files = sorted(feature_path.glob("*.npy"))
    if filter_by is not None:
        npy_files = [f for f in npy_files if f.stem.endswith(filter_by)]

    if not npy_files:
        raise FileNotFoundError(
            f"No se encontraron .npy en {feature_path}"
            + (f" con sufijo '{filter_by}'" if filter_by else "")
        )

    rows = []
    for npy_file in npy_files:
        base_name = npy_file.stem
        if filter_by and base_name.endswith(filter_by):
            base_name = base_name[: -len(filter_by)]

        pc_file = point_cloud_path / f"{base_name}.npy"
        txt_file = gt_path / f"{base_name}.txt"

        if not pc_file.exists():
            raise FileNotFoundError(
                f"Point cloud no encontrado para '{base_name}': {pc_file}"
            )
        if not txt_file.exists():
            raise FileNotFoundError(
                f"Planos GT no encontrados para '{base_name}': {txt_file}"
            )

        points = torch.from_numpy(np.load(pc_file))
        features = torch.from_numpy(np.load(npy_file))
        planes = _load_symmetry_planes(txt_file)

        rows.append(
            {
                "name": base_name,
                "invariance": compute_symmetry_invariance_object(
                    points, features, planes
                ),
                "n_planes": len(planes),
            }
        )

    return pd.DataFrame(rows)


def agg_eval_invariance_features(
    config: FeatureConfig,
    point_cloud_path: str | Path,
    feature_path: str | Path,
    gt_path: str | Path,
    model: torch.nn.Module,
    filter_by: str | None = None,
) -> float:
    """
    Distancia L1 promedio de invarianza sobre todos los objetos.
    Wrapper de eval_invariance_features que retorna el promedio del DataFrame.
    """
    df = eval_invariance_features(
        config, point_cloud_path, feature_path, gt_path, model, filter_by
    )
    return float(df["invariance"].mean())
