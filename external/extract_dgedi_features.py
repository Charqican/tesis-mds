import sys
from pathlib import Path

# resolve imports
DGEDI_ROOT = Path(__file__).parent / "dgedi"
sys.path.insert(0, str(DGEDI_ROOT))

import numpy as np
import open3d as o3d
import torch

from core.dgedi_distilled import dgedi
from utils import (
    load_yaml_config,
    normalize_and_center,
    extract_features,
    compute_diameter,
)

# INFO: the following code is entirely based on the DEMO script of the original repository
try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Proyect paths
POINTS_PT_DIR = Path("/mnt/data/dev/dataset/tesis/6dpose/lmo/cache/points_pT")
FEATURES_INPUT_DIR = Path(
    "/mnt/data/dev/dataset/tesis/6dpose/lmo/scalarfield/training/input"
)

# Dgedi Configuration
CONFIG_PATH = DGEDI_ROOT / "config_dgedi.yaml"
MODE = "multi_scale"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SKIP_EXISTING = True


def load_model():
    cfg = load_yaml_config(str(CONFIG_PATH))
    model_cfg = dict(cfg[MODE]["model_config"])
    model_cfg["weights_path"] = str(DGEDI_ROOT / cfg[MODE]["weights_path"])
    return dgedi({"query": model_cfg, "target": model_cfg, "device": DEVICE})


def process_one(npz_path: Path, model) -> None:
    data = np.load(npz_path)
    points = data["points"]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    diameter = compute_diameter(pcd)
    normalize_and_center(pcd, diameter)
    features = extract_features(pcd, model, DEVICE)

    FEATURES_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(FEATURES_INPUT_DIR / npz_path.name, features=features.astype(np.float32))


def main():
    print(f"Device: {DEVICE}")
    model = load_model()

    all_inputs = sorted(POINTS_PT_DIR.glob("*.npz"))
    if SKIP_EXISTING:
        pending = [p for p in all_inputs if not (FEATURES_INPUT_DIR / p.name).exists()]
        print(f"{len(all_inputs)} total, {len(pending)} pending")
    else:
        pending = all_inputs

    iterator = tqdm(pending, unit="instance") if TQDM_AVAILABLE else pending

    n_failed = 0
    for npz_path in iterator:
        try:
            process_one(npz_path, model)
        except Exception as e:
            n_failed += 1
            print(f"[ERROR] {npz_path.name}: {e}")

    print(f"Done. {len(pending) - n_failed} ok, {n_failed} Failed.")


if __name__ == "__main__":
    main()
