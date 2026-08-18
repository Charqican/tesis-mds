import sys
from pathlib import Path

DGEDI_ROOT = Path(__file__).parent / "dgedi"
sys.path.insert(0, str(DGEDI_ROOT))

import numpy as np
import open3d as o3d
import torch

from core.dgedi_distilled import dgedi
from utils import (
    load_yaml_config,
    compute_diameter,
    normalize_and_center,
    extract_features,
)

# This scripts extract offline features from dGedi model using the original repository
# The pipeline is based on the demo made available. The main idea is :
# 1. Load the dGedi model
# 2. Preprocess the query object to follow the hypotesis of the original paper.
# 3. Extract the features with the forward wrapper
# The objects in this case are excepcted to be partial pointclouds (pT)

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True

except ImportError:
    TQDM_AVAILABLE = False

POINTS_PT_DIR = Path("/mnt/data/dev/dataset/tesis/6dpose/cache/points_pT")
FEATURES_PT_DIR = Path("/mnt/data/dev/dataset/tesis/6dpose/cache/features_dgedi_pT")

CONFIG_PATH = DGEDI_ROOT / "config_dgedi.yaml"
MODE = "multi_scale"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SKIP_EXISTING = True


def load_model():
    cfg = load_yaml_config(str(CONFIG_PATH))
    model_cfg = dict(cfg[MODE]["model_config"])
    model_cfg["weights_path"] = str(DGEDI_ROOT / cfg[MODE]["weights_path"])
    return dgedi({"query": model_cfg, "target": model_cfg, "device": DEVICE})


def sanity_check_scale(points: np.ndarray, diameter: float, uid: str) -> None:
    sp_ext = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    ratio = sp_ext / diameter
    if not (0.05 < ratio < 20):
        print(
            f"[WARN] {uid}: posible unit missmatch. "
            f"extent = {sp_ext:.3f}, diameter={diameter:.3f}, ratio={ratio:.4f}"
        )


# Deserialize nz object from scene registration
# caclculate features
# serialize features + metadata for training pipeline
def process_one(npz_path: Path, model) -> None:
    data = np.load(npz_path)
    points = data["points"]
    diameter = float(data["diameter"])
    uid = npz_path.stem

    sanity_check_scale(points, diameter, uid)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    normalize_and_center(pcd, diameter)

    features = extract_features(pcd, model, DEVICE)

    FEATURES_PT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        FEATURES_PT_DIR / npz_path.name,
        features=features.astype(np.float32),
        obj_id=int(data["obj_id"]),
        scene_id=int(data["scene_id"]),
        img_id=int(data["img_id"]),
        inst_idx=int(data["inst_idx"]),
    )


def main():
    print(f"Device: {DEVICE}")
    model = load_model()

    all_inputs = sorted(POINTS_PT_DIR.glob("*.npz"))
    if SKIP_EXISTING:
        pending = [p for p in all_inputs if not (FEATURES_PT_DIR / p.name).exists()]
        print(f"{len(all_inputs)} totales, {len(pending)} pendientes")
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

    print(f"Listo. {len(pending) - n_failed} ok, {n_failed} fallidas.")


if __name__ == "__main__":
    main()
