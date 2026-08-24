import sys
import os
import argparse
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
# TODO: in case of other dataset the subfolder 'lmo' should be changed
try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Dgedi Configuration
CONFIG_PATH = DGEDI_ROOT / "config_dgedi.yaml"
MODE = "multi_scale"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SKIP_EXISTING = True

# external/ -> project root: find .env if available
ENV_PATH = Path(__file__).parent.parent / ".env"


def load_env_file(path: Path) -> None:
    """.env loader"""
    if not path.exists():
        return

    for line in path.read_text().splitlines():
        line = line.strip()
        # remove comments
        if not line or line.startswith("#") or "=" not in line:
            continue
        # partition to only obtain the first instance of =.
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def parse_args() -> argparse.Namespace:
    load_env_file(ENV_PATH)

    p = argparse.ArgumentParser(
        description="Extract dGeDi features from partial pointclouds (pT)."
    )
    p.add_argument(
        "--root",
        "-r",
        type=Path,
        help="Root directory for processed data (fallback: POSE6D_ROOT in .env). The script will target lmo/cache/points_pT inside root",
    )
    p.add_argument(
        "--experiment-name",
        "-e",
        type=str,
        default="scalarfield",
        help="Name of subfolder inside root. the resulting features will be saved in root/lmo/experiment-name/training/input",
    )
    args = p.parse_args()

    if args.root is None:
        env = os.getenv("POSE6D_ROOT")
        if env:
            args.root = Path(env)
    if args.root is None:
        p.error("Pass --root or set POSE6D_ROOT in .env")

    return args


def load_model():
    cfg = load_yaml_config(str(CONFIG_PATH))
    model_cfg = dict(cfg[MODE]["model_config"])
    model_cfg["weights_path"] = str(DGEDI_ROOT / cfg[MODE]["weights_path"])
    return dgedi({"query": model_cfg, "target": model_cfg, "device": DEVICE})


def process_one(npz_path: Path, model, features_dir: Path) -> None:
    data = np.load(npz_path)
    points = data["points"]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    diameter = compute_diameter(pcd)
    normalize_and_center(pcd, diameter)
    features = extract_features(pcd, model, DEVICE)

    features_dir.mkdir(parents=True, exist_ok=True)
    np.savez(features_dir / npz_path.name, features=features.astype(np.float32))


def main():
    args = parse_args()

    points_pt_dir = args.root / "lmo" / "cache" / "points_pT"
    features_input_dir = args.root / "lmo" / args.experiment_name / "training" / "input"

    print(f"Device: {DEVICE}")
    model = load_model()

    all_inputs = sorted(points_pt_dir.glob("*.npz"))
    if SKIP_EXISTING:
        pending = [p for p in all_inputs if not (features_input_dir / p.name).exists()]
        print(f"{len(all_inputs)} total, {len(pending)} pending")
    else:
        pending = all_inputs

    iterator = tqdm(pending, unit="instance") if TQDM_AVAILABLE else pending

    n_failed = 0
    for npz_path in iterator:
        try:
            process_one(npz_path, model, features_input_dir)
        except Exception as e:
            n_failed += 1
            print(f"[ERROR] {npz_path.name}: {e}")

    print(f"Done. {len(pending) - n_failed} ok, {n_failed} Failed.")


if __name__ == "__main__":
    main()
