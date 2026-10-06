import sys
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

# WARNING: instance diameter needs to be revised as its not clear if we should use the complete object
# INFO: the following code is entirely based on the DEMO script of the original repository
"""
Runs in its own environment (external/dgedi_env). It knows nothing about the project
layout: scripts/pose6d_prepare_data.py writes the list of pT files and prints the exact
command to run, e.g.

    external/dgedi_env/bin/python external/extract_dgedi_features.py \\
        --inputs-list {root}/experiments/{exp}/dgedi_inputs.txt \\
        --output-dir {root}/experiments/{exp}/input --dim 32
"""
try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Dgedi Configuration
CONFIG_PATH = DGEDI_ROOT / "config_dgedi.yaml"
# output dim = first decoder channel of each config_dgedi.yaml mode
MODE_BY_DIM = {32: "single_scale", 64: "multi_scale"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DimMismatchError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract dGeDi features from a list of partial pointclouds (pT)."
    )
    p.add_argument(
        "--inputs-list",
        type=Path,
        required=True,
        help="Text file with one pT .npz path per line.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the features are saved (same file name as the pT).",
    )
    p.add_argument("--dim", type=int, required=True, choices=sorted(MODE_BY_DIM))
    p.add_argument("--batch-size", "-b", type=int, default=None)
    return p.parse_args()


def load_model(mode: str):
    cfg = load_yaml_config(str(CONFIG_PATH))
    model_cfg = dict(cfg[mode]["model_config"])
    model_cfg["weights_path"] = str(DGEDI_ROOT / cfg[mode]["weights_path"])
    return dgedi({"query": model_cfg, "target": model_cfg, "device": DEVICE})


@torch.no_grad()
def process_one(npz_path: Path, model, features_dir: Path, dim: int) -> None:
    data = np.load(npz_path)
    points = data["points"]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    diameter = compute_diameter(pcd)
    normalize_and_center(pcd, diameter)
    features = extract_features(pcd, model, DEVICE)
    if features.shape[-1] != dim:
        raise DimMismatchError(
            f"mode {MODE_BY_DIM[dim]} produced {features.shape[-1]}-dim features, expected {dim}"
        )

    np.savez(features_dir / npz_path.name, features=features.astype(np.float32))

    del pcd, features
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    all_inputs = [
        Path(line.strip())
        for line in args.inputs_list.read_text().splitlines()
        if line.strip()
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pending = [p for p in all_inputs if not (args.output_dir / p.name).exists()]
    print(f"{len(all_inputs)} total, {len(pending)} pending")
    if not pending:
        return

    mode = MODE_BY_DIM[args.dim]
    batch_size = args.batch_size or len(pending)
    n_ok, n_failed = 0, 0

    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start : batch_start + batch_size]
        print(f"Batch {batch_start // batch_size + 1}: {len(batch)} instancias")

        print(f"Device: {DEVICE}, mode: {mode}")
        model = load_model(mode)  # modelo fresco por batch

        iterator = tqdm(batch, unit="instance") if TQDM_AVAILABLE else batch
        for npz_path in iterator:
            try:
                process_one(npz_path, model, args.output_dir, args.dim)
                n_ok += 1
            except DimMismatchError:
                raise
            except Exception as e:
                n_failed += 1
                print(f"[ERROR] {npz_path.name}: {e}")

        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print(f"Done. {n_ok} ok, {n_failed} failed.")


if __name__ == "__main__":
    main()
