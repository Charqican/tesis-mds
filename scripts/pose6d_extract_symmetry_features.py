from pathlib import Path
import os
import argparse

import numpy as np
from dotenv import load_dotenv

from pose6d.config import LMOConfig
from pose6d.loader import LMOLoader
from pose6d.features import compute_canonical_symmetry_field

# TODO: should be in preprocessing
from pose6d.geometry_utils import propagate_symmetry_to_target

"""
This script extract a symmetry field using canonical object data, then the fiels is
propagated to an existing partial point cloud using closest neighbour. It is expected 
the following file structure for a dataset (eg. lmo):

-----
Dataset root: 
    {data}/lmo
Partial pointclouds:
    {root}/lmo/cache/points_pT

Extracted features:
    {root}/lmo/{experiment_name}/training/target
-----

The only neccesary parameters are a 'data' root directory containing the dataset, a 'root' 
directory containing the processed data and an 'experiment_name'. 'root' and 'data' can be
the same folder, but it is expected to contain point_pT
"""


def main() -> None:
    args = parse_args()

    config = LMOConfig.from_root(args.dataset)
    loader = LMOLoader(config)

    canonical_cache: dict[int, tuple] = {}
    args.output.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(args.points_pt.rglob("*.npz"))
    if not pt_files:
        raise FileNotFoundError(f"No partial points found in {args.points_pt}.")

    for pt_path in pt_files:
        uid = pt_path.stem
        scene_id, img_id, obj_id, inst_idx = loader.parse_instance_uid(uid)

        pt_data = np.load(pt_path)
        points = pt_data["points"]

        instance = loader.load_instances(scene_id, img_id)[inst_idx]

        if obj_id not in canonical_cache:
            canonical_cache[obj_id] = compute_canonical_symmetry_field(
                config, loader, obj_id
            )
        sample_points, symmetry_scalar_field = canonical_cache[obj_id]

        # Obtain target symmetry field F_sym(pT)
        target = propagate_symmetry_to_target(
            sample_points,
            symmetry_scalar_field,
            points,
            instance.R,
            instance.t,
        )

        np.savez(args.output / pt_path.name, target=target.astype(np.float32))

    n_out = len(list(args.output.glob("*.npz")))
    print(f"Done: {n_out} instances in {args.output}")


def parse_args() -> argparse.Namespace:
    load_dotenv()

    p = argparse.ArgumentParser(
        description="Extract and propagate symmetry features from canonical objects to partial pointclouds."
    )
    p.add_argument(
        "--dataset",
        "-d",
        type=Path,
        help="Path to BOP dataset root (e.g. .../lmo). Fallback: POSE6D_DATASET in .env",
    )
    p.add_argument(
        "--root",
        "-r",
        type=Path,
        help="Root directory in which processed data will be saved. Fallback: POSE6D_ROOT in .env",
    )
    p.add_argument(
        "--experiment-name",
        "-e",
        type=str,
        default="scalarfield",
        help="name given to a subfolder containing the resulting artefacts",
    )

    args = p.parse_args()

    if args.dataset is None:
        env = os.getenv("POSE6D_DATASET")
        if env:
            args.dataset = Path(env)
    if args.dataset is None:
        p.error("Pass --dataset or set POSE6D_DATASET in .env")

    if args.root is None:
        env = os.getenv("POSE6D_ROOT")
        if env:
            args.root = Path(env)
    if args.root is None:
        p.error("Pass --root or set POSE6D_ROOT in .env")

    exp = args.experiment_name

    if args.points_pt is None:
        args.points_pt = args.root / "lmo" / "cache" / "points_pT"
    if args.output is None:
        args.output = args.root / "lmo" / exp / "training" / "target"

    return args


if __name__ == "__main__":
    main()
