from pathlib import Path
import os
import argparse

from dotenv import load_dotenv
from pose6d.preprocessing import (
    extract_scene_instances_pcs,
    extract_scene_frames_pcs,
    save_frame_pcs,
    save_instance_pcs,
)
from pose6d.config import LMOConfig
from pose6d.loader import LMOLoader

"""
This script extract a partial pointcloud using the segmentation mask available on the dataset. 
The expected directory sctructure for a dataset (eg. lmo) is: 
-----
Dataset:
    {data}/lmo

Process data:
    {root}/lmo/cache/points_pT
-----
data and root can be the same directory. 
"""


def main() -> None:
    args = parse_args()

    config = LMOConfig.from_root(args.dataset)
    loader = LMOLoader(config)

    target_obj_ids = loader.symmetric_obj_ids()
    loader.paths.scene_dir

    cache_path = args.root / "lmo" / "cache"

    saved = []  # for lint
    if args.mode == "pT":
        print("Saving pT")
        instances = extract_scene_instances_pcs(
            loader, args.scene_id, list(target_obj_ids), args.min_visib
        )
        saved = save_instance_pcs(instances, cache_path / "points_pT")

    if args.mode == "frame":
        print("Saving frames")
        frames = extract_scene_frames_pcs(
            loader,
            args.scene_id,
        )
        saved = save_frame_pcs(frames, cache_path / "points_frames")

    print(f"Saved {len(saved)} instances.")


def parse_args() -> argparse.Namespace:
    load_dotenv()

    p = argparse.ArgumentParser(
        description="Extract partial pointclouds from scene info + segmentation masks (pT)."
    )

    p.add_argument(
        "--dataset",
        "-d",
        type=Path,
        help="Dataset path (fallback: POSE6D_DATASET in .env)",
    )

    p.add_argument("--scene-id", "-s", type=int, default=2)

    p.add_argument(
        "--root",
        "-r",
        type=Path,
        help="Root directory for saving processed data (fallback: POSE6D_ROOT in .env)",
    )

    p.add_argument("--min-visib", "-v", type=float, default=0.05)

    p.add_argument("--experiment-name", "-n", type=str, default="scalarfield")

    p.add_argument(
        "--mode",
        "-m",
        type=str,
        default="pT",
        help=f"Modes available are pT and frames.",
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

    if args.mode not in ["pT", "frame"]:
        p.error("mode should be pTr or frame")

    return args


if __name__ == "__main__":
    main()
