from pathlib import Path
import os
import argparse

from dotenv import load_dotenv
from pose6d.preprocessing import (
    extract_scene_instances_pcs,
    extract_scene_frames_pcs,
    save_frame_pcs,
    save_instance_pcs,
    save_pT_version,
)
from pose6d.loader import build_loader
from logger import scripts_extraction_logger as log

"""
This script extract a partial pointcloud using the segmentation mask available on the dataset. 
The expected directory sctructure for a dataset (eg. lmo) is: 
-----
Dataset:
    {data}/lmo

Process data:
    {root}/{dataset_type}/cache/points_pT
-----
data and root can be the same directory. dataset_type is lmo or bpr.
"""


def parse_scene_range(value: str) -> list[int]:
    start, end = value.split("-", 1)
    return list(range(int(start), int(end) + 1))


def main() -> None:
    args = parse_args()

    loader = build_loader(args.dataset_type, args.dataset, args.model_root)

    target_obj_ids = loader.symmetric_obj_ids()

    cache_path = args.root / args.dataset_type / "cache"

    for scene_id in args.scene_ids:
        saved = []  # for lint
        if args.mode == "pT":
            log.info(f"Saving instances of objects (scene {scene_id})")

            instances = extract_scene_instances_pcs(
                loader,
                scene_id,
                list(target_obj_ids),
                args.min_visib,
                **args.outlier_removal_params,
            )
            save_path = (
                cache_path / args.version_name / "points_pT"
                if args.version_name
                else cache_path / "points_pT"
            )

            saved = save_instance_pcs(instances, save_path)
            data_version_name = "" if not args.version_name else args.version_name
            save_pT_version(
                loader,
                scene_id,
                data_version_name,
                0,
                [x.stem for x in saved],
                save_path,
            )
            log.info(f"Saved {len(saved)} object instances (scene {scene_id})")

        if args.mode == "frame":
            log.info(f"Saving frame of scenes (scene {scene_id})")
            frames = extract_scene_frames_pcs(
                loader,
                scene_id,
            )
            saved = save_frame_pcs(frames, cache_path / "points_frames")
            log.info(f"Saved {len(saved)} scene frames (scene {scene_id})")


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

    p.add_argument(
        "--dataset-type",
        "-t",
        type=str,
        default="lmo",
        choices=["lmo", "pbr"],
        help="Which BOP dataset layout to use.",
    )

    p.add_argument(
        "--model-root",
        type=Path,
        help="Object model root, required for --dataset-type bpr (fallback: POSE6D_MODEL_ROOT in .env)",
    )

    p.add_argument("--scene-id", "-s", type=int, default=2)

    p.add_argument(
        "--scene-range",
        type=str,
        default=None,
        help="Inclusive scene id range 'START-END' (e.g. 0-49), processes every scene in one call. Overrides --scene-id.",
    )

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

    p.add_argument(
        "--outlier-removal-params",
        "-rm",
        type=str,
        default=True,
        help="'nb_neighbours, std' tuple for statistical removal. floats.",
    )

    p.add_argument(
        "--version-name",
        "-vn",
        type=str,
        help="If this arguments is not empty, the extracted pT will be saved in a directory containing this name with a .json",
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

    if args.model_root is None:
        env = os.getenv("POSE6D_MODEL_ROOT")
        if env:
            args.model_root = Path(env)
    if args.dataset_type == "bpr" and args.model_root is None:
        p.error(
            "Pass --model-root or set POSE6D_MODEL_ROOT in .env for --dataset-type bpr"
        )

    if args.mode not in ["pT", "frame"]:
        p.error("mode should be pTr or frame")

    if args.outlier_removal_params:
        rm_stats = args.outlier_removal_params.split(",")
        args.outlier_removal_params = {
            "nb_neighbors": int(rm_stats[0].strip()),
            "std": float(rm_stats[1].strip()),
        }

    if args.scene_range:
        args.scene_ids = parse_scene_range(args.scene_range)
    else:
        args.scene_ids = [args.scene_id]

    return args


if __name__ == "__main__":
    main()
