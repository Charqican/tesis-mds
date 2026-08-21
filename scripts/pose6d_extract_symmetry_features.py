from pathlib import Path

import numpy as np

from pose6d.config import LMOConfig
from pose6d.loader import LMOLoader
from pose6d.features import compute_canonical_symmetry_field
from pose6d.geometry_utils import propagate_symmetry_to_target

"""
Script to extract and propagate symmetry features from canonical objects to partial pointclouds. 
"""
# TODO: should be able to pass this paths as arguments
POINTS_PT_DIR = Path("/mnt/data/dev/dataset/tesis/6dpose/lmo/cache/points_pT")
# expected path format: {dataset_name}/{experiment_name}/training/{input | target}
FEATURES_INPUT_DIR = Path(
    "/mnt/data/dev/dataset/tesis/6dpose/lmo/scalarfield/training/input"
)
TARGET_DIR = Path("/mnt/data/dev/dataset/tesis/6dpose/lmo/scalarfield/training/target")


def main():
    root = Path("/mnt/data/dev/dataset/tesis/BOP/lmo/lmo")
    config = LMOConfig.from_root(root)
    loader = LMOLoader(config)

    canonical_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    for pt_path in sorted(POINTS_PT_DIR.rglob("*.npz")):
        uid = pt_path.stem
        scene_id, img_id, obj_id, inst_idx = loader.parse_instance_uid(uid)

        pt_data = np.load(pt_path)
        points = pt_data["points"]

        # obtain instance
        instance = loader.load_instances(scene_id, img_id)[inst_idx]

        if obj_id not in canonical_cache:
            canonical_cache[obj_id] = compute_canonical_symmetry_field(
                config, loader, obj_id
            )
        sample_points, symmetry_scalar_field = canonical_cache[obj_id]

        target = propagate_symmetry_to_target(
            sample_points,
            symmetry_scalar_field,
            points,
            instance.R,
            instance.t,
        )

        np.savez(TARGET_DIR / pt_path.name, target=target.astype(np.float32))

    print(f"Done: {len(list(TARGET_DIR.glob('*.npz')))} instances in {TARGET_DIR}")


if __name__ == "__main__":
    main()
