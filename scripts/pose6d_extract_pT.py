from pathlib import Path
import numpy as np
from collections import Counter

from pose6d.config import LMOConfig
from pose6d.loader import LMOLoader, InstanceData, instance_uid
from pose6d.geometry_utils import isolate_object_points
from logger import registrate_logger

# Script to extract partial pointclouds from scene info + segmentation masks (pT).
CACHE_ROOT = Path("/mnt/data/dev/dataset/tesis/6dpose/lmo/cache")
POINTS_PT_DIR = CACHE_ROOT / "points_pT"

SYMMETRIC_OBJ_IDS = {10, 11}

MIN_VISIB_FRACT = 0.05


# def load_canonical_diameter(obj_id: int) -> float:
#     path = SYMMETRY_DIR / f"obj_{obj_id:06d}.npz"
#     if not path.exists():
#         raise FileNotFoundError(
#             f"No se encontró el cache de simetría/diámetro canónico para obj_id={obj_id}. "
#             f"Corré primero el pipeline que genera {path}."
#         )
#     data = np.load(path)
#     return float(data["diameter"])


def extract_scene(
    loader: LMOLoader,
    config: LMOConfig,
    scene_id: int,
    target_obj_ids: set[int],
    diameters: dict[int, float],
) -> int:
    """Procesa todas las imágenes de una escena. Devuelve cantidad de instancias guardadas."""
    n_saved = 0
    img_ids = loader.list_image_ids(scene_id)

    POINTS_PT_DIR.mkdir(parents=True, exist_ok=True)

    for img_id in img_ids:
        # TODO: make a private logger for this script
        registrate_logger.info(f"scene={scene_id} img={img_id}")

        K, depth_scale = loader.load_camera(scene_id, img_id)
        instances = loader.load_instances(scene_id, img_id)
        depth = loader.load_depth(scene_id, img_id)

        for inst_idx, instance in enumerate(instances):
            # TODO: add warning
            if instance.obj_id not in target_obj_ids:
                continue

            # TODO: add as a warning
            if (
                instance.visible_fract is not None
                and instance.visible_fract < MIN_VISIB_FRACT
            ):
                registrate_logger.info(
                    f"  skip inst={inst_idx} obj={instance.obj_id} "
                    f"(visib_fract={instance.visible_fract:.3f} < {MIN_VISIB_FRACT})"
                )
                continue

            mask = loader.load_mask_visib(scene_id, img_id, inst_idx)
            if mask is None:
                registrate_logger.info(
                    f"  skip inst={inst_idx} obj={instance.obj_id} (sin máscara)"
                )
                continue

            masked_points = isolate_object_points(depth, mask, K, depth_scale)
            if masked_points.shape[0] == 0:
                registrate_logger.warning(
                    f"  inst={inst_idx} obj={instance.obj_id}: 0 puntos tras aislar, se descarta"
                )
                continue

            uid = instance_uid(scene_id, img_id, instance.obj_id, inst_idx)
            np.savez(
                POINTS_PT_DIR / f"{uid}.npz",
                points=masked_points.astype(np.float32),
            )
            n_saved += 1

    return n_saved


def main():
    root = Path("/mnt/data/dev/dataset/tesis/BOP/lmo/lmo")
    scene_id = 2

    config = LMOConfig.from_root(root)
    loader = LMOLoader(config)

    models_info = loader.load_models_info()
    diameters = {oid: info.diameter for oid, info in models_info.items()}
    target_obj_ids = loader.symmetric_obj_ids()

    registrate_logger.info(f"Symmetric Objects found: {target_obj_ids}")

    n = extract_scene(loader, config, config.default_scene, target_obj_ids, diameters)
    registrate_logger.info(f"Done. {n} instances saved at {POINTS_PT_DIR}")

    # registration Info
    counts = Counter()
    visib_fracts = []
    for f in POINTS_PT_DIR.glob("*.npz"):
        uid = f.stem
        scene_id, img_id, obj_id, inst_idx = loader.parse_instance_uid(uid)
        counts[obj_id] += 1
        instance = loader.load_instances(scene_id, img_id)[inst_idx]
        visib_fracts.append(instance.visible_fract)

    print(counts)  # per object instance
    print(
        f"visib_fract: min={min(visib_fracts):.3f} mean={np.mean(visib_fracts):.3f} max={max(visib_fracts):.3f}"
    )  # visibility statistics across images


if __name__ == "__main__":
    main()
