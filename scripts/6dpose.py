import sys
from pathlib import Path
import json
from pose6d.config import LMOConfig
from pose6d.data_loader import LMOLoader
from pose6d.registrate import register_instance
from pose6d.sanity import compute_sanity_metrics
from pose6d.visualization import visualize, save_registration_plys


def main():
    root = Path("/mnt/data/dev/dataset/tesis/BOP/lmo/lmo")
    scene_id = 2

    config = LMOConfig.from_root(root)
    loader = LMOLoader(config)

    # Listar imágenes disponibles
    available_ids = loader.list_image_ids(scene_id)
    print(f"IDs disponibles: {available_ids[:10]}...")

    # Buscar primera imagen con instancia bien visible
    for img_id in available_ids:
        result = loader.get_best_instance(scene_id, img_id, min_visib=0.5)
        if result is not None:
            inst_idx, inst = result
            print(
                f"\nImagen {img_id}: obj_id={inst.obj_id}, visib_fract={inst.visible_fract}"
            )
            break
    else:
        raise SystemExit(
            "No hay instancias con visibilidad suficiente en ninguna imagen"
        )

    # Registro
    reg_result = register_instance(loader, config, scene_id, img_id, inst_idx)
    print(f"Scene pts: {len(reg_result.scene_pts)}")
    print(
        f"Object pts: {len(reg_result.object_pts) if reg_result.object_pts is not None else 'N/A'}"
    )
    print(f"Posed mesh pts: {len(reg_result.posed_mesh_pts)}")

    # Sanity check
    metrics = compute_sanity_metrics(reg_result)
    print(f"\n{metrics}")

    # Visualización
    visualize(reg_result, method="plotly")


if __name__ == "__main__":
    main()
