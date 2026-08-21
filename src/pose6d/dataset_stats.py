from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from .loader import LMOLoader
from collections import Counter


@dataclass(frozen=True)
class DatasetSummary:
    scene_id: int
    n_objects_total: int
    n_frames: int
    n_instances_total: int
    symmetric_obj_ids: set[int]
    instances_per_obj: dict[int, int]


def _dataset_summary(loader: LMOLoader, scene_id: int) -> DatasetSummary:
    models_info = loader.load_models_info()
    symmetric_ids = loader.symmetric_obj_ids()

    img_ids = loader.list_image_ids(scene_id)
    instances_per_obj: dict[int, int] = {}
    n_instances_total = 0

    for img_id in img_ids:
        for inst in loader.load_instances(scene_id, img_id):
            instances_per_obj[inst.obj_id] = instances_per_obj.get(inst.obj_id, 0) + 1
            n_instances_total += 1

    return DatasetSummary(
        scene_id=scene_id,
        n_objects_total=len(models_info),
        n_frames=len(img_ids),
        n_instances_total=n_instances_total,
        symmetric_obj_ids=symmetric_ids,
        instances_per_obj=instances_per_obj,
    )


def print_summary_table(summary: DatasetSummary) -> None:
    console = Console()
    table = Table(title=f"LM-O scene {summary.scene_id:06d} — summary")
    table.add_column("Metric")
    table.add_column("Value")

    table.add_row("Total objects in dataset", str(summary.n_objects_total))
    table.add_row("Frames", str(summary.n_frames))
    table.add_row("Total instances", str(summary.n_instances_total))
    table.add_row("Symmetric objects", f"{len(summary.symmetric_obj_ids)}")
    console.print(table)

    obj_table = Table(title="Instances by object")
    obj_table.add_column("obj_id")
    obj_table.add_column("n instances")
    obj_table.add_column("Symmetric")
    for obj_id, count in sorted(summary.instances_per_obj.items()):
        obj_table.add_row(
            str(obj_id),
            str(count),
            "Yes" if obj_id in summary.symmetric_obj_ids else "",
        )
    console.print(obj_table)
