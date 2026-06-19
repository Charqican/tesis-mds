from dataclasses import dataclass
from rich.console import Console
from rich.table import Table


@dataclass
class FeatureConfig:
    # --- paramaters ---
    num_views: int = 6
    rot_aug: int = 3
    resolution: int = 224
    fov: float = 60.0
    point_size: float = 0.01
    batch_size: int = 16
    points_per_pixel: int = 10
    max_points: int | None = 1000
    perspective: bool = True

    # --- stratic pipeline state ---
    using_fibonacci_sampling: bool = True
    using_colorized_renders: bool = True
    early_subsampling: bool = True
    using_point_clouds: bool = True

    def print_summary(self) -> None:
        table = Table(title="FeatureConfig")
        table.add_column("Parameter")
        table.add_column("Value")
        for field, value in self.__dict__.items():
            table.add_row(field, str(value))
        Console().print(table)
