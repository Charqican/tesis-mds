from dataclasses import dataclass


@dataclass
class FeatureConfig:
    num_views: int = 6
    rot_aug: int = 3
    resolution: int = 224
    fov: float = 60.0
    point_size: float = 0.01
    batch_size: int = 16
    points_per_pixel: int = 10
