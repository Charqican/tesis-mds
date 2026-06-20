from feature_extractor.rendering import render_point_cloud
from feature_extractor.config import FeatureConfig


def test_render_point_cloud_shapes(point_cloud, camera_R, camera_T):
    config = FeatureConfig()
    rendered_images, mappings = render_point_cloud(
        point_cloud, camera_R, camera_T, config
    )

    num_rotations = len(
        config.rot_aug if isinstance(config.rot_aug, list) else [0]
    )  # ajustar segun _ROTATION_MAPS
    assert rendered_images.shape[1:] == (config.resolution, config.resolution, 3)
    assert mappings.shape[1:] == (config.resolution, config.resolution)
    assert (mappings != -1).any()  # al menos algunos puntos visibles
