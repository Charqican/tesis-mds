from feature_extractor.sampling import sample_fibonacci_views
from feature_extractor.config import FeatureConfig


def test_sample_fibonacci_views_shapes(point_cloud):
    config = FeatureConfig()
    R, T = sample_fibonacci_views(point_cloud, config)

    assert R.shape == (config.num_views, 3, 3)
    assert T.shape == (config.num_views, 3)
