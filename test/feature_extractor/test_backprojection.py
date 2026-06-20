import torch
from feature_extractor.backprojection import aggregate_features
from feature_extractor.config import FeatureConfig


def test_aggregate_features_matches_reference(reference_features):
    assert reference_features.shape[-1] == 384  # DINOv2 small embedding dim
    assert not torch.all(reference_features == 0)
    assert not torch.isnan(reference_features).any()


def test_aggregate_features_shape_and_validity(model_outputs, mappings, point_cloud):
    config = FeatureConfig()
    features = aggregate_features(model_outputs, mappings, point_cloud, config)

    assert features.shape == (len(point_cloud), model_outputs.shape[-1])
    assert not torch.isnan(features).any()
